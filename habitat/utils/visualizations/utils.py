#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm
import colorsys
from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps

cv2 = try_cv2_import()

def make_rgb_palette(n=40):
    HSV_tuples = [(x*1.0/n, 0.8, 0.8) for x in range(n)]
    RGB_map = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))
    return RGB_map

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view

AUX_ABBREV = {
    "CPCA": "C1",
    "CPCA_A": "C2",
    "CPCA_B": "C4",
    "CPCA_C": "C8",
    "CPCA_D": "C16",
    "CPCA_Weighted": "CWe",
    "InverseDynamicsTask": "ID",
    "TemporalDistanceTask": "TD",
}

def save_semantic_frame(sem_arr, label):
    semantic_map = sem_arr.squeeze()
    colors = make_rgb_palette(42)
    semantic_colors = colors[semantic_map % 42] * 255
    semantic_colors = semantic_colors.astype(np.uint8)
    imageio.imwrite(f"{label}.png", semantic_colors)

def observations_to_image(observation: Dict, info: Dict, reward, weights_output=None, aux_tasks=[]) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        reward: float to append
        weights_output: attention weights for viz

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)

    # TODO add in ground truth semantics as well, maybe?
    if "semantic" in observation:
        semantic_map = observation["semantic"].squeeze()
        colors = make_rgb_palette(45)
        semantic_colors = colors[semantic_map % 45] * 255
        semantic_colors = semantic_colors.astype(np.uint8)

        egocentric_view.append(semantic_colors)
        if "gt_semantic" in observation:
            gt_semantic_map = observation["gt_semantic"].squeeze()
            gt_semantic_colors = colors[gt_semantic_map % 45] * 255
            gt_semantic_colors = gt_semantic_colors.astype(np.uint8)

            # actually, do half half so that we can see seams.
            egocentric_view[-1][:len(gt_semantic_colors)//2] = gt_semantic_colors[:len(gt_semantic_colors) // 2]
            # egocentric_view.append(gt_semantic_colors)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], egocentric_view.shape[0]
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)

    SHOW_ANNOTATIONS = False
    SHOW_ANNOTATIONS = True
    if SHOW_ANNOTATIONS:
        if "coverage" in info:
            frame = cv2.putText(img=frame,
                text="Cov: {}".format(info['coverage']['reached']), org=(20,50),
                fontFace=3, fontScale=1, color=(0,0,255), thickness=4)
            # vid_stats = (info["coverage"]["reached"], info["coverage"]["visit_count"], f'{reward:.2f}')
            # frame = cv2.putText(img=frame,
            #     text="{} - V: {} R: {}".format(*vid_stats), org=(20,50),
            #     fontFace=3, fontScale=1, color=(0,0,255), thickness=4)

        # if "distance_to_goal" in info:
        #     frame = cv2.putText(img=frame,
        #         text="D: {:.2f}".format(info["distance_to_goal"]), org=(20,100),
        #         fontFace=3, fontScale=1, color=(0,0,255), thickness=4
        #     )

        # Test objectgoal exists task
        if "semantic" in observation:
            task_cat2mpcat40 = np.array([
                3,  # ('chair', 2, 0)
                5,  # ('table', 4, 1)
                6,  # ('picture', 5, 2)
                7,  # ('cabinet', 6, 3)
                8,  # ('cushion', 7, 4)
                10,  # ('sofa', 9, 5),
                11,  # ('bed', 10, 6)
                13,  # ('chest_of_drawers', 12, 7),
                14,  # ('plant', 13, 8)
                15,  # ('sink', 14, 9)
                18,  # ('toilet', 17, 10),
                19,  # ('stool', 18, 11),
                20,  # ('towel', 19, 12)
                22,  # ('tv_monitor', 21, 13)
                23,  # ('shower', 22, 14)
                25,  # ('bathtub', 24, 15)
                26,  # ('counter', 25, 16),
                27,  # ('fireplace', 26, 17),
                33,  # ('gym_equipment', 32, 18),
                34,  # ('seating', 33, 19),
                38,  # ('clothes', 37, 20),
                43,  # ('foodstuff', 42, 21),
                44,  # ('stationery', 43, 22),
                45,  # ('fruit', 44, 23),
                46,  # ('plaything', 45, 24),
                47,  # ('hand_tool', 46, 25),
                48,  # ('game_equipment', 47, 26),
                49,  # ('kitchenware', 48, 27)
            ])

            semantic_obs = observation["semantic"] # w x h

            goal_category = task_cat2mpcat40[observation["objectgoal"]]
            matches = (semantic_obs == goal_category).sum() / semantic_obs.size

            frame = cv2.putText(img=frame,
                text=f"SGE: {matches:.2f}", org=(20,150),
                fontFace=3, fontScale=1, color=(0,0,255), thickness=4
            )

    if weights_output is not None and len(aux_tasks) > 1 and False: # * Pass
        # add a strip to the right of the video
        strip_height = egocentric_view.shape[0] # ~256 -> we'll have 5-10 tasks, let's do 24 pixels each
        strip_gap = 24
        strip_width = strip_gap + 12
        strip = np.ones((strip_height, strip_width, 3), dtype=np.uint8) * 255 # white bg

        num_tasks = weights_output.size(0)
        total_height = num_tasks * strip_gap
        offset = int((strip_height - total_height)/2)
        assert offset > 0, "too many aux tasks to visualize"
        for i in range(num_tasks):
            start_height = i * strip_gap + offset
            strength = int(255 * weights_output[i])
            color = np.array([strength, 0, 0])
            if weights_output[i] > 1.001:
                raise Exception(f"weights is {weights_output}, that's too big")
            strip[start_height: start_height + strip_gap] = color

            task_name = AUX_ABBREV.get(aux_tasks[i], aux_tasks[i])
            task_abbrev = task_name[:3]
            cv2.putText(img=strip,
                text=f"{task_abbrev}", org=(2, int(start_height + strip_gap / 2)),
                fontFace=2, fontScale=.4, color=(256, 256, 256), thickness=1)
        frame = np.concatenate((frame, strip), axis=1)
    return frame


def append_text_to_image(image: np.ndarray, text: str):
    r""" Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
