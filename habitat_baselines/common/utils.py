#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import numbers
import os
import time
from collections import defaultdict, OrderedDict
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from distutils.version import StrictVersion

import numpy as np
import torch
from torch import Size, Tensor
import torch.nn as nn
from torch.nn import functional as F
from gym.spaces import Box

from habitat import logger
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

import colorsys

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
#     def sample(
#         self, sample_shape: Size = torch.Size()  # noqa: B008
#     ) -> Tensor:
#         return super().sample(sample_shape).unsqueeze(-1)

#     def log_probs(self, actions: Tensor) -> Tensor:
#         return (
#             super()
#             .log_prob(actions.squeeze(-1))
#             .view(actions.size(0), -1)
#             .sum(-1)
#             .unsqueeze(-1)
#         )

#     def mode(self):
#         return self.probs.argmax(dim=-1, keepdim=True)

class CustomCategorical(nn.Module):

    @torch.jit.export
    def act(self, logits, sample: bool = True) -> Dict[str, torch.Tensor]:
        logits = logits.float()

        if sample:
            neg_gumbles = torch.empty_like(logits).exponential_().log_()
            actions = torch.argmax(
                logits - neg_gumbles, dim=logits.dim() - 1, keepdim=True
            )
        else:
            actions = torch.argmax(logits, dim=logits.dim() - 1, keepdim=True)

        log_probs = F.log_softmax(logits, dim=logits.dim() - 1)
        action_log_probs = torch.gather(log_probs, log_probs.dim() - 1, actions)

        return {
            "actions": actions,
            "action_log_probs": action_log_probs.view(-1, 1),
        }

    @torch.jit.export
    def evaluate_actions(self, logits, actions) -> Dict[str, torch.Tensor]:
        logits = logits.float()
        probs = F.softmax(logits, dim=logits.dim() - 1)
        log_probs = F.log_softmax(logits, dim=logits.dim() - 1)

        return {
            "action_log_probs": torch.gather(log_probs, log_probs.dim() - 1, actions),
            "entropy": -(probs * log_probs).sum(log_probs.dim() - 1, keepdim=True),
        }


class CategoricalNet(nn.Module):
    HIDDEN_SIZE = 32
    def __init__(self, num_inputs: int, num_outputs: int, layers: int=1):
        super().__init__()
        if layers == 1:
            self.linear = nn.Linear(num_inputs, num_outputs)
            nn.init.orthogonal_(self.linear.weight, gain=0.01)
            nn.init.constant_(self.linear.bias, 0)
        else:
            self.linear = nn.Sequential(
                nn.Linear(num_inputs, self.HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(self.HIDDEN_SIZE, num_outputs)
            )
            nn.init.orthogonal_(self.linear[0].weight, gain=0.01)
            nn.init.constant_(self.linear[0].bias, 0)

        self.dist = CustomCategorical()

    # def forward(self, x: torch.Tensor) -> CustomFixedCategorical:
    #     x = self.linear(x)
    #     return CustomFixedCategorical(logits=x.float())

    def forward(self, x):
        x = self.linear(x)
        return x


class ResizeCenterCropper(nn.Module):
    def __init__(self, size: Tuple[int, int], channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in trans_keys and observation_space.spaces[key].shape[:len(size)] != size:
                    logger.info("Overwriting CNN input size of %s: %s" % (key, size))
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            image_resize_shortest_edge(
                input, max(self._size), channels_last=self.channels_last
            ),
            self._size,
            channels_last=self.channels_last,
        )


def linear_decay(
    epoch: int, total_num_updates: int = 1.0, final_decay: float = 0.0
) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """

    return (1.0 - final_decay) * (1 - epoch / float(total_num_updates)) + final_decay


def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None, dtype=torch.float,
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    return batch_list(
        observations, device=device, whitelist=[[key] for key in observations[0].keys()], dtype=dtype
    )

def batch_list(
    list_of_info: List[Dict], device: Optional[torch.device] = None, whitelist: List[List[str]] = [], dtype=torch.float
):
    r"""
        Batch potentially nested information. Everything is returned as `dtype`.
        Each item in the whitelist is the sequence of keys for a particuar item of interest.
    """
    batch = defaultdict(list)

    for info in list_of_info:
        for key_seq in whitelist:
            item = info
            for step in key_seq:
                item = item[step]
            batch[key_seq[-1]].append(
                _to_tensor(item).to(device=device, non_blocking=True)
            )

    for item in batch:
        batch[item] = torch.stack(batch[item], dim=0).to(dtype=dtype)

    return dict(**batch)


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(filter(os.path.isfile, glob.glob(checkpoint_folder + "/*")))
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    tag: str,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        if k not in ['coverage.visit_count', 'softspl', 'collisions.count'] and 'reward' not in k:
            metric_strs.append(f"{k}={v:.2f}")

    video_name = f"{tag}-episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(metric_strs)
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def image_resize_shortest_edge(
    img: Tensor, size: int, channels_last: bool = False, mode='area'
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
        mode: interpolation mode as in F.interpolate
    Returns:
        The resized array as a torch tensor.
    """
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    h, w = get_image_height_width(img, channels_last=channels_last)
    if channels_last:
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode=mode
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(
    img: Tensor, size: Union[int, Tuple[int, int]], channels_last: bool = False
) -> Tensor:
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (h, w) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    h, w = get_image_height_width(img, channels_last=channels_last)

    if isinstance(size, int):
        size_tuple: Tuple[int, int] = (int(size), int(size))
    else:
        size_tuple = size
    assert len(size_tuple) == 2, "size should be (h,w) you wish to resize to"
    cropy, cropx = size_tuple

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def get_image_height_width(
    img: Union[Box, np.ndarray, torch.Tensor], channels_last: bool = False
) -> Tuple[int, int]:
    if img.shape is None or len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]
    return h, w


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


# def image_resize_shortest_edge(
#     img, size: int, channels_last: bool = False
# ) -> torch.Tensor:
#     """Resizes an img so that the shortest side is length of size while
#         preserving aspect ratio.

#     Args:
#         img: the array object that needs to be resized (HWC) or (NHWC)
#         size: the size that you want the shortest edge to be resize to
#         channels: a boolean that channel is the last dimension
#     Returns:
#         The resized array as a torch tensor.
#     """
#     img = _to_tensor(img)
#     no_batch_dim = len(img.shape) == 3
#     if len(img.shape) < 3 or len(img.shape) > 5:
#         raise NotImplementedError()
#     if no_batch_dim:
#         img = img.unsqueeze(0)  # Adds a batch dimension
#     if channels_last:
#         h, w = img.shape[-3:-1]
#         if len(img.shape) == 4:
#             # NHWC -> NCHWs
#             img = img.permute(0, 3, 1, 2)
#         else:
#             # NDHWC -> NDCHW
#             img = img.permute(0, 1, 4, 2, 3)
#     else:
#         # ..HW
#         h, w = img.shape[-2:]

#     # Percentage resize
#     scale = size / min(h, w)
#     h = int(h * scale)
#     w = int(w * scale)
#     img = torch.nn.functional.interpolate(img.float(), size=(h, w), mode="area").to(
#         dtype=img.dtype
#     )
#     if channels_last:
#         if len(img.shape) == 4:
#             # NCHW -> NHWC
#             img = img.permute(0, 2, 3, 1)
#         else:
#             # NDCHW -> NDHWC
#             img = img.permute(0, 1, 3, 4, 2)
#     if no_batch_dim:
#         img = img.squeeze(dim=0)  # Removes the batch dimension
#     return img


# def center_crop(img, size: Tuple[int, int], channels_last: bool = False):
#     """Performs a center crop on an image.

#     Args:
#         img: the array object that needs to be resized (either batched or unbatched)
#         size: A sequence (w, h) or a python(int) that you want cropped
#         channels_last: If the channels are the last dimension.
#     Returns:
#         the resized array
#     """
#     if channels_last:
#         # NHWC
#         h, w = img.shape[-3:-1]
#     else:
#         # NCHW
#         h, w = img.shape[-2:]

#     if isinstance(size, numbers.Number):
#         size = (int(size), int(size))
#     assert len(size) == 2, "size should be (h,w) you wish to resize to"
#     cropx, cropy = size

#     startx = w // 2 - (cropx // 2)
#     starty = h // 2 - (cropy // 2)
#     if channels_last:
#         return img[..., starty : starty + cropy, startx : startx + cropx, :]
#     else:
#         return img[..., starty : starty + cropy, startx : startx + cropx]


# def overwrite_gym_box_shape(box: Box, shape) -> Box:
#     if box.shape == shape:
#         return box
#     shape = list(shape) + list(box.shape[len(shape) :])
#     low = box.low if np.isscalar(box.low) else np.min(box.low)
#     high = box.high if np.isscalar(box.high) else np.max(box.high)
#     return Box(low=low, high=high, shape=shape, dtype=box.dtype)


class AttrDict(OrderedDict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


class AvgTime:
    def __init__(self, num_values_to_avg):
        if num_values_to_avg == "inf":
            self._sum = 0
            self._count = 0
        else:
            self.values = deque([], maxlen=num_values_to_avg)

        self.num_values_to_avg = num_values_to_avg

    def append(self, x):
        if self.num_values_to_avg == "inf":
            self._sum += x
            self._count += 1
        else:
            self.values.append(x)

    def __str__(self):
        if self.num_values_to_avg == "inf":
            avg_time = self._sum / self._count
        else:
            avg_time = sum(self.values) / max(1, len(self.values))
        return f"{avg_time:.4f}"


EPS = 1e-5


class TimingContext:
    def __init__(self, timer, key, additive=False, average=None):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None

    def __enter__(self):
        if self._key not in self._timer:
            if self._average is not None:
                self._timer[self._key] = AvgTime(num_values_to_avg=self._average)
            else:
                self._timer[self._key] = 0

        self._time_enter = time.time()

    def __exit__(self, type_, value, traceback):
        time_passed = max(
            time.time() - self._time_enter, EPS
        )  # EPS to prevent div by zero

        if self._additive:
            self._timer[self._key] += time_passed
        elif self._average is not None:
            self._timer[self._key].append(time_passed)
        else:
            self._timer[self._key] = time_passed


class Timing(AttrDict):
    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key):
        return TimingContext(self, key, additive=True)

    def avg_time(self, key, average="inf"):
        return TimingContext(self, key, average=average)

    def __str__(self):
        s = ""
        i = 0
        for key, value in self.items():
            str_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            s += f"{key}: {str_value}"
            if i < len(self) - 1:
                s += ", "
            i += 1
        return s

def is_fp16_supported() -> bool:
    return StrictVersion(torch.__version__) >= StrictVersion("1.6.0")


def is_fp16_autocast_supported() -> bool:
    return StrictVersion(torch.__version__) >= StrictVersion("1.7.1")