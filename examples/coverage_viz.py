#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil

import cv2
import numpy as np
import argparse


import habitat
from habitat.config import Config
from habitat_baselines.config.default import get_config
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.agents.coverage_agents import RandomCoverageAgent
from habitat_baselines.agents.ppo_coverage_agents import PPOAgent

IMAGE_DIR = os.path.join("/srv/share/jye72/vis", "videos")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Use PPO trainer

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map

def get_default_agent_config():
    c = Config()
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    return c

def agent_on_map(agent, task_config, output_dir):
    config = task_config
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    env = SimpleRLEnv(config=config, dataset=dataset)
    env.reset()

    print("Environment creation successful")
    dirname = os.path.join(
        IMAGE_DIR, output_dir
    )
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    for episode in range(10):
        observations = env.reset()
        agent.reset()

        print("Agent running coverage eval.")
        images = []
        step = 0

        total_reward = 0
        while not env.habitat_env.episode_over:
            step += 1
            best_action = agent.act(observations)
            observations, reward, done, info = env.step(best_action)
            im = observations["rgb"]
            top_down_map = draw_top_down_map(
                info, observations["heading"], im.shape[0]
            )
            print("Reward")
            print(reward)
            total_reward += reward
            output_im = np.concatenate((im, top_down_map), axis=1)
            texted_image = cv2.putText(img=np.copy(output_im), text="{} - V: {} R: {}".format(info["coverage"]["reached"], info["coverage"]["visit_count"], total_reward), org=(20,50),
                fontFace=3, fontScale=1, color=(0,0,255), thickness=4)

            images.append(texted_image)

            # benchmark basics
            print("Step {}: {}".format(step, info['coverage']))
        # if reward > 0:
        #     continue
        print("making episode {}{}".format(dirname, episode))
        images_to_video(images, dirname, "%02d" % episode)
        print("Episode finished")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="rgb",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--exp-config", type=str, default="habitat_baselines/config/coverage/ppo_coverage.baseline.yaml"
    )
    args = parser.parse_args()
    config = get_config(args.exp_config)
    agent_config = get_default_agent_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path
    agent_config.HIDDEN_SIZE = config.RL.PPO.hidden_size
    agent_config.POLICY = config.RL.PPO.policy
    agent = PPOAgent(agent_config)

    model_stem = args.model_path.split('/')[-1]
    model_info = model_stem.split('.')[:2]
    run_type = args.exp_config.split('.')[-2]
    output_dir = '{}_{}'.format(run_type, model_info[1])
    agent_on_map(agent, config.TASK_CONFIG, output_dir)


if __name__ == "__main__":
    main()
