#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import shutil

import argparse
import random

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import Diagnostics

SIM_SENSORS = {
    "SEMANTIC_SENSOR"
}

TASK_SENSORS = {
    "GPS_SENSOR", "COMPASS_SENSOR"
}

# Whether to fail if run files exist
DO_PRESERVE_RUNS = False # TODO sets to true
RUN_FOLDER_MODE = False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        "-r",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )

    parser.add_argument(
        "--exp-config",
        "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--base-config",
        "-b",
        type=str,
        default="/srv/flash1/jye72/projects/embodied-recall/habitat_baselines/config/objectnav/obj_base.on.yaml",
        help="path to universal config (for most of your experiments)"
    )

    parser.add_argument(
        "--ckpt-path",
        "-c",
        default=None,
        type=str,
        help="full path to a ckpt"
    )

    parser.add_argument(
        "--run-id",
        "-i",
        type=int,
        required=False,
        help="running a batch - give run id",
    )

    parser.add_argument(
        "--run-suffix",
        "-s",
        type=str,
        required=False,
        help="Modify run name (for bookkeeping when changes aren't recorded in config)"
    )

    # Exp Admin things
    parser.add_argument('--wipe-only', '-w', dest='wipe_only', action='store_true')
    parser.add_argument('--no-wipe-only', '-nw', dest='wipe_only', action='store_false')
    parser.set_defaults(wipe_only=False)

    parser.add_argument('--viz', '-v', dest='eval_viz', action='store_true')
    parser.add_argument('--no-viz', '-nv', dest='eval_viz', action='store_false')
    parser.set_defaults(eval_viz=False)

    parser.add_argument('--debug', '-d', dest='debug', action='store_true') # FYI, this will truncate scenes
    # Only safe for reproducing minival/small scale runs.
    parser.add_argument('--no-debug', '-nd', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    parser.add_argument('--train-split', '-t', dest='train_split', action='store_true')
    parser.add_argument('--no-train-split', '-nt', dest='train_split', action='store_false')
    parser.set_defaults(train_split=False)

    parser.add_argument('--deterministic', dest='deterministic', action='store_true') # Eval only
    parser.add_argument('--nondeterministic', dest='deterministic', action='store_false')
    parser.set_defaults(deterministic=False)

    parser.add_argument('--gt-semantics', '-gt', dest='gt_semantics', action='store_true')
    parser.add_argument('--no-gt-semantics', '-ngt', dest='gt_semantics', action='store_false')
    parser.set_defaults(gt_semantics=False)

    parser.add_argument('--record-all', '-ra', dest='record_all', action='store_true')
    parser.add_argument('--no-record-all', '-nra', dest='record_all', action='store_false')
    parser.set_defaults(record_all=False)

    parser.add_argument('--skip-log', '-sl', dest='skip_log', action='store_true') # Eval
    parser.add_argument('--no-skip-log', '-nsl', dest='skip_log', action='store_false')
    parser.set_defaults(skip_log=False)

    parser.add_argument('--simple-eval', '-se', dest='simple_eval', action='store_true') # Eval
    parser.add_argument('--no-simple-eval', '-nse', dest='simple_eval', action='store_false')
    parser.set_defaults(simple_eval=False)

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser

def main():
    # Change dir so that no matter where we invoke this script (i.e. manually via CLI or from another script like `eval_cron`), imports are correct
    os.chdir('/srv/flash1/jye72/projects/embodied-recall')
    parser = get_parser()
    args = parser.parse_args()
    run_exp(**vars(args))

def wipe_and_exit(config):
    if os.path.exists(config.TENSORBOARD_DIR):
        print("Removing tensorboard directory...")
        shutil.rmtree(config.TENSORBOARD_DIR, ignore_errors=True)
    if os.path.exists(config.CHECKPOINT_FOLDER):
        print("Removing checkpoint folder...")
        shutil.rmtree(config.CHECKPOINT_FOLDER, ignore_errors=True)
    if os.path.exists(config.LOG_FILE):
        print("Removing log file...")
        shutil.rmtree(config.LOG_FILE, ignore_errors=True)
    exit(0)

def run_exp(
    exp_config: str,
    run_type: str,
    base_config="",
    ckpt_path="",
    eval_viz=False,
    debug=False,
    train_split=False,
    gt_semantics=False,
    run_id=None,
    run_suffix=None,
    wipe_only=False,
    deterministic=False,
    record_all=False,
    skip_log=False,
    simple_eval=False,
    opts=None
) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        ckpt_path: If training, ckpt to resume. If evaluating, ckpt to evaluate.
        run_id: If using slurm batch, run id to prefix.s
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    if run_suffix is not None:
        exp_dir, exp_yaml = os.path.split(exp_config)
        exp_config = os.path.join(exp_dir, run_suffix, exp_yaml)
    config = get_config([base_config, exp_config], opts)

    variant_name = os.path.split(exp_config)[1].split('.')[0]
    config.defrost()

    # If we have a suffix, update the variants appropriately
    if run_suffix is not None:
        if RUN_FOLDER_MODE:
            config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, run_suffix, variant_name)
            config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, run_suffix, variant_name)
            config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, run_suffix, variant_name)
        variant_name = f"{variant_name}-{run_suffix}"
    if not RUN_FOLDER_MODE:
        config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, variant_name)
        config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, variant_name)
        config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, variant_name)
    config.LOG_FILE = os.path.join(config.LOG_FILE, f"{variant_name}.log") # actually a logdir

    if wipe_only:
        wipe_and_exit(config)

    if debug:
        config.DEBUG = True
        config.LOG_INTERVAL = 1
        config.NUM_PROCESSES = 1
        # config.NUM_PROCESSES = 3

    if train_split:
        config.EVAL.SPLIT = "train"

    if deterministic:
        config.EVAL.DETERMINISTIC = True

    config.RL.PPO.POLICY.EVAL_GT_SEMANTICS = gt_semantics

    diagnostic_label = "train_gt_" if train_split else "eval_gt_"
    diagnostic_label = f"{diagnostic_label}{config.RL.PPO.POLICY.EVAL_GT_SEMANTICS}"

    eval_stats_dir = ""

    if run_type == "eval":
        config.TRAINER_NAME = "ppo"
        if not debug:
            config.NUM_PROCESSES = 6
        if skip_log:
            config.NUM_PROCESSES = 1

        map_cfg = config.TASK_CONFIG.TASK.TOP_DOWN_MAP
        # map_cfg.MAP_RESOLUTION = 400
        map_cfg.MAP_RESOLUTION = 1200
        log_diagnostics = []

        eval_stats_dir = osp.join(f'/srv/share/jye72/objectnav_eval/', variant_name)
        if eval_viz:
            config.TEST_EPISODE_COUNT = 30
            config.VIDEO_OPTION = ["disk"]
        else:
            config.VIDEO_OPTION = []
            log_diagnostics = [Diagnostics.basic, Diagnostics.episode_info]
        if record_all:
            # will carry over video option from eval_viz
            eval_stats_dir = osp.join(f'/srv/share/jye72/objectnav_detailed/', variant_name)
            config.TEST_EPISODE_COUNT = 300
            # config.TEST_EPISODE_COUNT = 22

            if train_split:
                config.EVAL.SPLIT = f"train_{config.TEST_EPISODE_COUNT}"
            else:
                config.EVAL.SPLIT = f"val_{config.TEST_EPISODE_COUNT}"

            config.VIDEO_DIR = osp.join("/srv/share/jye72/vis/videos/objectnav_detailed/", variant_name)

            config.TASK_CONFIG.TASK.MEASUREMENTS.append("GOAL_OBJECT_VISIBLE_PIXELS")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("REGION_LEVEL_INFO")

            # Also record probing information
            log_diagnostics = [
                Diagnostics.basic, Diagnostics.internal_activations, Diagnostics.observations,
                Diagnostics.actions, Diagnostics.episode_info, Diagnostics.weights,
                Diagnostics.d2g, Diagnostics.room_cat, Diagnostics.visit_count,
                Diagnostics.collisions_t, Diagnostics.coverage_t, Diagnostics.sge_t
            ]

    # if run_type == "train" or (config.RL.PPO.POLICY.USE_SEMANTICS):
    if run_type == "train" or (config.RL.PPO.POLICY.USE_SEMANTICS and config.RL.PPO.POLICY.EVAL_GT_SEMANTICS):
        # Add necessary supervisory signals
        if config.ENV_NAME == "ExploreThenNavRLEnv" or "SemanticGoalExists" in config.RL.AUX_TASKS.tasks:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("GOAL_OBJECT_VISIBLE_PIXELS")
        if config.RL.COVERAGE_TYPE == "VIEW":
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        train_sensors = config.RL.AUX_TASKS.required_sensors
        train_sim_sensors = list(filter(lambda x: x in SIM_SENSORS, train_sensors))

        # ! FIXME some kind of routing for semantics on gibson
        if not config.RL.POLICY.TRAIN_PRED_SEMANTICS:
            config.SENSORS.extend(train_sim_sensors) # the task cfg sensors are overwritten by this one
        train_task_sensors = list(filter(lambda x: x in TASK_SENSORS, train_sensors))
        config.TASK_CONFIG.TASK.SENSORS.extend(train_task_sensors) # ! i think we're doing this just in case
    if run_type == "eval" and config.RL.PPO.POLICY.USE_SEMANTICS:
        if config.RL.PPO.POLICY.EVAL_GT_SEMANTICS:
            config.TENSORBOARD_DIR = osp.join(config.TENSORBOARD_DIR, "gt_sem")
            config.VIDEO_DIR = osp.join(config.VIDEO_DIR, "gt_sem")
        else:
            config.TENSORBOARD_DIR = osp.join(config.TENSORBOARD_DIR, "pred_sem")
            config.VIDEO_DIR = osp.join(config.VIDEO_DIR, "pred_sem")
            print("Running evaluation with semantic predictions")

    if ckpt_path is not None:
        ckpt_dir, ckpt_file = os.path.split(ckpt_path)
        _, *ckpt_file_others = ckpt_file.split(".")
        ckpt_num = ckpt_file_others[-2]
        eval_stats_dir = osp.join(eval_stats_dir, ckpt_num)
        ckpt_file = ".".join([variant_name, *ckpt_file_others])
        ckpt_path = os.path.join(config.CHECKPOINT_FOLDER, ckpt_file)

    if run_id is None:
        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)
        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
        trainer = trainer_init(config)

        # If not doing multiple runs (with run_id), default behavior is to overwrite
        if run_type == "train":
            if ckpt_path is None:
                ckpt_path = config.RL.POLICY.PRETRAINED_CKPT # Try this one
            if ckpt_path != "":
                ckpt_dir, ckpt_file = os.path.split(ckpt_path)
                ckpt_index = ckpt_file.split('.')[1]
                ckpt = int(ckpt_index)
                start_updates = ckpt * config.CHECKPOINT_INTERVAL + 1
                trainer.train(ckpt_path=ckpt_path, ckpt=ckpt, start_updates=start_updates)
            elif not DO_PRESERVE_RUNS:
                if os.path.exists(config.TENSORBOARD_DIR):
                    print("Removing tensorboard directory...")
                    shutil.rmtree(config.TENSORBOARD_DIR, ignore_errors=True)
                if os.path.exists(config.CHECKPOINT_FOLDER):
                    print("Removing checkpoint folder...")
                    shutil.rmtree(config.CHECKPOINT_FOLDER, ignore_errors=True)
                if os.path.exists(config.LOG_FILE):
                    print("Removing log file...")
                    shutil.rmtree(config.LOG_FILE, ignore_errors=True)
                trainer.train()
            else:
                if os.path.exists(config.TENSORBOARD_DIR) or os.path.exists(config.CHECKPOINT_FOLDER) \
                    or os.path.exists(config.LOG_FILE):
                    print(f"TB dir exists: {os.path.exists(config.TENSORBOARD_DIR)}")
                    print(f"Ckpt dir exists: {os.path.exists(config.CHECKPOINT_FOLDER)}")
                    print(f"Log file exists: {os.path.exists(config.LOG_FILE)}")
                    print("Run artifact exists, please clear manually")
                    exit(1)
                trainer.train()
        else:
            if debug:
                seed = 7
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.random.manual_seed(seed)
                config.defrost()
                config.RANDOM_SEED = seed
                config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
                config.freeze()
                # https://pytorch.org/docs/stable/notes/randomness.html
                # torch.set_deterministic(True)
                # torch.backends.cudnn.benchmark = False

                # torch.backends.cudnn.deterministic = True - encapsulates
            trainer.eval(ckpt_path, log_diagnostics=log_diagnostics, output_dir=eval_stats_dir, label=diagnostic_label, skip_log=skip_log, simple_eval=simple_eval)
        return

    run_prefix = f'run_{run_id}'
    seed = run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    # Exetnds off old modifications
    tb_dir = os.path.join(config.TENSORBOARD_DIR, run_prefix)
    ckpt_dir = os.path.join(config.CHECKPOINT_FOLDER, run_prefix)
    log_dir, log_file = os.path.split(config.LOG_FILE)
    log_file_extended = f"{run_prefix}--{log_file}"
    log_file_path = os.path.join(log_dir, log_file_extended)

    config.TASK_CONFIG.SEED = seed
    config.TENSORBOARD_DIR = tb_dir
    config.CHECKPOINT_FOLDER = ckpt_dir
    config.LOG_FILE = log_file_path

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    if run_type == "train":
        if ckpt_path is None and config.RL.POLICY.PRETRAINED_CKPT == "":
            if DO_PRESERVE_RUNS and (os.path.exists(tb_dir) or os.path.exists(ckpt_dir) or os.path.exists(log_file_extended)):
                print(f"TB dir exists: {os.path.exists(tb_dir)}")
                print(f"Ckpt dir exists: {os.path.exists(ckpt_dir)}")
                print(f"Log file exists: {os.path.exists(log_file_extended)}")
                print("Run artifact exists, please clear manually")
                exit(1)
            else:
                shutil.rmtree(tb_dir, ignore_errors=True)
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                if os.path.exists(log_file_extended):
                    os.remove(log_file_extended)
            trainer.train()
        else: # Resume training from checkpoint
            if ckpt_path is None:
                ckpt_path = config.RL.POLICY.PRETRAINED_CKPT
            # Parse the checkpoint #, calculate num updates, update the config
            ckpt_dir, ckpt_file = os.path.split(ckpt_path)
            ckpt_index = ckpt_file.split('.')[1]
            if osp.exists(ckpt_path):
                true_path = ckpt_path # Use without suffix if full valid path was provided
            else:
                true_path = os.path.join(ckpt_dir, run_prefix, f"{run_prefix}.{ckpt_index}.pth")
            ckpt = int(ckpt_index)
            start_updates = ckpt * config.CHECKPOINT_INTERVAL + 1
            trainer.train(ckpt_path=true_path, ckpt=ckpt, start_updates=start_updates)
    else:
        ckpt_dir, ckpt_file = os.path.split(ckpt_path)
        ckpt_index = ckpt_file.split('.')[1]
        true_path = os.path.join(ckpt_dir, run_prefix, f"{run_prefix}.{ckpt_index}.pth")
        trainer.eval(true_path, log_diagnostics=log_diagnostics, output_dir=eval_stats_dir, label=diagnostic_label)

if __name__ == "__main__":
    main()
