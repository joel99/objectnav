#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.DEBUG = False # Debug all the things

# task config can be a list of conifgs like "A.yaml,B.yaml"
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.MOCK_OBJECTNAV = False # Hardcode bypass to fake semantics and objectgoal in env that doesn't have it.
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = -1
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.FORCE_BLIND_POLICY = False
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.DETERMINISTIC = False

_C.EVAL.PROJECT_OUT = -1 # Project time to a specific timestep. It's not quite straightforwrd though, there's some scaling factor.
_C.EVAL.PROJECTION_PATH = '/srv/share/jye72/base-full_timesteps.pth' # Project time to a specific timestep. It's not quite straightforwrd though, there's some scaling factor.

_C.EVAL.restrict_gps = True

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.fp16_mode = "off" # off, autocast, mixed

_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 2.5
_C.RL.SLACK_REWARD = -0.01
_C.RL.COVERAGE_REWARD = 0.25
_C.RL.COVERAGE_ATTENUATION = 0.99
_C.RL.COVERAGE_VISIT_EXP = 1
_C.RL.COVERAGE_BONUS_SCALE = 0.5 # We see coverages of around 5-10
_C.RL.COVERAGE_FALLOFF_RADIUS = 2.0
_C.RL.COVERAGE_TYPE = "VISIT" # Or "FOG_OF_WAR"
_C.RL.EXPLORE_GOAL_SEEN_THRESHOLD = 0.05 # Transition for ExploreThenNav
_C.RL.POLICIES = ["none"] # specifies reward measures used to train policy (actor + critic) heads
# the key named corresponds to the reward (metric) that pair uses
# Note that the first key will have the env reward added to it
# "none" is also a special keyword, indicating no rewards are used
# e.g. if just "none", we default to just the RL env (the baseline)
# use "ExploreRLReward" and "ObjectNavReward" to have a slack-driven exploration, and then objectnav.
_C.RL.REWARD_FUSION = CN()
_C.RL.REWARD_FUSION.STRATEGY = "SUM" # sum with reward -> might as well put it all into the env
# SUM - add rewards, i.e. equivalent to having a single env.
# RAMP and SPLIT both are intended for 2 rewards.
# If there are more than 2, they will be grouped into [0, l-2], {l-1}
# RAMP - phase between 2 rewards
# SPLIT - two sets of policy (actor/critic) heads.

_C.RL.REWARD_FUSION.ENV_ON_ALL = True # Apply env reward everywhere
# ^ a quick hack to only apply slack to first reward

_C.RL.REWARD_FUSION.RAMP = CN() # only supports 2
_C.RL.REWARD_FUSION.RAMP.START = 0.0 # `count_steps`
_C.RL.REWARD_FUSION.RAMP.END = 5.0e7

_C.RL.REWARD_FUSION.SPLIT = CN()
_C.RL.REWARD_FUSION.SPLIT.TRANSITION = 3e7 # change from following policy 1 to policy 2
_C.RL.REWARD_FUSION.SPLIT.IMPORTANCE_WEIGHT = False # add importance weighting to target policy updates
# ^ only implemented for use_gae = true

# -----------------------------------------------------------------------------
# POLICY CONFIG
# -----------------------------------------------------------------------------
_C.RL.POLICY = CN()
_C.RL.POLICY.name = "PointNavBaselinePolicy"
_C.RL.POLICY.PRETRAINED_CKPT = "" # Will load in ckpt (assuming identical arch). Used as quick hack for transplanting reward
_C.RL.POLICY.TRAIN_PRED_SEMANTICS = False
_C.RL.POLICY.FULL_VISION = False # Hack to load the right rednet.
# -----------------------------------------------------------------------------
# OBS_TRANSFORMS CONFIG
# -----------------------------------------------------------------------------
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = tuple()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER.WIDTH = 256
_C.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.WIDTH = 512
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.SENSOR_UUIDS = list()
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.WIDTH = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.FOV = 180
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.PARAMS = (0.2, 0.2, 0.2)
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.SENSOR_UUIDS = list()
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE = CN()
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE.WIDTH = 256
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE.SENSOR_UUIDS = list()

# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.aux_loss_coef = 1.0
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50

# Rollout
_C.RL.PPO.ROLLOUT = CN()
_C.RL.PPO.ROLLOUT.METRICS = [] # 'reached', 'visit_count', 'goal_vis']

# Split policy clamps
_C.RL.PPO.SPLIT_IW_BOUNDS = [0.01, 1.0]

# Policy
_C.RL.PPO.policy = "BASELINE" # Legacy
_C.RL.PPO.POLICY = CN()
_C.RL.PPO.POLICY.name = "BASELINE" # This is the one that's actually used.
_C.RL.PPO.POLICY.jit = False # JIT entire policy if possible
_C.RL.PPO.POLICY.FULL_RESNET = False

_C.RL.PPO.POLICY.use_mean_and_var = False
_C.RL.PPO.POLICY.pretrained_encoder = False
_C.RL.PPO.POLICY.pretrained_weights = "/srv/share/ewijmans3/resnet-18-mp3d-rgbd-100m.pth"
_C.RL.PPO.POLICY.use_cuda_streams = False
_C.RL.PPO.POLICY.restrict_gps = True # Legacy
_C.RL.PPO.POLICY.embed_actions = False
_C.RL.PPO.POLICY.embed_sge = False # feature engineering, yay
_C.RL.PPO.POLICY.input_drop = 0.1
_C.RL.PPO.POLICY.output_drop = 0.0 # dropout from after core
_C.RL.PPO.POLICY.midlevel_medium = 'depth_zbuffer' # "depth_zbuffer"
_C.RL.PPO.POLICY.ACTOR_HEAD_LAYERS = 1
_C.RL.PPO.POLICY.CRITIC_HEAD_LAYERS = 1

_C.RL.PPO.POLICY.DOUBLE_PREPROCESS_BUG = True # ! This should be flagged off going forward.

_C.RL.PPO.CURIOSITY = CN() # ! not supported for single beliefs
_C.RL.PPO.CURIOSITY.USE_CURIOSITY = False
_C.RL.PPO.CURIOSITY.REWARD_SCALE = 0.1
_C.RL.PPO.CURIOSITY.LOSS_SCALE = 0.1
_C.RL.PPO.CURIOSITY.BLOCK_ENCODER_GRADIENTS = True # No gradients back to e.g. CNN, agent
_C.RL.PPO.CURIOSITY.USE_INVERSE_SPACE = False
_C.RL.PPO.CURIOSITY.USE_BELIEF = False # Is curiosity conditioned on belief? (episodic curiosity)
_C.RL.PPO.CURIOSITY.INVERSE_BETA = 0.8 # Tradeoff inverse loss and forward loss
_C.RL.PPO.CURIOSITY.HIDDEN_SIZE = 256 # Pathak's default hidden size # ! Not used after we introduced belief! Sharing one hidden size
_C.RL.PPO.CURIOSITY.VISION_KEY = "rgbd" # highly prefer "semantic" if available, since it'll have simpler statistics that we also care about more

# Which encoders to feed into our beliefs. "all", "rgb", "rgbd", "semantic". Whaa... the modules aren't even gonna be the same size at all...s
_C.RL.PPO.POLICY.BELIEFS = CN()
_C.RL.PPO.POLICY.BELIEFS.NUM_BELIEFS = -1 # Default to num aux tasks. They'll all have the same hidden size, just not the
# ! ^ Not fully implemented

# Map to encoders to use (rgb, d, semantic)
# (encoders should be registered..?)
# should be length 1 (concat all available features) or length NUM_BELIEFS.
# "all" means to concat all available features.
# Supports: "all", "semantic", "rgbd"
_C.RL.PPO.POLICY.BELIEFS.ENCODERS = ["rgbd"]

# Map task index to belief index. Should be a list of len(tasks), with elements in range(NUM_BELIEFS).
# If empty, will default to range(len(tasks)).
# NOTE. NUM_BELIEFS check is not asserted. Be careful.
# Originally implemented to investigated SGE
_C.RL.PPO.POLICY.BELIEFS.AUX_MAP = []
_C.RL.PPO.POLICY.BELIEFS.OBS_KEY = "rgbd" # Used for fusion
_C.RL.PPO.POLICY.BELIEFS.POLICY_INDEX = 0 # Used for comms policy

_C.RL.PPO.POLICY.USE_SEMANTICS = False # Feed semantic to policy.
_C.RL.PPO.POLICY.EVAL_GT_SEMANTICS = False # Experimental - we'll keep semantics on at test as well, just to see. Otherwise, default to rednet ckpt (specified below)
_C.RL.PPO.POLICY.EVAL_SEMANTICS_CKPT = "/srv/share/jye72/rednet/rednet_semmap_mp3d_tuned.pth"
_C.RL.PPO.POLICY.EVAL_SEMANTICS_STABILIZE = False # Squash and upsample
_C.RL.PPO.POLICY.EVAL_SEMANTICS_MUTE = False # Set to void

# Auxiliary Tasks
_C.RL.AUX_TASKS = CN()
_C.RL.AUX_TASKS.tasks = []

_C.RL.AUX_TASKS.required_sensors = []
_C.RL.AUX_TASKS.distribution = "uniform" # one-hot, TODO gaussian
_C.RL.AUX_TASKS.entropy_coef = 0.0

_C.RL.AUX_TASKS.InverseDynamicsTask = CN()
_C.RL.AUX_TASKS.InverseDynamicsTask.loss_factor = 0.01
_C.RL.AUX_TASKS.InverseDynamicsTask.subsample_rate = 0.1

_C.RL.AUX_TASKS.TemporalDistanceTask = CN()
_C.RL.AUX_TASKS.TemporalDistanceTask.loss_factor = 0.1
_C.RL.AUX_TASKS.TemporalDistanceTask.num_pairs = 1 # in lieu of subsample rate

_C.RL.AUX_TASKS.CPCA_Single = CN()
_C.RL.AUX_TASKS.CPCA_Single.loss_factor = 0.05
_C.RL.AUX_TASKS.CPCA_Single.num_steps = 8
_C.RL.AUX_TASKS.CPCA_Single.subsample_rate = 0.2

_C.RL.AUX_TASKS.CPCA_Single_A = _C.RL.AUX_TASKS.CPCA_Single.clone()
_C.RL.AUX_TASKS.CPCA_Single_A.num_steps = 2

_C.RL.AUX_TASKS.CPCA_Single_B = _C.RL.AUX_TASKS.CPCA_Single.clone()
_C.RL.AUX_TASKS.CPCA_Single_B.num_steps = 4

_C.RL.AUX_TASKS.CPCA_Single_C = _C.RL.AUX_TASKS.CPCA_Single.clone()
_C.RL.AUX_TASKS.CPCA_Single_C.num_steps = 8

_C.RL.AUX_TASKS.CPCA_Single_D = _C.RL.AUX_TASKS.CPCA_Single.clone()
_C.RL.AUX_TASKS.CPCA_Single_D.num_steps = 16

_C.RL.AUX_TASKS.CPCA = CN()
_C.RL.AUX_TASKS.CPCA.loss_factor = 0.05
_C.RL.AUX_TASKS.CPCA.num_steps = 1
_C.RL.AUX_TASKS.CPCA.subsample_rate = 0.2
_C.RL.AUX_TASKS.CPCA.sample = "random"
_C.RL.AUX_TASKS.CPCA.dropout = 0.0

_C.RL.AUX_TASKS.CPCA_A = _C.RL.AUX_TASKS.CPCA.clone()
_C.RL.AUX_TASKS.CPCA_A.num_steps = 2

_C.RL.AUX_TASKS.CPCA_B = _C.RL.AUX_TASKS.CPCA.clone()
_C.RL.AUX_TASKS.CPCA_B.num_steps = 4

_C.RL.AUX_TASKS.CPCA_C = _C.RL.AUX_TASKS.CPCA.clone()
_C.RL.AUX_TASKS.CPCA_C.num_steps = 8

_C.RL.AUX_TASKS.CPCA_D = _C.RL.AUX_TASKS.CPCA.clone()
_C.RL.AUX_TASKS.CPCA_D.num_steps = 16

_C.RL.AUX_TASKS.CPCA_Weighted = CN()
_C.RL.AUX_TASKS.CPCA_Weighted.loss_factor = 0.05
_C.RL.AUX_TASKS.CPCA_Weighted.subsample_rate = 0.2

_C.RL.AUX_TASKS.SemanticGoalExists = CN()
_C.RL.AUX_TASKS.SemanticGoalExists.loss_factor = 0.5
_C.RL.AUX_TASKS.SemanticGoalExists.subsample_rate = 0.2
_C.RL.AUX_TASKS.SemanticGoalExists.threshold = 0.01 # detection threshold. Approx a 10 x 10 in 256 x 256

_C.RL.AUX_TASKS.GID = CN()
_C.RL.AUX_TASKS.GID.loss_factor = 0.2
_C.RL.AUX_TASKS.GID.num_steps = 4
_C.RL.AUX_TASKS.GID.subsample_rate = 0.2

_C.RL.AUX_TASKS.ActionDist = CN()
_C.RL.AUX_TASKS.ActionDist.loss_factor = 0.2
_C.RL.AUX_TASKS.ActionDist.num_steps = 4
_C.RL.AUX_TASKS.ActionDist.subsample_rate = 0.2

_C.RL.AUX_TASKS.ActionDist_A = _C.RL.AUX_TASKS.ActionDist.clone()
_C.RL.AUX_TASKS.ActionDist_A.num_steps = 2

_C.RL.AUX_TASKS.CoveragePrediction = CN()
_C.RL.AUX_TASKS.CoveragePrediction.loss_factor = 0.1
_C.RL.AUX_TASKS.CoveragePrediction.num_steps = 16
_C.RL.AUX_TASKS.CoveragePrediction.subsample_rate = 0.2
_C.RL.AUX_TASKS.CoveragePrediction.key = "reached"
_C.RL.AUX_TASKS.CoveragePrediction.regression = True
_C.RL.AUX_TASKS.CoveragePrediction.hidden_size = 16

_C.RL.AUX_TASKS.PBL = CN()
_C.RL.AUX_TASKS.PBL.loss_factor = 0.15
_C.RL.AUX_TASKS.PBL.num_steps = 1
_C.RL.AUX_TASKS.PBL.subsample_rate = 0.2
_C.RL.AUX_TASKS.PBL.sample = "random"

_C.RL.AUX_TASKS.Dummy = CN()

_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.hidden_size = 512
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "NCCL"
_C.RL.DDPPO.rnn_type = "LSTM"
_C.RL.DDPPO.num_recurrent_layers = 2
_C.RL.DDPPO.backbone = "resnet50"
_C.RL.DDPPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Loads just the visual encoder backbone weights
_C.RL.DDPPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.DDPPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config
