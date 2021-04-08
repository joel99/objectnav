#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    DistanceToGoal,
    Success,
    TopDownMap,
)

task_cat2mpcat40 = [
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
]

@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = (self.config.GOAL_SPEC_MAX_VAL - 1,)
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            if len(episode.goals) == 0:
                logger.error(
                    f"No goal specified for episode {episode.episode_id}."
                )
                return None
            if not isinstance(episode.goals[0], ObjectGoal):
                logger.error(
                    f"First goal should be ObjectGoal, episode {episode.episode_id}."
                )
                return None
            category_name = episode.object_category
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            return np.array([episode.goals[0].object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """
    pass

"""
    ObjectNavReward ported from Oleksandr's code
"""

@registry.register_measure
class ObjectNavReward(Measure):
    r"""ObjectNavReward"""

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "objnav_reward"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToGoal.cls_uuid,
                TopDownMap.cls_uuid,
                GoalObjectVisible.cls_uuid,
                Success.cls_uuid,
            ],
        )
        self._metric = 0
        self.step_count = 0
        self._goal_was_seen = False
        self._previous_distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def reward_exploration(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        map_measures = task.measurements.measures[
            TopDownMap.cls_uuid
        ].get_metric()
        if map_measures:
            exploration_ratio = (
                map_measures["fog_of_war_mask"].sum()
                / map_measures["fog_of_war_mask"].size
            )
        else:
            exploration_ratio = 0.0
        # "map": clipped_house_map,
        # "fog_of_war_mask": clipped_fog_of_war_map,
        if not hasattr(self, "_previous_exploration_ratio"):
            self._previous_exploration_ratio = exploration_ratio
        reward += (
            exploration_ratio - self._previous_exploration_ratio
        ) * self._config.EXP_REWARD_COEF
        self._previous_exploration_ratio = exploration_ratio
        return reward

    def reward_distance_to_goal(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        reward += self._previous_distance_to_target - distance_to_target
        self._previous_distance_to_target = distance_to_target
        return reward

    def reward_expl_dt_when_visible(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        goal_object_visible = task.measurements.measures[
            GoalObjectVisible.cls_uuid
        ].get_metric()

        if goal_object_visible > self._config.GOAL_SEEN_THRESHOLD and not self._goal_was_seen:
            self._goal_was_seen = True
            reward += self._config.GOAL_SEEN_REWARD
            distance_to_target = task.measurements.measures[
                DistanceToGoal.cls_uuid
            ].get_metric()
            self._previous_distance_to_target = distance_to_target


        if self._goal_was_seen:
            reward += self.reward_distance_to_goal(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )
        else:
            reward += self.reward_exploration(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )

        return reward

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self.step_count += 1
        reward = self._config.SLACK_REWARD if self.step_count > 20 else 0

        if (
            self._config.MODE
            == "DISTANCE_TO_GOAL_WHEN_VISIBLE_OTHERWISE_EXPLORE"
        ):
            reward += self.reward_expl_dt_when_visible(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )
        elif self._config.MODE == "DISTANCE_TO_GOAL":
            reward += self.reward_distance_to_goal(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )
        elif self._config.MODE == "EXPLORATION":
            reward += self.reward_exploration(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )

        if task.measurements.measures[Success.cls_uuid].get_metric():
            reward += self._config.SUCCESS_REWARD

        self._metric = reward
        print(f"reward_expl_dt_when_visible: {self._metric:.3f} self._goal_was_seen: {self._goal_was_seen} dt {self._previous_distance_to_target:.3f} ")


@registry.register_measure
class ObjectNavSparseReward(Measure):
    r"""Binary success reward, sans shaping"""
    cls_uuid: str = "objnav_sparse_reward"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.cls_uuid,
            [
                Success.cls_uuid,
            ],
        )
        self._metric = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        if task.measurements.measures[Success.cls_uuid].get_metric():
            reward += self._config.SUCCESS_REWARD
        self._metric = reward


# Second duplicated reward to be used for split curricula. (like aux tasks, reward names aren't easily dup-ed)
@registry.register_measure
class ObjectNavSparseRewardA(ObjectNavSparseReward):
    cls_uuid: str = "objnav_sparse_reward_a"


@registry.register_measure
class DistanceToGoalReward(Measure):
    r"""Binary success reward, sans shaping"""
    cls_uuid: str = "d2g_reward"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_distance_to_target = 0
        self._metric = 0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.cls_uuid,
            [
                Success.cls_uuid,
                DistanceToGoal.cls_uuid,
            ],
        )
        self._metric = 0
        self._previous_distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._previous_distance_to_target = distance_to_target
        reward += self._previous_distance_to_target - distance_to_target
        if task.measurements.measures[Success.cls_uuid].get_metric():
            reward += self._config.SUCCESS_REWARD
        self._metric = reward



# Agent and the environment are one and the same... how do we avoid the redundant calculation
@registry.register_measure
class GoalObjectVisible(Measure):
    r"""GoalObjectVisible"""
    cls_uuid = "goal_vis"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.task_cat2mpcat40 = task_cat2mpcat40

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self._metric = 0
        # if "obj_semantic" in observations and "objectgoal" in observations:
        #     obj_semantic = observations["obj_semantic"]
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"]
            # permute tensor to dimension [CHANNEL x HEIGHT X WIDTH]
            idx = self.task_cat2mpcat40[
                observations["objectgoal"][0]
            ]  # task._dataset.category_to_task_category_id[episode.object_category], task._dataset.category_to_scene_annotation_category_id[episode.object_category], observations["objectgoal"][0]

            goal_visible_pixels = (obj_semantic == idx).sum() # Sum over all since we're not batched
            goal_visible_area = goal_visible_pixels / obj_semantic.size
            self._metric = goal_visible_area



@registry.register_measure
class RegionLevelInfo(Measure):
    r"""Region (room) and Level Information for probing agent location and whether they used stairs"""
    cls_uuid = "region_level"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        # self.levels = {}
        self.regions = None

        self._found_any = False
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        annot = self._sim.semantic_annotations()
        # TODO check the containing logic -- we're getting very few hits..
        # self.levels = {l.id: l.aabb for l in annot.levels}

        if not self._found_any and self.regions is not None:
            print(f"Didn't find any regions in {len(self.regions)}")
        self._found_any = False

        self.regions = {r.id: (r.aabb, r.category) for r in annot.regions}
        self._metric = {
            'room_cat': REGION_ANNOTATIONS['no label'],
            # 'level_id': 100
        }
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _is_in(self, loc, aabb):
        og_sizes = aabb.sizes / 2.0
        sizes = [size for size in og_sizes]
        sizes[1] += 0.1 # A little extra vertical buffer room since the agent tis positioned at ground
        return all(
            loc[i] > aabb.center[i] - sizes[i] and \
                loc[i] < aabb.center[i] + sizes[i] for i in range(len(loc))
        )

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        agent_position = self._sim.get_agent_state().position
        room_cat = REGION_ANNOTATIONS['no label']
        for r_id in self.regions:
            if self._is_in(agent_position, self.regions[r_id][0]):
                room_cat = REGION_ANNOTATIONS[self.regions[r_id][1].name()]
                self._found_any = True
                break
        # level = 100
        # for l_id in self.levels:
        #     if self._is_in(agent_position, self.levels[l_id]):
        #         level = l_id
        #         break

        self._metric = {
            'room_cat': room_cat,
            # 'level_id': int(level) # ! level annotations are flaky
        }

# From https://github.com/niessner/Matterport/blob/master/data_organization.md
REGION_ANNOTATIONS = {
    'bathroom': 0,
    'bedroom': 1,
    'closet': 2,
    'dining room': 3,
    'entryway/foyer/lobby': 4, # (should be the front door, not any door)
    'familyroom/lounge': 5, # (should be a room that a family hangs out in, not any area with couches)
    'garage': 6,
    'hallway': 7,
    'library': 8, # (should be room like a library at a university, not an individual study)
    'laundryroom/mudroom': 9, # (place where people do laundry, etc.)
    'kitchen': 10,
    'living room': 11, # (should be the main “showcase” living room in a house, not any area with couches)
    'meetingroom/conferenceroom': 12,
    'lounge': 13, # (any area where people relax in comfy chairs/couches that is not the family room or living room
    'office': 14, # (usually for an individual, or a small set of people)
    'porch/terrace/deck': 15, # (must be outdoors on ground level)
    'rec/game': 16, # (should have recreational objects, like pool table, etc.)
    'stairs': 17,
    'toilet': 18, # (should be a small room with ONLY a toilet)
    'utilityroom/toolroom': 19,
    'tv': 20, # (must have theater-style seating)
    'workout/gym/exercise': 21,
    'outdoor': 22, # areas containing grass, plants, bushes, trees, etc.
    'balcony': 23, # (must be outside and must not be on ground floor)
    'other room': 24, # (it is clearly a room, but the function is not clear)
    'bar': 25,
    'classroom': 26,
    'dining booth': 27,
    'spa/sauna': 28,
    'junk': 29, # (reflections of mirrors, random points floating in space, etc.)
    'no label': 30
}