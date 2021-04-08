from typing import Type
import torch.nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry

def get_aux_task_class(aux_task_name: str) -> Type[nn.Module]:
    r"""Return auxiliary task class based on name.

    Args:
        aux_task_name: name of task.

    Returns:
        Type[nn.Module]: aux task class.
    """
    return baseline_registry.get_aux_task(aux_task_name)

def get_aux_task_classes(cfg) -> Type[nn.Module]:
    r"""Return auxiliary task classes based on list of names.

    Args:
        cfg: aux_cfg

    Returns:
        Type[nn.Module]: aux task class list and sensor encoders, if necessary.
    """
    return baseline_registry.get_aux_tasks(cfg)

from habitat_baselines.common.auxiliary_tasks.aux_utils import ACTION_EMBEDDING_DIM, RolloutAuxTask
import habitat_baselines.common.auxiliary_tasks.supervised_auxiliary_tasks
import habitat_baselines.common.auxiliary_tasks.auxiliary_tasks
