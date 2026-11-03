import typing as tp
from dataclasses import dataclass, asdict

import torch
from parameters.params_transforms import TransformParameters


@dataclass
class Parameter:
    """A slider parameter of a level type."""
    param_name: str  # unique
    min_val: float
    max_val: float
    curr_val: float
    step: float
    distribution: str
    dsl_param: bool  # True if DSL parameter
    enabled: bool

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(param_dict):
        param_name = param_dict["param_name"]
        curr_val = param_dict["curr_val"]
        min_val = param_dict["min_val"]
        max_val = param_dict["max_val"]
        step = param_dict["step"]
        dsl_param = param_dict["dsl_param"]
        distribution = param_dict["distribution"]
        enabled = param_dict["enabled"]
        return Parameter(param_name, min_val, max_val, curr_val,
                         step, distribution, dsl_param, enabled)


@dataclass
class TraceInfo:
    """Success rates over the sweep of a single parameter."""
    param_name: str
    values: list[float]
    success_rates: list[float]

    def to_dict(self):
        return asdict(self)


@dataclass
class TrajectoryVisualization:
    x: list[float]
    y: list[float]
    z: list[float]
    t: list[float]
    reward: list[float]
    value: list[float]
    action: list[float]
    action_dist: list[float]
    success: bool
    block_x: list[float]
    block_y: list[float]
    block_z: list[float]

    def to_dict(self):
        return asdict(self)


@dataclass
class Settings:
    """Backend settings: simulator size, sampling and the current policy."""
    num_worlds: int
    trials_per_level: int
    episode_len: int
    num_samples: int
    seed: int
    transform_params: TransformParameters
    ckpt_path: str
    gpu_sim: bool
    gpu_id: int
    num_channels: int = 256
    separate_values: bool = False
    no_level_obs: bool = False
    policy: tp.Any = None
    device: torch.device = torch.device('cpu')
