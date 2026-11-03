"""Parameters of the agent transforms (action delay, noise, limits)."""
import inspect
import shutil
import json
import torch

from madrona_puzzle_bench_learn import ActionTransform, ObsTransform
from madrona_puzzle_bench import SimManager


class TransformParameters:
    def __init__(self, num_worlds, device):
        self.num_worlds = num_worlds
        self.device = device
        self.obs_transform = None
        self.action_transform = None
        return

    @staticmethod
    def default_params(num_worlds, device):
        transform_params = TransformParameters(num_worlds, device)
        transform_params.from_params(
            p_drop=0.0,
            p_random=0.0,
            p_repeat=0.0,
            delay_buffer_size=1,
            rate_frequency=1,
            rate_frequency_std=1e-8,
            no_jump=False,
            noise_mean=0.0,
            noise_std=1e-8,
        )
        return transform_params

    def from_params(self,
                    p_drop: float,
                    p_random: float,
                    p_repeat: float,
                    delay_buffer_size: int,
                    rate_frequency: int,
                    rate_frequency_std: float,
                    no_jump: bool,
                    noise_mean: float,
                    noise_std: float):
        
        # expand() changes only the strides, so it allocates no new memory.
        p_drop_tensor = torch.tensor([p_drop], device=self.device).expand(self.num_worlds)
        p_random_tensor = torch.tensor([p_random], device=self.device).expand(self.num_worlds)
        p_repeat_tensor = torch.tensor([p_repeat], device=self.device).expand(self.num_worlds)

        rate_frequency_tensor = torch.tensor([float(rate_frequency)], device=self.device).expand(self.num_worlds)
        rate_frequency_std_tensor = torch.tensor([rate_frequency_std], device=self.device).expand(self.num_worlds)

        no_jump_tensor = torch.tensor([no_jump], device=self.device).expand(self.num_worlds)

        action_transform = ActionTransform(num_worlds=self.num_worlds,
                                           delay_buffer_size=delay_buffer_size,
                                           p_drop=p_drop_tensor,
                                           p_random=p_random_tensor,
                                           p_repeat=p_repeat_tensor,
                                           rate_frequency=rate_frequency_tensor,
                                           rate_frequency_std=rate_frequency_std_tensor,
                                           no_jump=no_jump_tensor,
                                           device=self.device)
        
        noise_mean_tensor = torch.tensor([noise_mean], device=self.device).expand(self.num_worlds)
        noise_std_tensor = torch.tensor([noise_std], device=self.device).expand(self.num_worlds)

        obs_transform = ObsTransform(num_worlds=self.num_worlds, 
                                     noise_mean=noise_mean_tensor, 
                                     noise_std=noise_std_tensor,
                                     device=self.device)

        self.action_transform = action_transform
        self.obs_transform = obs_transform
        return

    def from_json(self, json_file: str):
        """Load the transform parameters from a JSON file."""
        with open(json_file, 'r') as f:
            transform_params_json = json.load(f)
            self.from_params(
                p_drop=transform_params_json['action_transform']['p_drop'][0],
                p_random=transform_params_json['action_transform']['p_random'][0],
                p_repeat=transform_params_json['action_transform']['p_repeat'][0],
                delay_buffer_size=transform_params_json['action_transform']['delay_buffer_size'],
                rate_frequency=transform_params_json['action_transform']['rate_frequency'][0],
                rate_frequency_std=transform_params_json['action_transform']['rate_frequency_std'][0],
                no_jump=transform_params_json['action_transform']['no_jump'][0],
                noise_mean=transform_params_json['obs_transform']['noise_mean'][0],
                noise_std=transform_params_json['obs_transform']['noise_std'][0],
            )

    def get_action_transform(self):
        return self.action_transform

    def get_obs_transform(self):
        return self.obs_transform

    def save(self, path):
        file_path = inspect.getfile(self.__class__)
        print(f"Loaded params from {file_path}, saving to {path}")
        shutil.copy(file_path, path)
        return

    def jsonify(self, path):
        info = {
            'action_transform': self.action_transform.jsonify(),
            'obs_transform': self.obs_transform.jsonify()
        }
        j = json.dumps(info, indent=4)
        with open(path, 'w') as f:
            f.write(j)



def _sample_jumps(n, device):
    """Sample jump budgets: 50% from U{10..20}, 50% from U{21..100}"""
    low = torch.randint(10, 21, (n,), device=device).float()
    high = torch.randint(21, 101, (n,), device=device).float()
    mask = torch.rand(n, device=device) < 0.5
    return torch.where(mask, low, high)


def generateLimitationStepFunc(
    sim: SimManager,
    obs: list[torch.Tensor],
    action_transform: ActionTransform,
    obs_transform: ObsTransform,
    dev: torch.device,
    jumps_remaining: torch.Tensor = None,
    jumps_limitation: bool = False):
    def limitation_step():
        actions = sim.action_tensor().to_torch()
        unmodified_actions = actions.clone()
        with torch.no_grad():
            actions[:] = action_transform.apply(actions).to(dev)
            if jumps_remaining is not None and jumps_limitation:
                actions[jumps_remaining[:, 0] <= 0, 3] = 0

        jumped = (actions[:, 3] == 1).float() if jumps_remaining is not None else None

        sim.step()

        if jumps_remaining is not None:
            with torch.no_grad():
                dones = sim.done_tensor().to_torch().squeeze(-1).float()
                jumps_remaining[:, 0] -= jumped
                new_jumps = _sample_jumps(jumps_remaining.shape[0], dev)
                jumps_remaining[:, 0] = torch.where(dones > 0, new_jumps, jumps_remaining[:, 0])

        # Clear the action-delay buffers of the worlds that are done.
        action_transform.reset_delay(sim.done_tensor().to_torch())
        # Restore the unmodified actions (for batch replay).
        actions[:] = unmodified_actions
        # Apply the observation transform.
        with torch.no_grad():
            obs[:] = obs_transform.apply(obs)
    return limitation_step