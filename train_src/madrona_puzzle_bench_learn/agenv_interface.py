import torch
from torch.distributions import Normal


class ObsTransform:
    def __init__(self,
                 num_worlds: int,
                 noise_mean: torch.FloatTensor,
                 noise_std: torch.FloatTensor,
                 device=torch.device('cpu')):
        
        self.noise_mean = noise_mean
        self.noise_mean = self.noise_mean.to(device)

        assert not torch.any(noise_std == 0.0), "Normal distribution requires std > 0"
        self.noise_std = noise_std
        self.noise_std = self.noise_std.to(device)

        self.noise_distribution = Normal(self.noise_mean, self.noise_std)
        self.device = device


    def apply(self, obs, obs_clamp=200):
        lidar_depth_obs = obs[6]
        lidar_depth_obs += self.noise_distribution.sample().unsqueeze(1).expand(-1, lidar_depth_obs.shape[1]).to(lidar_depth_obs.device)

        lidar_depth_obs = torch.clamp(lidar_depth_obs, 0, obs_clamp)  # clamp to the lidar range
        obs[6] = lidar_depth_obs
        return obs

    def jsonify(self):
        return {
            'noise_mean': self.noise_mean.tolist(),
            'noise_std': self.noise_std.tolist(),
        }


class ActionTransform:
    def __init__(self,
                 num_worlds: int,
                 delay_buffer_size: int,
                 p_drop: torch.FloatTensor,
                 p_random: torch.FloatTensor,
                 p_repeat: torch.FloatTensor,
                 rate_frequency: torch.FloatTensor,  # one action per step
                 rate_frequency_std: torch.FloatTensor,
                 no_jump: torch.BoolTensor,
                 device,
                 action_dims=(2, 8, 3, 2)):

        assert delay_buffer_size > 0, "Action buffer size must be greater than 0"

        self.num_worlds = num_worlds
        self.action_dims = action_dims
        self.delay_buffer_size = delay_buffer_size

        # Per-step action transforms
        self.p_drop = p_drop
        self.p_drop = self.p_drop.to(device)
        
        self.p_random = p_random
        self.p_random = self.p_random.to(device)

        self.p_repeat = p_repeat
        self.p_repeat = self.p_repeat.to(device)

        assert not torch.any(rate_frequency_std == 0.0), "Rate limiting requires std > 0"
        self.rate_frequency = rate_frequency
        self.rate_frequency = self.rate_frequency.to(device)

        self.rate_frequency_std = rate_frequency_std
        self.rate_frequency_std = self.rate_frequency_std.to(device)

        # a per-world Normal distribution
        self.rate_freq_dist = Normal(self.rate_frequency, self.rate_frequency_std)
        
        # rate-limiting countdown per world
        self.countdown = torch.zeros(num_worlds, dtype=torch.int32, device=device)

        self.action_buffers = torch.zeros(
            (num_worlds, delay_buffer_size, len(action_dims)), dtype=torch.int32, device=device)
        self.prev_actions = torch.zeros((num_worlds, len(action_dims)),
                                        dtype=torch.int32, device=device)
        self.world_indices = torch.arange(self.num_worlds, dtype=torch.int32, device=device)

        self.no_jump = no_jump

        self.device = device

    def apply_delay(self, actions):
        # Take the actions at the front of the buffer.
        ret_actions = self.action_buffers[:, 0]

        # Shift the buffer and append the new actions.
        self.action_buffers = torch.cat(
            [self.action_buffers[:, 1:], actions.clone().unsqueeze(1).to(self.device)], dim=1)

        return ret_actions

    def reset_delay(self, dones):
        # Worlds that are done get null actions in the buffer.
        self.action_buffers[torch.where(dones)] = 0

    def apply_drop(self, actions):
        ret_actions = actions
        drop_mask = torch.rand(actions.shape[0], device=self.device) < self.p_drop
        ret_actions[drop_mask] = 0
        return ret_actions

    def apply_repeat(self, actions):
        ret_actions = actions
        repeat_mask = torch.rand(actions.shape[0], device=self.device) < self.p_repeat
        ret_actions[repeat_mask] = self.prev_actions[repeat_mask]
        return ret_actions

    def apply_random(self, actions):
        ret_actions = actions.clone()
        random_actions = torch.rand(actions.shape, device=self.device)
        for i, dim in enumerate(self.action_dims):
            random_actions[:, i] *= (dim - 1)
            random_actions[:, i] = torch.round(random_actions[:, i])
        random_actions = random_actions.int()
        random_mask = torch.rand(actions.shape[0], device=self.device) < self.p_random
        ret_actions[random_mask] = random_actions[random_mask]

        # Keep the move / no-move action.
        ret_actions[:, 0] = actions[:, 0]
        # Keep the jump / no-jump action.
        ret_actions[:, 3] = actions[:, 3]

        return ret_actions

    def apply_rate_limit(self, actions):
        ret_actions = actions
        # Worlds whose countdown is 0 resample their action.
        rate_limit_mask = self.countdown == 0
        self.countdown[rate_limit_mask] = self.sample_rate_frequency()[rate_limit_mask]

        # The other worlds repeat their previous action.
        ret_actions[~rate_limit_mask] = self.prev_actions[~rate_limit_mask]

        # Count the countdowns down.
        self.countdown -= 1
        return ret_actions

    def apply_no_jump(self, actions):
        """ Set the jump action to 0 (no jump) """
        ret_actions = actions
        no_jump_mask = self.no_jump
        ret_actions[no_jump_mask, 3] = 0
        return ret_actions

    def apply(self, actions):
        delayed_actions = self.apply_delay(actions)
        repeated_actions = self.apply_repeat(delayed_actions)
        rate_limited_actions = self.apply_rate_limit(repeated_actions)
        dropped_actions = self.apply_drop(rate_limited_actions)
        randomized_actions = self.apply_random(dropped_actions)
        no_jump_actions = self.apply_no_jump(randomized_actions)
        self.prev_actions = no_jump_actions
        return no_jump_actions

    def sample_rate_frequency(self):
        rate_freq = torch.round(self.rate_freq_dist.sample()).int()
        rate_freq = torch.max(rate_freq, torch.ones_like(rate_freq))
        return rate_freq

    def jsonify(self):
        return {
            'action_dims': self.action_dims,
            'delay_buffer_size': self.delay_buffer_size,
            'p_drop': self.p_drop.tolist(),
            'p_random': self.p_random.tolist(),
            'p_repeat': self.p_repeat.tolist(),
            'rate_frequency': self.rate_frequency.tolist(),
            'rate_frequency_std': self.rate_frequency_std.tolist(),
            'no_jump': self.no_jump.tolist(),
        }
