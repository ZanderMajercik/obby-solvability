from madrona_puzzle_bench_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder, RNDModel,
    EntitySelfAttentionNet,
)

from madrona_puzzle_bench_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from madrona_puzzle_bench_learn.rnn import LSTM

import math
import torch


NUM_LEVEL_TYPES = -1
NUM_ENTITY_TYPES = -1

def setup_obs(sim, enum_counts_tensor, no_level_obs = False, use_onehot = True, separate_entity = False, need_torch_conversion=True, jumps_remaining=None, bin_coord_obs=None):

    global NUM_LEVEL_TYPES
    global NUM_ENTITY_TYPES

    # First define all the observation tensors
    agent_txfm_obs_tensor = sim.agent_txfm_obs_tensor().to_torch() # Scalars
    agent_interact_obs_tensor = sim.agent_interact_obs_tensor().to_torch() # Bool, whether or not grabbing
    agent_level_type_obs_tensor = sim.agent_level_type_obs_tensor().to_torch() # Enum
    agent_exit_obs_tensor = sim.agent_exit_obs_tensor().to_torch() # Scalars
    entity_physics_state_obs_tensor = sim.entity_physics_state_obs_tensor().to_torch() # Scalars
    entity_type_obs_tensor = sim.entity_type_obs_tensor().to_torch() # Enum
    lidar_depth_tensor = sim.lidar_depth_tensor().to_torch() # Scalars
    lidar_hit_type_tensor = sim.lidar_hit_type().to_torch() # Enum (EntityType)
    steps_remaining_obs_tensor = sim.steps_remaining_obs_tensor().to_torch() # Int, used as is

    if no_level_obs:
        agent_level_type_obs_tensor = torch.zeros_like(agent_level_type_obs_tensor)

    NUM_LEVEL_TYPES  = int(enum_counts_tensor[0])
    NUM_ENTITY_TYPES = int(enum_counts_tensor[1])


    # Convert enums to one-hot.
    agent_level_type_obs_tensor_onehot = torch.nn.functional.one_hot(agent_level_type_obs_tensor.to(torch.int64), num_classes=NUM_LEVEL_TYPES)
    entity_type_obs_tensor_onehot = torch.nn.functional.one_hot(entity_type_obs_tensor.to(torch.int64), num_classes=NUM_ENTITY_TYPES)
    lidar_hit_type_tensor_onehot = torch.nn.functional.one_hot(lidar_hit_type_tensor.to(torch.int64), num_classes=NUM_ENTITY_TYPES)

    # Non-entity observations: (batch_size, -1).
    agent_txfm_obs_tensor = agent_txfm_obs_tensor.view(agent_txfm_obs_tensor.shape[0], -1)
    agent_interact_obs_tensor = agent_interact_obs_tensor.view(agent_interact_obs_tensor.shape[0], -1)
    agent_level_type_obs_tensor_onehot = agent_level_type_obs_tensor_onehot.view(agent_level_type_obs_tensor.shape[0], -1)
    agent_exit_obs_tensor = agent_exit_obs_tensor.view(agent_exit_obs_tensor.shape[0], -1)
    lidar_depth_tensor = lidar_depth_tensor.view(lidar_depth_tensor.shape[0], -1)
    lidar_hit_type_tensor_onehot = lidar_hit_type_tensor_onehot.view(lidar_hit_type_tensor.shape[0], -1)
    steps_remaining_obs_tensor = steps_remaining_obs_tensor.view(steps_remaining_obs_tensor.shape[0], -1)

    # bin_coord_obs and jumps_remaining are optional trailing obs and are not
    # supported simultaneously (process_obs disambiguates them positionally;
    # only PBGE uses bin_coord_obs and it never sets jumps_remaining).
    assert not (bin_coord_obs is not None and jumps_remaining is not None), \
        "bin_coord_obs and jumps_remaining obs are not supported together"
    if bin_coord_obs is not None:
        bin_coord_obs = bin_coord_obs.view(bin_coord_obs.shape[0], -1)

    # Entity observations: (batch_size, num_entities, -1).
    entity_physics_state_obs_tensor = entity_physics_state_obs_tensor.view(entity_physics_state_obs_tensor.shape[0], entity_physics_state_obs_tensor.shape[1], -1)
    entity_type_obs_tensor_onehot = entity_type_obs_tensor_onehot.view(entity_type_obs_tensor.shape[0], entity_type_obs_tensor.shape[1], -1)

    scalar_obs = [
        agent_txfm_obs_tensor,
        agent_interact_obs_tensor,
        agent_level_type_obs_tensor_onehot if use_onehot else agent_level_type_obs_tensor.view(agent_level_type_obs_tensor.shape[0], -1),
        agent_exit_obs_tensor,
        lidar_depth_tensor,
        lidar_hit_type_tensor_onehot if use_onehot else lidar_hit_type_tensor.view(lidar_hit_type_tensor.shape[0], -1),
        steps_remaining_obs_tensor,
    ]
    if bin_coord_obs is not None:
        scalar_obs.append(bin_coord_obs)
    obs_tensor = torch.cat(scalar_obs, dim=1)

    entity_tensor = torch.cat([
        entity_physics_state_obs_tensor,
        entity_type_obs_tensor_onehot if use_onehot else entity_type_obs_tensor.view(entity_type_obs_tensor.shape[0], entity_type_obs_tensor.shape[1], -1),
    ], dim=2)

    num_obs_features = obs_tensor.shape[1]
    num_entity_features = entity_tensor.shape[2]

    # Per-world tensors that process_obs takes; the last two carry the
    # use_onehot and separate_entity flags.
    obs_list = [
        agent_txfm_obs_tensor,
        agent_interact_obs_tensor,
        agent_level_type_obs_tensor,
        agent_exit_obs_tensor,
        entity_physics_state_obs_tensor,
        entity_type_obs_tensor,
        lidar_depth_tensor,
        lidar_hit_type_tensor,
        steps_remaining_obs_tensor,
        torch.zeros_like(steps_remaining_obs_tensor) + use_onehot,
        torch.zeros_like(steps_remaining_obs_tensor) + separate_entity,
    ]

    # Optional trailing obs (process_obs reads them in this order via *args):
    # bin_coord_obs (PBGE goal-bin coords; already counted in num_obs_features),
    # then jumps_remaining (1 extra feature, not in obs_tensor above).
    if bin_coord_obs is not None:
        obs_list.append(bin_coord_obs)
    if jumps_remaining is not None:
        obs_list.append(jumps_remaining)

    extra = 1 if jumps_remaining is not None else 0
    if separate_entity:
        return obs_list, num_obs_features + extra, num_entity_features
    else:
        return obs_list, num_obs_features + num_entity_features*entity_tensor.shape[1] + extra

def process_obs(agent_txfm_obs_tensor, agent_interact_obs_tensor, agent_level_type_obs_tensor, agent_exit_obs_tensor, entity_physics_state_obs_tensor, entity_type_obs_tensor, lidar_depth_tensor, lidar_hit_type_tensor, steps_remaining_obs_tensor, use_onehot, separate_entity, *args):
    global NUM_LEVEL_TYPES
    global NUM_ENTITY_TYPES

    # args layout matches setup_obs's obs_list tail:
    # args[0] = bin_coord_obs (None when not threaded; PBGE writes into it)
    # args[1] = jumps_remaining (only present when jumps-remaining obs is enabled)
    bin_coord_obs = args[0] if len(args) >= 1 else None
    jumps_remaining = args[1] if len(args) >= 2 else None

    use_onehot = use_onehot.sum() > 0
    separate_entity = separate_entity.sum() > 0

    agent_level_type_obs_tensor_onehot = torch.nn.functional.one_hot(agent_level_type_obs_tensor.to(torch.int64), num_classes=NUM_LEVEL_TYPES)
    entity_type_obs_tensor_onehot = torch.nn.functional.one_hot(entity_type_obs_tensor.to(torch.int64), num_classes=NUM_ENTITY_TYPES)
    lidar_hit_type_tensor_onehot = torch.nn.functional.one_hot(lidar_hit_type_tensor.to(torch.int64), num_classes=NUM_ENTITY_TYPES)

    # Non-entity observations: (batch_size, -1).
    agent_txfm_obs_tensor = agent_txfm_obs_tensor.view(agent_txfm_obs_tensor.shape[0], -1)
    agent_interact_obs_tensor = agent_interact_obs_tensor.view(agent_interact_obs_tensor.shape[0], -1)
    agent_level_type_obs_tensor_onehot = agent_level_type_obs_tensor_onehot.view(agent_level_type_obs_tensor.shape[0], -1)
    agent_exit_obs_tensor = agent_exit_obs_tensor.view(agent_exit_obs_tensor.shape[0], -1)
    lidar_depth_tensor = lidar_depth_tensor.view(lidar_depth_tensor.shape[0], -1)
    lidar_hit_type_tensor_onehot = lidar_hit_type_tensor_onehot.view(lidar_hit_type_tensor.shape[0], -1)
    steps_remaining_obs_tensor = steps_remaining_obs_tensor.view(steps_remaining_obs_tensor.shape[0], -1)

    if bin_coord_obs is not None:
        bin_coord_obs = bin_coord_obs.view(bin_coord_obs.shape[0], -1)

    # Entity observations: (batch_size, num_entities, -1).
    entity_physics_state_obs_tensor = entity_physics_state_obs_tensor.view(entity_physics_state_obs_tensor.shape[0], entity_physics_state_obs_tensor.shape[1], -1)
    entity_type_obs_tensor_onehot = entity_type_obs_tensor_onehot.view(entity_type_obs_tensor.shape[0], entity_type_obs_tensor.shape[1], -1)

    scalar_obs = [
        agent_txfm_obs_tensor,
        agent_interact_obs_tensor,
        agent_level_type_obs_tensor_onehot if use_onehot else agent_level_type_obs_tensor.view(agent_level_type_obs_tensor.shape[0], -1),
        agent_exit_obs_tensor,
        lidar_depth_tensor,
        lidar_hit_type_tensor_onehot if use_onehot else lidar_hit_type_tensor.view(lidar_hit_type_tensor.shape[0], -1),
        steps_remaining_obs_tensor,
    ]
    if bin_coord_obs is not None:
        scalar_obs.append(bin_coord_obs)
    if jumps_remaining is not None:
        scalar_obs.append(jumps_remaining.view(jumps_remaining.shape[0], -1).float())
    obs_tensor = torch.cat(scalar_obs, dim=1)

    entity_tensor = torch.cat([
        entity_physics_state_obs_tensor,
        entity_type_obs_tensor_onehot if use_onehot else entity_type_obs_tensor.view(entity_type_obs_tensor.shape[0], entity_type_obs_tensor.shape[1], -1),
    ], dim=2)

    combined_tensor = torch.cat([obs_tensor, entity_tensor.view(entity_tensor.shape[0], -1)], dim=1)

    # A degenerate level or a physics blow-up can put NaN, inf or huge values
    # in a position or lidar channel, which overflow the actor to inf logits.
    # Zero NaN/inf and clamp to +-1e4 (normal values are below 1e3).
    def _sanitize(t):
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
        t.clamp_(-1e4, 1e4)
        return t

    if separate_entity:
        return _sanitize(obs_tensor), _sanitize(entity_tensor)
    else:
        return _sanitize(combined_tensor)

def make_policy(num_obs_features, num_entity_features, num_channels, separate_value, action_dims, intrinsic=False, separate_entity=False):
    if separate_entity:
        encoder = BackboneEncoder(
            net = EntitySelfAttentionNet(num_obs_features, num_entity_features, 128, num_channels, 4)
        )
    else:
        encoder = BackboneEncoder(
            net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 3,
            ),
        )

    if separate_value:
        second_encoder = BackboneEncoder(
            net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 3,
            ),
        )
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = encoder,
            critic_encoder = second_encoder,
        )
    else:
        backbone = BackboneShared(
            process_obs = process_obs,
            encoder = encoder,
        )

    if intrinsic:
        # Intrinsic reward module
        critic_intrinsic = LinearLayerCritic(num_channels)
        rnd_model = RNDModel(
            process_obs = process_obs,
            target_net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 4,
            ),
            predictor_net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 1,
            ),
        )
    else:
        critic_intrinsic = None
        rnd_model = None

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor( #outputs the logits of a discrete action distribution
            action_dims, #[2, 8, 3, 2]
            num_channels, # the backbone's output width
        ),
        critic = LinearLayerCritic(num_channels), # scalar value
        critic_intrinsic = critic_intrinsic,
        rnd_model = rnd_model,
    )
