#pragma once

#include <cstdint>

#include "types.hpp"

namespace madPuzzle {

enum class SimFlags : uint32_t {
    Default                              = 0,
    UseFixedWorld                        = 1 << 0,
    IgnoreEpisodeLength                  = 1 << 1,
    UseNormalizedAABBObs                 = 1 << 2,
    UnlockAndRandomizeAgentOrientation   = 1 << 3,
    NoActionDelay                        = 1 << 4,
    DisablePhysObs                       = 1 << 5,
    DisableStepObs                       = 1 << 6,
    DisableLidar                         = 1 << 7
};

enum class RewardMode : uint32_t {
    Dense1,
    DenseNoGoal,
    Dense1Traj,
    Dense2,
    PerLevel,
    EndOnly,
    LavaButton,
    PerGoal,
    HandTunedTraj,
    PolicyBasedGoExplore // sparse waypoint reward (sparseRewardSystem)
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);
inline SimFlags operator|(SimFlags lhs, LevelType rhs);

inline LevelType extractLevelType(SimFlags flags);

}

#include "sim_flags.inl"