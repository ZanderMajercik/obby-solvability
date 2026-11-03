#pragma once

#include "sim.hpp"

namespace madPuzzle {

Entity createAgent(Engine &ctx);
void resetAgent(Engine &ctx,
                Vector3 spawn_pos,
                Quat spawn_rot,
                Vector2 spawn_size,
                Vector3 exit_pos,
                Vector3 spawn_lin_vel,
                float maxHealth);

LevelType generateLevel(Engine &ctx, LevelType level_type);
void destroyLevel(Engine &ctx);

}
