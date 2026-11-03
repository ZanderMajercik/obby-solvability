#pragma once

#include <madrona/types.hpp>

namespace madPuzzle {
using CountT = madrona::CountT;

namespace consts {

// Steps a recorded trajectory can hold. It is large so that several full
// episodes can be recorded at 60 steps per second (e.g. Tightrope takes about
// 600-650 steps per episode).
inline constexpr CountT maxTrajectorySteps = 10240;
inline constexpr CountT maxTrajectoryObservations = 25; // max that fits in the JsonLevel struct

// JSON level constants
inline constexpr CountT maxJsonLevelDescriptions = 1024;
inline constexpr CountT maxJsonObjects = 256; // includes walls.
inline constexpr CountT maxParameterizedBehaviors = 4;
inline constexpr CountT maxBehaviorParameters = 12;
inline constexpr CountT maxBehaviorConditions = 4;

// Use the basic EntitySemantic labels.
inline constexpr bool useBasicLabels = true;

// Number of entities in an agent's observation. Training with 8192 worlds
// supports at most 72.
inline constexpr CountT maxObservationsPerAgent = 16;
inline constexpr CountT maxObjectsPerLevel = 256;
// The agent observes the entities inside a bounding box of obsAABBFactor around it.
inline constexpr float obsAABBFactor = 10.0f;

// World and entity sizes.
inline constexpr float wallWidth = 1.f;

inline constexpr float gravity = 196.2; // Roblox default, in studs/s^2

inline constexpr float worldLength = 40.0f;
inline constexpr float worldWidth = 20.f;
inline constexpr float keyWidth = 0.5f;
inline constexpr float agentRadius = 2.55f;

// Agent movement forces.
inline constexpr float agent_move_force = 216.034f;
inline constexpr float agent_jump_force = 55.0f;
inline constexpr float air_res_constant = 4.0f;
inline constexpr float agent_turn_force = 100.0f;

// Forces when the agent leaves a truss with a jump.
inline constexpr float agent_truss_force = 3000.f;
inline constexpr float agent_truss_vertical_force = 200.f;
inline constexpr madrona::math::Vector3 agentExtents = 
{2.0f, 1.948f, 5.392f};

inline constexpr madrona::math::Vector3 agentHeightCorrection = 
{0, 0, (3.198f - (consts::agentExtents.z * 0.5f))};
inline constexpr float roomLength = worldLength / 3.f;

// Number of discrete options per action component.
inline constexpr madrona::CountT numMoveAmountBuckets = 2;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 3;

// Number of lidar rays, on a spherical Fibonacci lattice. useFullLidarSphere
// covers the full sphere; false covers only the lower hemisphere.
inline constexpr madrona::CountT numLidarSamples = 128;
inline constexpr bool useFullLidarSphere = true;

// When true, the viewer draws every other lidar ray (batch-renderer limit).
inline constexpr bool visualizeLidar = false;

// Seconds per simulation step.
inline constexpr float deltaT = 0.016667f;
inline constexpr float minDeltaT = 0.016667f;

// Speed at which doors rise and lower.
inline constexpr float doorSpeed = 30.f;

// Physics substeps per step.
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

}

}
