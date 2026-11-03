#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>
#include "consts.hpp"

namespace madPuzzle {
    // Madrona types used throughout the simulator.
    using madrona::Entity;
    using madrona::RandKey;
    using madrona::CountT;
    using madrona::Query;
    using madrona::base::Position;
    using madrona::base::Rotation;
    using madrona::base::Scale;
    using madrona::base::ObjectID;
    using madrona::base::ObjectInstance;
    using madrona::phys::Velocity;
    using madrona::phys::FrameLinearVelocity;
    using madrona::phys::ResponseType;
    using madrona::phys::ExternalForce;
    using madrona::phys::ExternalTorque;
    using madrona::phys::RigidBody;
    using madrona::phys::AgentTouched;
    using madrona::math::Vector3;
    using madrona::math::Vector2;
    using madrona::math::Quat;
    using madrona::math::AABB;

    namespace PhysicsSystem = madrona::phys::PhysicsSystem;
    namespace RenderingSystem = madrona::render::RenderingSystem;

    // Per-world reference trajectory (a singleton component).
    struct WorldTrajectory {
        Vector3 observations[consts::maxTrajectoryObservations]; // one point per recorded step
        uint32_t seed;
        uint32_t exists; // nonzero when trajectory data is present
    };

    struct TrajectorySection {
        Vector3 previous;
        Vector3 closest;
        Vector3 next;
    };

    // Per-world singleton: setting reset = 1 ends the current episode and
    // regenerates the world (ctx.singleton<WorldReset>().reset = 1).
    struct WorldReset {
        int32_t reset;
    };

    struct TimeStep {
        float delta;
    };

    struct DebugMaxAirRes {
        Vector3 maxAirRes;
    };

    enum class LevelType : uint32_t {
        Chase,
        LavaPath,
        LavaButton,
        SingleButton,
        SingleBlockButton,
        ObstructedBlockButton,
        BlockStack,
        PatternMatch,
        ChickenCoop,
        SimpleLocomotion,
        LavaCorridor,
        NumTypes,
    };

    // Bit flags for object behaviors. scripts/generateLevels.py writes the same
    // bit mask, so keep the two in sync.
    enum class BehaviorType : uint32_t {
        None = 0,
        Spawn = 1 << 0,
        Checkpoint = 1 << 1,
        Damage = 1 << 2,
        Rotate = 1 << 3,
        Motion = 1 << 4,
        Trampoline = 1 << 5,
        Conveyor = 1 << 6,
        Truss = 1 << 7, // climbable truss
        PressState = 1 << 8,
        HoldState = 1 << 9,
        Waypoint = 1 << 10,
        NoCollide = 1 << 11,
        Unanchored = 1 << 12
    };

    // Entity type, used to classify lidar hits. An archetype needs an
    // EntityType to be deleted by level_gen.cpp:destroyLevel().
    enum class EntityType : uint32_t {
        None,
        Agent,
        Block,
        Lava,
        LavaMotion,
        TrussMotion,
        Button,
        Goal,
        Spawn,
        Impulse,
        Truss,
        Motion,
        Door,
        NumTypes,
    };

    enum class EntitySemantics : uint32_t {
        None,
        Normal,
        Avoid,
        NumTypes,
    };

    template<bool B>
    struct EntityLabelSelector {
        using type = EntityType;
    };

    template<>
    struct EntityLabelSelector<true> {
        using type = EntitySemantics;
    };

    using EntityLabel = EntityLabelSelector<consts::useBasicLabels>::type;

    struct ObjectState
    {
        float state_index;
        bool state_value;
    };

    struct ObjectBehavior
    {
        float behaviorType;
        float behaviorConditions[consts::maxBehaviorConditions];
        float behaviorParameters[consts::maxBehaviorParameters];
    };

    struct JSONObject {
        Vector3 position;
        Vector3 extents;
        Vector3 orientation;
        float simObject;
        ObjectState objectState;
        ObjectBehavior objectBehavior[consts::maxParameterizedBehaviors];
    };

    struct PhysicsParameters {
        float gravity;
        float moveForce;
        float jumpForce;
        float airResistance;
        Vector3 spawnPosition;
        Vector3 spawnVelocity;
    };

    struct JSONLevel {
        // Per-level physics parameters, padded to the size of a JSONObject so
        // that they can be registered as a singleton component.
        PhysicsParameters physicsParams;
        float padding[(sizeof(JSONObject) - sizeof(PhysicsParameters))  / sizeof(float)];

        // The reference trajectory, padded the same way, so that it is
        // imported with the level tensor and registered as a singleton.
        WorldTrajectory worldTraj;
        float traj_padding[(sizeof(JSONObject) - sizeof(WorldTrajectory))  / sizeof(float)];

	    JSONObject objects[consts::maxJsonObjects];

    };

    // Index of the JSON level in use. JSONIndex holds the next level to load.
    struct CurrentJSONIndex {
        int32_t index;
    };

    struct JSONIndex {
        int32_t index;
    };

    struct EpisodeState {
        uint32_t curStep;
        uint32_t curLevel;
        int32_t curJumps;
        bool isDead;
        bool reachedExit;
        bool reachedWaypoint;
        int32_t activeWaypointIdx;
        uint32_t numWaypoints;
        bool episodeFinished;
        float deltaT;
    };

    // Discrete action. The ranges come from consts::numMoveAmountBuckets,
    // numMoveAngleBuckets and numTurnBuckets.
    struct Action {
        int32_t moveAmount; // [0, 1]
        int32_t moveAngle; // [0, 7]
        int32_t rotate; // 0 = do nothing, 1 = clockwise, 2 = anticlockwise
        int32_t interact; // 0 = do nothing, 1 = jump
    };

    struct ActionLatch {
        Action action;
    };

    // Per-agent reward, exported to training as an [N * A, 1] float tensor.
    struct Reward {
        float v;
    };

    // Per-agent flag: the agent's episode is finished.
    struct Done {
        // int32_t because bool components are not supported (padding).
        int32_t v;
    };

    // Episode success, kept separate from the reward for the search methods.
    struct Success {
        int32_t v;
    };

    // Egocentric polar coordinates. theta is the angle off the agent's forward
    // direction.
    struct PolarObs {
        float r;
        float theta;
        float phi;
    };

    struct AgentTxfmObs {
        Vector3 localRoomPos;
        AABB levelAABB;
        float theta;
        Vector3 linearVelocity;
        float activeWaypointIdx;
        Action action;
        float health;
    };

    struct AgentInteractObs {
        int32_t isGrabbing;
    };

    struct AgentLevelTypeObs {
        LevelType type;
    };

    struct AgentExitObs {
        PolarObs toExitPolar;
        float minDistToExit;
    };

    // Egocentric observation of one nearby entity.
    struct EntityPhysicsStateObs {
        PolarObs positionPolar;
        PolarObs velocityPolar;
        PolarObs angVelocityPolar;
        Vector3 extents;
        Vector3 entityRotation;
    };

    struct EntityTypeObs {
        EntityLabel entityType;
    };

    struct EntityPhysicsStateObsArray {
        EntityPhysicsStateObs obs[consts::maxObservationsPerAgent];
    };

    struct EntityTypeObsArray {
        EntityTypeObs obs[consts::maxObservationsPerAgent];
    };

    // Lidar: depth and hit entity type for each ray around the agent.
    struct LidarDepth {
        float samples[consts::numLidarSamples];
    };

    struct LidarHitType {
        EntityLabel samples[consts::numLidarSamples];
    };

    // Steps remaining in the episode, so that a non-recurrent policy can
    // track time.
    struct StepsRemainingObservation {
        uint32_t t;
    };

    struct StepsRemaining {
        uint32_t t;
    };

    // The agent's best progress so far; the reward pays only for new progress.
    struct Progress {
        float minDistToExit;
        float startDistToExit;
        float minDistToButton;
    };

    // Health of agents and enemies.
    struct Health {
        float amount;
        float maxAmount; // cap for healing
    };

    // The constraint that holds a grabbed entity, if any.
    struct GrabState {
        Entity constraintEntity;
    };

    struct AirResForce {
        Vector3 airResistance;
        bool isSet;
    };

    struct ClimbState {
        bool isClimbing;
        Vector3 normal;
        Entity truss;
        Entity constraintEntity;
    };

    // Whether a door is open.
    struct OpenState {
        bool isOpen;
    };

    struct EntityExtents : Vector3 {
        inline EntityExtents(Vector3 v) : Vector3(v) {
        }
    };

    struct LevelAABB : AABB {
        inline LevelAABB(AABB aabb) : AABB(aabb) {
        }
    };

    struct DoorRooms {
        Entity linkedRooms[2];
    };

    struct Orientation {
        Rotation r;
    };

    // isPressed is true during the frames in which a button is pressed.
    struct ButtonState {
        bool isPressed;
        bool isHold; // a hold button stays active only while the agent presses it
    };

    struct PatternMatchState {
        bool isMatched;
    };

    struct IsViewBox {
    };

    struct CoopState {
        bool isOccupied;
    };

    struct EntityLinkedListElem {
        Entity next;
    };

    struct ButtonListElem : EntityLinkedListElem {
    };

    struct PatternListElem : EntityLinkedListElem {
    };

    struct ChickenListElem : EntityLinkedListElem {
    };

    struct ConditionsList {
        uint32_t numConditions;
        Entity conditions[consts::maxBehaviorConditions];
    };

    struct Level {
        Entity exit;
        Entity waypoint;
    };

    struct DoorProperties {
        bool isPersistent;
    };

    struct DeferredDelete {
        Entity e;
    };

    struct IsExit {
    };

    struct LavaProperties {
        float damage;
    };

    struct WaypointProperties {
        int32_t idx;
        int32_t jumps;
    };

    struct IsOrientationArrow {
    };

    struct IsLidar {
    };

    struct ImpulseObjectState {
        Vector3 bounce;
    };

    struct IsTruss {
    };

    struct EnemyState {
        float moveForce;
        float moveTorque;
        bool isChicken;
        bool isDead;
    };

    struct MotionState {
        Vector3 start_pos;
        Vector3 end_pos;
        Vector3 linear_vel;
        Vector3 angular_vel;
        bool reverses;
        int32_t loops; // -1 for infinite
        int32_t loopsCompleted;
        float time;
    };

    // Saved world state that a world can be restored to.
    struct Checkpoint {
        struct EntityState {
            EntityType entityType;
            Vector3 position;
            Quat rotation;
            Velocity velocity;

            struct ConditionsState {
                uint32_t numConditions;
                uint32_t conditions[consts::maxBehaviorConditions];
            } conditions;

            union {
                OpenState doorOpen;
                ButtonState button;
                CoopState coop;
                IsTruss truss;
                ConditionsState conditionsState;
            };
        };

        struct AgentState {
            Vector3 position;
            Quat rotation;
            Quat orientation;
            Velocity velocity;
            Progress taskProgress;
            // Index of the grabbed entity in entityStates; -1 if not grabbing.
            int32_t grabIdx;
            madrona::phys::JointConstraint grabJoint;
            // Index of the climbed truss in entityStates, and its joint.
            int32_t climbIdx;
            madrona::phys::JointConstraint climbJoint;
        };

        madrona::RandKey initRNDCounter;
        int32_t curEpisodeStep;
        int32_t curEpisodeLevel;
        int32_t curEpisodeJumps;
        int32_t activeWaypointIdx;
        // Index of the JSON level in use.
        int32_t jsonIndex;

        EntityState entityStates[consts::maxObjectsPerLevel];
        AgentState agentState;
    };

    struct AgentPolicy {
        int32_t policyIdx;
    };

    struct EpisodeResult {
        float score;
    };

    struct RewardHyperParams {
        float distToExitScale;
    };

    struct CheckpointReset {
        int32_t reset;
    };

    // Set by the viewer to save a checkpoint.
    struct CheckpointSave {
        int32_t save;
    };

    struct DeferredDeleteEntity : public madrona::Archetype<
                DeferredDelete
            > {
    };

    // Debug flag: a jump was performed this step.
    struct JumpPerformed {
        uint32_t performed;
    };

    struct DynamicBlockPos {
        Vector3 pos;
    };

    // ECS archetypes.

    // The agent, which tries to reach the level's exit.
    struct Agent : public madrona::Archetype<
                // The physics system requires these components first, in this
                // exact order.
                RigidBody,

                // Internal logic state.
                GrabState,
                Progress,
                EntityType,
                Health,
                ClimbState,
                AirResForce,

                // Input
                Action,
                JumpPerformed,

                // Observations
                AgentTxfmObs,
                AgentInteractObs,
                AgentLevelTypeObs,
                AgentExitObs,
                StepsRemainingObservation,
                EntityPhysicsStateObsArray,
                EntityTypeObsArray,
                LidarDepth,
                LidarHitType,

                // Reward and episode termination
                Reward,
                Done,
                AgentPolicy,

                // Frame for the agent's actions and observations; it can differ
                // from the character model's orientation.
                Orientation,

                // Lets src/viewer.cpp view the scene from this entity, in addition
                // to the fly camera.
                madrona::render::RenderCamera,
                // Drawn by the viewer and the batch renderer.
                madrona::render::Renderable
            > {
    };

    // A door, opened by linked buttons, patterns or chickens.
    struct DoorEntity : public madrona::Archetype<
                RigidBody,

                OpenState,
                DoorProperties,
                ButtonListElem,
                PatternListElem,
                ChickenListElem,
                DoorRooms,
                EntityType,
                EntityExtents,
                madrona::render::Renderable
            > {
    };

    // A button that opens doors.
    struct ButtonEntity : public madrona::Archetype<
                RigidBody,

                ButtonState,
                EntityType,
                EntityExtents,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    struct TrussEntity : public madrona::Archetype<
                RigidBody,

                EntityType,
                EntityExtents,
                IsTruss,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    struct ImpulseObjectEntity : public madrona::Archetype<
                RigidBody,

                EntityType,
                EntityExtents,
                ImpulseObjectState,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    // An entity with physics but no custom logic.
    struct PhysicsEntity : public madrona::Archetype<
                RigidBody,

                EntityType,
                EntityExtents,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    // The level's exit.
    struct ExitEntity : public madrona::Archetype<
                RigidBody,

                EntityType,
                EntityExtents,
                madrona::render::Renderable,
                IsExit,
                WaypointProperties
            > {
    };

    // An enemy (for example a chicken).
    struct EnemyEntity : public madrona::Archetype<
                RigidBody,

                EntityType,
                ChickenListElem,
                EntityExtents,
                EnemyState,
                madrona::render::Renderable
            > {
    };

    struct LavaEntity : public madrona::Archetype<
                RigidBody,

                EntityType,
                EntityExtents,
                LavaProperties,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    struct MotionEntity : public madrona::Archetype<
                RigidBody,
                EntityType,
                EntityExtents,
                MotionState,
                ConditionsList,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    struct LavaMotionEntity : public madrona::Archetype<
                RigidBody,
                EntityType,
                EntityExtents,
                ConditionsList,
                LavaProperties,
                MotionState,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    struct TrussMotionEntity : public madrona::Archetype<
                RigidBody,
                EntityType,
                EntityExtents,
                ConditionsList,
                IsTruss,
                MotionState,
                madrona::render::Renderable,
                WaypointProperties
            > {
    };

    // A pattern location that a door can require to be matched.
    struct PatternEntity : public madrona::Archetype<
                ObjectInstance,

                PatternMatchState,
                PatternListElem,
                madrona::render::Renderable
            > {
    };

    // The view box entity (Sim::viewBox), drawn in the viewer.
    struct ViewBoxEntity : public madrona::Archetype<
                ObjectInstance,
                IsViewBox,
                madrona::render::Renderable
            > {
    };

    struct OrientationArrowEntity : public madrona::Archetype<
                ObjectInstance,
                IsOrientationArrow,
                madrona::render::Renderable
            > {
    };

    struct LidarEntity : public madrona::Archetype<
                ObjectInstance,
                IsLidar,
                madrona::render::Renderable
            > {
    };

    struct CoopEntity : public madrona::Archetype<
                RigidBody,

                CoopState,
                ChickenListElem,
                EntityType,
                EntityExtents,
                madrona::render::Renderable
            > {
    };
}