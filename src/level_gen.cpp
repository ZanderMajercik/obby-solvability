#include "level_gen.hpp"
#include <iostream>

namespace madPuzzle {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace {


static inline Quat PYRToQuat(Vector3 rot) {

    float pitchRad = toRadians(rot.x) * 0.5f;
    float cp = cosf(pitchRad);
    float sp = sinf(pitchRad);

    float yawRad = toRadians(rot.z) * 0.5f;
    float cy = cosf(yawRad);
    float sy = sinf(yawRad);

    float rollRad = toRadians(rot.y) * 0.5f;
    float cr = cosf(rollRad);
    float sr = sinf(rollRad);

    Quat q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.y = sr * cp * cy - cr * sp * sy;
    q.x = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

enum class WallType : uint32_t {
    None,
    Solid,
    Entrance,
    Door,
};

struct WallConfig {
    WallType type;
    float entranceT = FLT_MAX;
};
}

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.sampleUniform() * range - range / 2.f;
}

// Set the components that every physics rigid body needs.
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Velocity vel,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = vel;
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system. Every entity with the
// physics components needs this: the broadphase systems run over all such
// entities and crash on an unregistered one.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) = PhysicsSystem::registerEntity(ctx, e, obj_id);
}

// Agent entity persists across all episodes.
Entity createAgent(Engine &ctx)
{
    Entity agent = ctx.makeRenderableEntity<Agent>();

    // Create a render view for the agent.
    if (ctx.data().enableRender) {
        render::RenderingSystem::attachEntityToView(ctx,
                agent,
                100.f, 0.001f,
                1.5f * math::up);
    }

    ctx.get<Scale>(agent) = Diag3x3 { consts::agentExtents.x, consts::agentExtents.y, consts::agentExtents.z};
    ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
    ctx.get<ResponseType>(agent) = ResponseType::Agent;
    ctx.get<GrabState>(agent).constraintEntity = Entity::none();
    ctx.get<ClimbState>(agent).constraintEntity = Entity::none();
    ctx.get<ClimbState>(agent).isClimbing = false;
    ctx.get<ClimbState>(agent).truss = Entity::none();
    ctx.get<EntityType>(agent) = EntityType::Agent;
    ctx.get<AirResForce>(agent).airResistance = Vector3 {0.0f, 0.0f, 0.0f};
    ctx.get<AirResForce>(agent).isSet = false;
    ctx.get<Orientation>(agent).r = Quat::angleAxis(pi, math::up);
    ctx.get<JumpPerformed>(agent).performed = 0;

    // Without population-based training, policy 0 selects the single
    // RewardHyperParams entry.
    ctx.get<AgentPolicy>(agent).policyIdx = 0;

    return agent;
}

// Re-register the agent with the broadphase system and reset its transform
// and physics state to the spawn.
void resetAgent(Engine &ctx,
                Vector3 spawn_pos,
                Quat spawn_rot,
                Vector2 spawn_size,
                Vector3 exit_pos,
                Vector3 spawn_lin_vel,
                float maxHealth)
{
    Entity agent_entity = ctx.data().agent;
    registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);


    Vector3 spawn_center = spawn_pos;
    spawn_pos.x += randInRangeCentered(ctx, spawn_size.x);
    spawn_pos.y += randInRangeCentered(ctx, spawn_size.y);
    spawn_pos = spawn_rot.rotateVec(spawn_pos - spawn_center) + spawn_center;


    ctx.get<Position>(agent_entity) = spawn_pos;
    // The spawn rotation turns the character model to face along the obstacle.
    ctx.get<Rotation>(agent_entity) = spawn_rot;

    // With UnlockAndRandomizeAgentOrientation, start with a random orientation.
    // The viewer does not show it on startup when a human controls the agent
    // (replay does).

    if ((ctx.data().simFlags & SimFlags::UnlockAndRandomizeAgentOrientation) ==
        SimFlags::UnlockAndRandomizeAgentOrientation)
    {
        ctx.get<Orientation>(agent_entity).r =
        Quat::angleAxis(randInRangeCentered(ctx, 2 * math::pi), math::up);
    }

    auto &grab_state = ctx.get<GrabState>(agent_entity);
    if (grab_state.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab_state.constraintEntity);
        grab_state.constraintEntity = Entity::none();
    }

    auto &climb_state = ctx.get<ClimbState>(agent_entity);
    if (climb_state.constraintEntity != Entity::none()) {
        ctx.destroyEntity(climb_state.constraintEntity);
        climb_state.constraintEntity = Entity::none();
        climb_state.isClimbing = false;
        climb_state.truss = Entity::none();
    }

    ctx.get<Progress>(agent_entity) = Progress {
        .minDistToExit = spawn_pos.distance(exit_pos),
        .startDistToExit = spawn_pos.distance(exit_pos),
        .minDistToButton = 0.f,
    };

    ctx.get<Velocity>(agent_entity) = {
        spawn_lin_vel,
        Vector3::zero(),
    };
    ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
    ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
    ctx.get<Action>(agent_entity) = Action {
        .moveAmount = 0,
        .moveAngle = 0,
        .rotate = 0,
        .interact = 0,
    };

    ctx.get<JumpPerformed>(agent_entity).performed = 0;

    ctx.get<Health>(agent_entity) = Health {
        .amount = maxHealth,
        .maxAmount = maxHealth
    };

}


static Entity makeExitPlatformEntity(Engine &ctx, Vector3 center, Quat rotation, Diag3x3 scale) {
    Entity e = ctx.makeRenderableEntity<ExitEntity>();

    setupRigidBodyEntity(
    ctx,
    e,
    center,
    Velocity {
            Vector3::zero(),
            Vector3::zero()
    },
    rotation,
    SimObject::BlueCube,
    EntityType::Block,
    ResponseType::Static,
    scale);
    ctx.get<EntityExtents>(e) = Vector3 {
        .x = scale.d0,
        .y = scale.d1,
        .z = scale.d2,
    };
    registerRigidBodyEntity(ctx, e, SimObject::BlueCube);
    return e;
}


static void jsonLevel(Engine &ctx)
{
    // Spawn parameters.
    Vector3 spawn_pos(0,0,0);
    Quat spawn_rot(1,0,0,0);
    Vector2 spawn_size(0.0f, 0.0f);
    Vector3 exit_pos(0, 30, 0);

    AABB obs_aabb;
    bool aabb_init = false;
    Level &level = ctx.singleton<Level>();

    JSONLevel *jsonLevels = ctx.data().jsonLevels;
    int32_t jsonIdx = ctx.singleton<JSONIndex>().index;

    // Record which JSON level is loaded.
    ctx.singleton<CurrentJSONIndex>().index = jsonIdx;

    // Physics parameters of this level.
    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    PhysicsSystem::updatePhysicsStepParameters(ctx,
        episode_state.deltaT,
        consts::numPhysicsSubsteps,
        jsonLevels[jsonIdx].physicsParams.gravity,
        jsonLevels[jsonIdx].physicsParams.airResistance);

    // The default spawn velocity is (0, 0, 0).
    Vector3 spawn_vel = jsonLevels[jsonIdx].physicsParams.spawnVelocity;

    bool setPositionForGridSearch = jsonLevels[jsonIdx].physicsParams.spawnPosition.y != -1.0f;

    // A spawnPosition other than y = -1 overrides the spawn block (used for
    // parameter grid search).
    if (setPositionForGridSearch) {
        spawn_pos = jsonLevels[jsonIdx].physicsParams.spawnPosition;
        spawn_pos -= consts::agentHeightCorrection;
    }

    // Entity of each object in the level, by state index.
    Entity states[consts::maxJsonObjects];
    for(Entity &state : states) {
        state = Entity::none();
    }

    int num_waypoints = 0;

    // Create the entity for each JSON object.
    for (const JSONObject &jsonObj : jsonLevels[jsonIdx].objects) {
        // Skip unused object slots.
        if (jsonObj.extents.x == 0.0f &&
            jsonObj.extents.y == 0.0f &&
            jsonObj.extents.z == 0.0f) {
            continue;
        }
        Vector3 pos = jsonObj.position;
        Vector3 yaw = Vector3(0, 0, jsonObj.orientation.z);
        Vector3 pitch = Vector3(jsonObj.orientation.x, 0, 0);
        Vector3 roll = Vector3(0, jsonObj.orientation.y, 0);
        Quat rot = (PYRToQuat(yaw).normalize() * (PYRToQuat(pitch).normalize() * PYRToQuat(roll).normalize()).normalize()).normalize();

        // Axis-aligned bounding box of the oriented object.
        Vector3 object_size = jsonObj.extents * 0.5f;
        Vector3 corners[8] = {
            Vector3(1,1,1),
            Vector3(-1,1,1),
            Vector3(1,-1,1),
            Vector3(-1,-1,1),
            Vector3(1,1,-1),
            Vector3(-1,1,-1),
            Vector3(1,-1,-1),
            Vector3(-1,-1,-1)
        };
        AABB objAABB;
        for (int i = 0; i < 8; ++i) {
            Vector3 local_corner = rot.rotateVec(
                Vector3(
                    object_size.x * corners[i].x,
                    object_size.y * corners[i].y,
                    object_size.z * corners[i].z
                )) + pos;
            if (i == 0) {
                objAABB.pMin = local_corner;
                objAABB.pMax = local_corner;
            }
            objAABB.pMin = Vector3::min(objAABB.pMin, local_corner);
            objAABB.pMax = Vector3::max(objAABB.pMax, local_corner);
        }

        if (!aabb_init) {
            obs_aabb = objAABB;
            aabb_init = true;
        }

        obs_aabb = AABB::merge(obs_aabb, objAABB);

        // The behavior flags decide which renderable entity type to create.
        Entity e = Entity::none();
        Velocity vel = {
            Vector3::zero(),
            Vector3::zero()
        };
        Diag3x3 scale = Diag3x3::fromVec(jsonObj.extents);
        // TODO: infer the SimObject from shape and behavior instead of reading it from JSON.
        SimObject sim_obj = SimObject(uint32_t(jsonObj.simObject));
        EntityType entity_type = EntityType::Block;
        ResponseType response_type = ResponseType::Static;

        // First pass over the behaviors: choose the SimObject and EntityType.
        for(const ObjectBehavior &objectBehavior : jsonObj.objectBehavior)
        {

            // Process the behavior type.
            int32_t behaviorType = objectBehavior.behaviorType;
            if(behaviorType < 0) {
                continue;
            }

            BehaviorType b = BehaviorType(behaviorType);

            switch (b) {
                case BehaviorType::Spawn:
                    sim_obj = SimObject::GreenCube; break;
                case BehaviorType::Checkpoint:
                    sim_obj = SimObject::BlueCube; 
                    entity_type = EntityType::Goal;
                    num_waypoints++;
                    break;
                case BehaviorType::Damage:
                    // Damaging objects are drawn red.
                    switch(sim_obj) {
                        case SimObject::Cube:
                            sim_obj = SimObject::RedCube; break;
                        case SimObject::Sphere:
                            sim_obj = SimObject::RedSphere; break;
                        case SimObject::Wedge:
                            sim_obj = SimObject::RedWedge; break;
                        case SimObject::Cylinder:
                            sim_obj = SimObject::RedCylinder; break;
                        default: break;
                    }
                    if (entity_type == EntityType::Motion) {
                        entity_type = EntityType::LavaMotion;
                    } else if (entity_type != EntityType::LavaMotion) {
                        entity_type = EntityType::Lava;
                    }
                    break;
                case BehaviorType::Rotate:
                case BehaviorType::Motion:
                    if (entity_type == EntityType::Lava) {
                        entity_type = EntityType::LavaMotion;
                    } else if (entity_type == EntityType::Truss) {
                        entity_type = EntityType::TrussMotion;
                    } else if (entity_type != EntityType::LavaMotion) {
                        entity_type = EntityType::Motion;
                    }
                    break;
                case BehaviorType::Trampoline:
                case BehaviorType::Conveyor:
                    sim_obj = SimObject::PurpleCube; 
                    entity_type = EntityType::Impulse;
                    break;
                case BehaviorType::Truss:
                    sim_obj = SimObject::YellowCube;
                    if (entity_type == EntityType::Motion) {
                        entity_type = EntityType::TrussMotion;
                    } else if (entity_type != EntityType::TrussMotion) {
                        entity_type = EntityType::Truss;
                    }
                    break;
                case BehaviorType::PressState:
                    sim_obj = SimObject::YellowCube;
                    entity_type = EntityType::Button;
                    break;
                case BehaviorType::Waypoint:
                    // Every entity has a waypoint index, so the entity type stays.
                    sim_obj = SimObject::CyanCube;
                    num_waypoints++;
                    break;
                case BehaviorType::NoCollide:
                    response_type = ResponseType::StaticNoCollide;
                    break;
                case BehaviorType::Unanchored:
                    sim_obj = SimObject::OrangeCube;
                    response_type = ResponseType::Dynamic;
                default: break;
            }
        }

        // Create the renderable entity.
        switch (entity_type) {
            case EntityType::Goal:
                e = ctx.makeRenderableEntity<ExitEntity>(); break;
            case EntityType::Lava: 
                e = ctx.makeRenderableEntity<LavaEntity>(); break;
            case EntityType::LavaMotion:
                e = ctx.makeRenderableEntity<LavaMotionEntity>(); break;
            case EntityType::Motion:
                e = ctx.makeRenderableEntity<MotionEntity>(); break;
            case EntityType::Impulse:
                e = ctx.makeRenderableEntity<ImpulseObjectEntity>(); break;
            case EntityType::Truss:
                e = ctx.makeRenderableEntity<TrussEntity>(); break;
            case EntityType::TrussMotion:
                e = ctx.makeRenderableEntity<TrussMotionEntity>(); break;
            case EntityType::Button:
                e = ctx.makeRenderableEntity<ButtonEntity>(); break;
            case EntityType::Block:
            default:
                e = ctx.makeRenderableEntity<PhysicsEntity>(); break;
        }


        // The entity must exist.
        assert(e.id != Entity::none().id);

        bool isSpawn = false;

        // -1: not a waypoint.
        ctx.get<WaypointProperties>(e).idx = -1;

        // Record the entity under its state index (conditions refer to it).
        int32_t state_idx = (int32_t) jsonObj.objectState.state_index;
        states[state_idx] = e;

        // Second pass: apply the behavior parameters (Truss has none).
        for(const ObjectBehavior &objectBehavior : jsonObj.objectBehavior)
        {
            // Process the behavior type.
            int32_t behaviorType = objectBehavior.behaviorType;
            if(behaviorType < 0) {
                continue;
            }
            auto behaviorParameters = objectBehavior.behaviorParameters;
            if (behaviorType & uint32_t(BehaviorType::Spawn)) {
                isSpawn = true; 
                continue;
            }
            else if (behaviorType & uint32_t(BehaviorType::Checkpoint)) {
                // The goal is the level exit.
                exit_pos = pos;
                level.exit = e;
                // Initial value; waypointSystem sets the active waypoint on its first run.
                level.waypoint = e;

                // The checkpoint is always the final waypoint (index 0) and
                // has no jump limit.
                ctx.get<WaypointProperties>(e).idx = 0;
                ctx.get<WaypointProperties>(e).jumps = -1;
            }
            else if (behaviorType & uint32_t(BehaviorType::Waypoint)) {
                // waypointSystem makes this waypoint active when its turn comes.
                ctx.get<WaypointProperties>(e).idx = behaviorParameters[0];
                ctx.get<WaypointProperties>(e).jumps = behaviorParameters[1];
            }
            else if (behaviorType & uint32_t(BehaviorType::Damage)) {
                ctx.get<LavaProperties>(e).damage = behaviorParameters[0];
            }
            else if (behaviorType & uint32_t(BehaviorType::Rotate)) {
                // behaviorParameters[0]: angular velocity in radians/second.
                // behaviorParameters[1]: rotation axis (0 = right, 1 = up, else forward).
                float angular = behaviorParameters[0];
                float axis = behaviorParameters[1];
                ctx.get<MotionState>(e).start_pos = pos;
                ctx.get<MotionState>(e).end_pos = pos;
                ctx.get<MotionState>(e).linear_vel = Vector3::zero();
                // Express the axis in the object's frame.
                if (axis > -1.0 && axis < 1.0) {
                    ctx.get<MotionState>(e).angular_vel = rot.rotateVec(math::right * angular);
                } else if (axis > 0.0 && axis < 2.0) {
                    ctx.get<MotionState>(e).angular_vel = rot.rotateVec(math::up * angular);
                } else {
                    ctx.get<MotionState>(e).angular_vel = rot.rotateVec(math::fwd * angular);
                }
                ctx.get<MotionState>(e).reverses = false;
                ctx.get<MotionState>(e).loops = -1; // rotates forever
                ctx.get<MotionState>(e).loopsCompleted = 0;
                ctx.get<MotionState>(e).time = 1.0f;

                response_type = ResponseType::Kinematic;
            }
            else if (behaviorType & uint32_t(BehaviorType::Motion)) {
                // Motion from start_pos to end_pos in `time` seconds, relative to
                // the block's starting position and orientation.
                Vector3 &start_pos = ctx.get<MotionState>(e).start_pos;
                start_pos = Vector3{
                    behaviorParameters[0],
                    behaviorParameters[1],
                    behaviorParameters[2]
                };
                
                Vector3 &end_pos = ctx.get<MotionState>(e).end_pos;
                end_pos = Vector3{
                    behaviorParameters[3],
                    behaviorParameters[4],
                    behaviorParameters[5]
                };

                float time = behaviorParameters[6];
                // behaviorParameters[11] scales the velocity (its sign sets the direction).
                ctx.get<MotionState>(e).linear_vel = float(behaviorParameters[11]) * (end_pos - start_pos) / time;
                ctx.get<MotionState>(e).angular_vel = Vector3::zero();
                ctx.get<MotionState>(e).reverses = (bool) behaviorParameters[9];
                ctx.get<MotionState>(e).loops = (int32_t) behaviorParameters[10];
                ctx.get<MotionState>(e).loopsCompleted = 0;
                ctx.get<MotionState>(e).time = time;

                // Start moving with the motion's velocity.
                vel.linear = ctx.get<MotionState>(e).linear_vel;
                response_type = ResponseType::Kinematic;
            }
            else if ((behaviorType & uint32_t(BehaviorType::Trampoline)) || (behaviorType & uint32_t(BehaviorType::Conveyor))) {
                Vector3 &bounce = ctx.get<ImpulseObjectState>(e).bounce;
                bounce = {
                    behaviorParameters[0],
                    behaviorParameters[1],
                    behaviorParameters[2],
                };
            }
            else if (behaviorType & uint32_t(BehaviorType::PressState)) {
                ctx.get<ButtonState>(e).isPressed = jsonObj.objectState.state_value;
                ctx.get<ButtonState>(e).isHold = false;
            }
            else if (behaviorType & uint32_t(BehaviorType::HoldState)) {
                ctx.get<ButtonState>(e).isPressed = jsonObj.objectState.state_value;
                ctx.get<ButtonState>(e).isHold = true;
            }
            // Only rotation and motion support conditions.
            if (behaviorType & uint32_t(BehaviorType::Rotate) || behaviorType & uint32_t(BehaviorType::Motion)) {
                ConditionsList &conditionsList = ctx.get<ConditionsList>(e);
                conditionsList.numConditions = 0;
                for(float objectCondition : objectBehavior.behaviorConditions)
                {
                    int32_t conditionIdx = (int32_t) objectCondition;
                    if(conditionIdx < 0) {
                        continue;
                    }
                    conditionsList.conditions[conditionsList.numConditions] = states[conditionIdx];
                    conditionsList.numConditions++;
                }
            }
        }

        // Set the spawn after all behaviors, since some move the block's start.
        if (isSpawn) {
            spawn_pos = pos;
            spawn_pos.z += (jsonObj.extents.z + consts::agentExtents.z) * 0.5f;
            // The agent always starts at the center of the spawn block.
            spawn_size = Vector2(0.0f,0.0f);
            // Use only the yaw; the extra 180 degrees turns the agent to face
            // along the obstacle.
            spawn_rot = PYRToQuat(yaw).normalize() * PYRToQuat(Vector3(0, 0, 180)).normalize();
        }

        setupRigidBodyEntity(
            ctx,
            e,
            pos,
            vel,
            rot,
            sim_obj,
            entity_type,
            response_type,
            scale
        );
        ctx.get<EntityExtents>(e) = Vector3 {
            .x = scale.d0,
            .y = scale.d1,
            .z = scale.d2,
        };
        registerRigidBodyEntity(ctx, e, sim_obj);
    }


    // Waypoint state for the waypoint-based rewards.
    episode_state.numWaypoints = num_waypoints;
    episode_state.activeWaypointIdx = num_waypoints - 1;

    ctx.singleton<LevelAABB>() = obs_aabb;

    Vector2 spawn_size_area = spawn_size;

    resetAgent(ctx, spawn_pos, spawn_rot, spawn_size_area, exit_pos, spawn_vel, 100.f);
}

LevelType generateLevel(Engine &ctx, LevelType level_type)
{
    level_type = (LevelType)ctx.data().rng.sampleI32(
        0, (uint32_t)LevelType::NumTypes);
    // Load the JSON level. The level type is not used for JSON levels.
    if (ctx.singleton<JSONIndex>().index != -1) {
        level_type = LevelType(0);
        // The training code chooses the JSON level indices.
        jsonLevel(ctx);
        return level_type;
    }

    // Levels come only from JSON, but the JSON index can be set only after the
    // simulator exists. So initialization loads this placeholder level; the
    // first reset replaces it once the training code has written the indices.

    Level &level = ctx.singleton<Level>();
    level.exit = makeExitPlatformEntity(ctx, Vector3::zero(), Quat{1,0,0,0}, Diag3x3{1,1,1});
    level.waypoint = makeExitPlatformEntity(ctx, Vector3::zero(), Quat{1,0,0,0}, Diag3x3{1,1,1});

    resetAgent(ctx, Vector3(0,0, -5), Quat{1,0,0,0}, Vector2(1, 1), Vector3::zero(), Vector3::zero(), 100.f);

    return LevelType::SimpleLocomotion;
}

void destroyLevel(Engine &ctx)
{
    ctx.iterateQuery(ctx.data().simEntityQuery, [
        &ctx
    ](Entity e, EntityType type)
    {
        if (type == EntityType::Agent) {
            return;
        }

        Loc l = ctx.makeTemporary<DeferredDeleteEntity>();
        ctx.get<DeferredDelete>(l).e = e;
    });
}

}
