#include <madrona/mw_gpu_entry.hpp>
#include <iostream>

#include "sim.hpp"
#include "level_gen.hpp"
#include "../external/madrona/src/physics/physics_impl.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace madPuzzle {

static inline Vector3 quatToEuler(Quat q)
{
    // Rotation about y.
    float sinp = 2.f * (q.w * q.y - q.x * q.z);
    float pitch;
    if (fabsf(sinp) >= 1.f) {
        pitch = copysignf(math::pi / 2.f, sinp);
    } else {
        pitch = asinf(sinp);
    }

    // Rotation about x.
    float sinr_cosp = 2.f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.f - 2.f * (q.x * q.x + q.y * q.y);

    // Rotation about z (yaw).
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return Vector3 {
        .x = atan2f(sinr_cosp, cosr_cosp),
        .y = pitch,
        .z = atan2f(siny_cosp, cosy_cosp),
    };
}

static inline Vector3 airResistanceForce(Vector3 v) {
    Vector3 f_d = {0.0f, 0.0f, 0.0f};
    v.z = 0.0f; // Ignore vertical resistance.
    float v_2 = v.length2();
    float c_d = 4.0f;
    if (v_2 > 1e-5f) {
        f_d = 0.5f * v_2 * v.normalize() * c_d;
    }

    return f_d;
}

// Convert a 3D offset to polar coordinates for the observations.
static inline PolarObs xyzToPolar(Vector3 v)
{
    float r = v.length();

    if (r < 1e-5f) {
        return PolarObs {
            .r = 0.f,
            .theta = 0.f,
            .phi = 0.f,
        };
    }

    v /= r;

    // theta is the angle off the +y (forward) axis.
    float theta = -atan2f(v.x, v.y);
    float phi = asinf(std::max(-1.0f, std::min(1.0f, v.z)));

    return PolarObs {
        .r = r,
        .theta = theta,
        .phi = phi,
    };
}

static inline Vector3 polarToXYZ(PolarObs p) {
    float r = p.r;
    float theta = p.theta;
    float phi = p.phi;

    return Vector3 {
        r * sinf(theta) * cosf(phi),
        r * cosf(theta) * cosf(phi),
        r * sinf(phi)
    };
}


static void initEpisodeRNG(Engine &ctx)
{
    RandKey new_rnd_counter;
    if ((ctx.data().simFlags & SimFlags::UseFixedWorld) ==
            SimFlags::UseFixedWorld) {
        new_rnd_counter = { 0, 0 };
    } else {
        if (ctx.singleton<CheckpointReset>().reset == 1) {
            // When loading a checkpoint, reuse the seed of the saved world.
            new_rnd_counter = ctx.singleton<Checkpoint>().initRNDCounter;
        } else {
            new_rnd_counter = {
                .a = ctx.data().curWorldEpisode++,
                .b = (uint32_t)ctx.worldID().idx,
            };
        }
    }

    ctx.data().curEpisodeRNDCounter = new_rnd_counter;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        new_rnd_counter.a, new_rnd_counter.b));
}

static EpisodeState initEpisodeState()
{
    return EpisodeState {
        .curStep = 0,
        .curLevel = 0,
        .curJumps = 0,
        .isDead = false,
        .reachedExit = false,
        .reachedWaypoint = false,
        .activeWaypointIdx = 0,
        .episodeFinished = true,
        .deltaT = 0
    };
}

inline void checkpointSystem(Engine &ctx, CheckpointSave &save)
{
    if (save.save == 0) {
        // Checkpoint every step unless save is 0 (the viewer sets 0 to keep a
        // specific saved state).
        return;
    }

    Checkpoint &ckpt = ctx.singleton<Checkpoint>();

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    // Save the random seed.
    ckpt.initRNDCounter = ctx.data().curEpisodeRNDCounter;
    ckpt.curEpisodeStep = episode_state.curStep;
    ckpt.curEpisodeLevel = episode_state.curLevel;
    ckpt.curEpisodeJumps = episode_state.curJumps;
    ckpt.activeWaypointIdx = episode_state.activeWaypointIdx;

    // Save the JSON level index.
    ckpt.jsonIndex = ctx.singleton<CurrentJSONIndex>().index;

    Entity agent = ctx.data().agent;
    Optional<JointConstraint> agent_grab_joint =
        Optional<JointConstraint>::none();
    {
        Entity grab_constraint = ctx.get<GrabState>(agent).constraintEntity;
        if (grab_constraint != Entity::none()) {
            agent_grab_joint = ctx.get<JointConstraint>(grab_constraint);
        }
    }
    Optional<JointConstraint> agent_climb_joint =
        Optional<JointConstraint>::none();
    {
        Entity climb_constraint = ctx.get<ClimbState>(agent).constraintEntity;
        if (climb_constraint != Entity::none()) {
            agent_climb_joint = ctx.get<JointConstraint>(climb_constraint);
        }
    }

    CountT agent_grab_idx = -1;
    CountT agent_climb_idx = -1;
    { // Save entities
        Checkpoint::EntityState *entity_states = &ckpt.entityStates[0];

        CountT out_idx = 0;
        ctx.iterateQuery(ctx.data().simEntityQuery,
        [&](Entity e, EntityType type) {
            if (type == EntityType::None || type == EntityType::Agent || type == EntityType::Goal) {
                return;
            }

            if (agent_grab_joint.has_value() && e == agent_grab_joint->e2) {
                agent_grab_idx = out_idx;
            }
            if (agent_climb_joint.has_value() && e == agent_climb_joint->e2) {
                agent_climb_idx = out_idx;
            }

            auto &state_out = entity_states[out_idx];

            if (type > EntityType::NumTypes || type < EntityType::None) {
                printf("type out of bounds %d\n", (int)type);
            }
            state_out.entityType = type;
            state_out.position = ctx.get<Position>(e);
            state_out.rotation = ctx.get<Rotation>(e);
            state_out.velocity = ctx.get<Velocity>(e);

            switch (type) {
            case EntityType::Door: {
                state_out.doorOpen = ctx.get<OpenState>(e);
            } break;
            case EntityType::Button: {
                state_out.button = ctx.get<ButtonState>(e);
            } break;
            case EntityType::Truss: {
                state_out.truss = ctx.get<IsTruss>(e);
            } break;
            case EntityType::Motion: {

                Checkpoint::EntityState::ConditionsState &conditionsState = state_out.conditionsState;
                conditionsState.numConditions = 0;

                ConditionsList conditions = ctx.get<ConditionsList>(e);
                while (conditionsState.numConditions < conditions.numConditions)
                {
                    uint32_t idx = 0;
                    ctx.iterateQuery(ctx.data().conditionQuery,
                    [&](Entity e, ButtonState bs) {
                        (void)bs;
                        if(e == conditions.conditions[conditionsState.numConditions])
                        {
                            conditionsState.conditions[conditionsState.numConditions] = idx;
                            conditionsState.numConditions++;
                        }
                        idx++;
                    });
                }
            } break;
            default: break;
            }

            out_idx += 1;
        });

        for (CountT i = out_idx; i < consts::maxObjectsPerLevel; i++) {
            entity_states[i].entityType = EntityType::None;
        }
    }

    { // Save agent
        JointConstraint grab_joint_out{};
        if (agent_grab_joint.has_value()) {
            grab_joint_out = *agent_grab_joint;
        }
        JointConstraint climb_joint_out{};
        if (agent_climb_joint.has_value()) {
            climb_joint_out = *agent_climb_joint;
        }

        ckpt.agentState = {
            .position = ctx.get<Position>(agent),
            .rotation = ctx.get<Rotation>(agent),
            .orientation = ctx.get<Orientation>(agent).r,
            .velocity = ctx.get<Velocity>(agent),
            .taskProgress = ctx.get<Progress>(agent),
            .grabIdx = (int32_t)agent_grab_idx,
            .grabJoint = grab_joint_out,
            .climbIdx = (int32_t)agent_climb_idx,
            .climbJoint = climb_joint_out,
        };
    }
}

inline void loadCheckpointJsonIdxSystem(Engine &ctx, CheckpointReset &reset)
{
    // Only when a checkpoint is being loaded.
    if (reset.reset == 0) {
        return;
    }

    // Load the checkpoint's JSON level before the level is generated.
    Checkpoint& ckpt = ctx.singleton<Checkpoint>();

    ctx.singleton<JSONIndex>().index = ckpt.jsonIndex;
}

inline void loadCheckpointSystem(Engine &ctx, CheckpointReset &reset)
{
    // Only when a checkpoint is being loaded.
    if (reset.reset == 0) {
        return;
    }
    reset.reset = 0;

    Checkpoint& ckpt = ctx.singleton<Checkpoint>();

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    // Restore the episode state.
    episode_state.curStep = ckpt.curEpisodeStep;
    episode_state.curLevel = ckpt.curEpisodeLevel;
    episode_state.curJumps = ckpt.curEpisodeJumps;
    episode_state.activeWaypointIdx = ckpt.activeWaypointIdx;


    Entity agent_grab_entity = Entity::none();
    Entity agent_climb_entity = Entity::none();
    { // Load entities
        const Checkpoint::EntityState *entity_states = &ckpt.entityStates[0];

        CountT load_idx = 0;
        ctx.iterateQuery(ctx.data().simEntityQuery,
        [&](Entity e, EntityType type) {
            if (type == EntityType::None || type == EntityType::Agent || type == EntityType::Goal) {
                return;
            }

            if (load_idx == (CountT)ckpt.agentState.grabIdx) {
                agent_grab_entity = e;
            }
            if (load_idx == (CountT)ckpt.agentState.climbIdx) {
                agent_climb_entity = e;
            }

            const auto &state_in = entity_states[load_idx];
            assert(state_in.entityType == type);

            ctx.get<Position>(e) = state_in.position;
            ctx.get<Rotation>(e) = state_in.rotation;
            ctx.get<Velocity>(e) = state_in.velocity;

            switch (type) {
            case EntityType::Door: {
                ctx.get<OpenState>(e) = state_in.doorOpen;
            } break;
            case EntityType::Button: {
                ctx.get<ButtonState>(e) = state_in.button;
            } break;
            case EntityType::Truss: {
                ctx.get<IsTruss>(e) = state_in.truss;
            } break;
            case EntityType::Motion: {
                ConditionsList &conditions_list = ctx.get<ConditionsList>(e);
                conditions_list.numConditions = state_in.conditionsState.numConditions;
                ctx.iterateQuery(ctx.data().conditionQuery,
                [&](Entity eb, ButtonState bs) {
                    (void)bs;
                    for (uint32_t i = 0; i < state_in.conditionsState.numConditions; i++) {
                        if (state_in.conditionsState.conditions[i] == i) {
                            conditions_list.conditions[i] = eb;
                        }
                    }
                });
            } break;
            default: break;
            }

            load_idx += 1;
        });
    }

    {
        Entity agent = ctx.data().agent;
        const auto &state_in = ckpt.agentState;

        ctx.get<Position>(agent) = state_in.position;
        ctx.get<Rotation>(agent) = state_in.rotation;
        ctx.get<Orientation>(agent).r = state_in.orientation;
        ctx.get<Velocity>(agent) = state_in.velocity;
        ctx.get<Progress>(agent) = state_in.taskProgress;

        if (state_in.grabIdx != -1) {
            assert(agent_grab_entity != Entity::none());

            Entity joint_entity = PhysicsSystem::makeFixedJoint(
                ctx, agent, agent_grab_entity, {}, {}, {}, {}, 0.f);

            JointConstraint joint_constraint = state_in.grabJoint;
            joint_constraint.e1 = agent;
            joint_constraint.e2 = agent_grab_entity;

            ctx.get<JointConstraint>(joint_entity) = joint_constraint;

            ctx.get<GrabState>(agent).constraintEntity = joint_entity;
        }
        if (state_in.climbIdx != -1) {
            assert(agent_climb_entity != Entity::none());

            Entity joint_entity = PhysicsSystem::makeFixedJoint(
                ctx, agent, agent_climb_entity, {}, {}, {}, {}, 0.f);

            JointConstraint joint_constraint = state_in.climbJoint;
            joint_constraint.e1 = agent;
            joint_constraint.e2 = agent_climb_entity;

            ctx.get<JointConstraint>(joint_entity) = joint_constraint;

            ctx.get<ClimbState>(agent).constraintEntity = joint_entity;
            ctx.get<ClimbState>(agent).isClimbing = true;
            ctx.get<ClimbState>(agent).truss = agent_climb_entity;
        }
    }

}

// Runs each step. When the episode is finished (or external code forced a reset
// through the WorldReset singleton), start a new episode; the level systems
// below then replace the world.
inline void newEpisodeSystem(Engine &ctx, EpisodeState &episode_state)
{
    if (!episode_state.episodeFinished) {
        return;
    }

    ctx.singleton<WorldReset>().reset = 0;

    episode_state = initEpisodeState();

    // reachedExit makes cleanupLevelSystem and generateLevelSystem replace
    // the level.
    episode_state.reachedExit = true;

    initEpisodeRNG(ctx);

    // Sample this episode's time step.
    episode_state.deltaT = consts::minDeltaT +
    ctx.data().rng.sampleUniform() * (consts::deltaT - consts::minDeltaT);

    // The level's physics parameters are not loaded yet; use placeholders.
    float dummy_gravity = 0.0f;
    float dummy_air_res = 0.0f;
    PhysicsSystem::updatePhysicsStepParameters(ctx,
        episode_state.deltaT,
        consts::numPhysicsSubsteps,
        dummy_gravity,
        dummy_air_res);
}

inline PhysicsParameters getPhysicsParams(Engine &ctx)
{
    if(ctx.singleton<CurrentJSONIndex>().index == -1)
    {
        return {
            consts::gravity,
            consts::agent_move_force,
            consts::agent_jump_force,
            consts::air_res_constant,
            Vector3::zero(),
            Vector3::zero()
        };
    }

    return ctx.data().jsonLevels[ctx.singleton<CurrentJSONIndex>().index].physicsParams;

}

inline void updateDeltaTSystem(Engine &ctx, EpisodeState &episode_state)
{

    // Sample the time step, then apply the level's physics parameters.
    episode_state.deltaT = consts::minDeltaT +
    ctx.data().rng.sampleUniform() * (consts::deltaT - consts::minDeltaT);
    if(ctx.singleton<CurrentJSONIndex>().index == -1)
    {
        return;
    }

    PhysicsParameters physicsParams = getPhysicsParams(ctx);

    PhysicsSystem::updatePhysicsStepParameters(ctx,
        episode_state.deltaT,
        consts::numPhysicsSubsteps,
        physicsParams.gravity,
        physicsParams.airResistance);

    ctx.singleton<TimeStep>().delta = episode_state.deltaT;
}

inline void cleanupLevelSystem(Engine &ctx, const EpisodeState &episode_state)
{
    if (!episode_state.reachedExit) {
        return;
    }

    destroyLevel(ctx);
}

inline void deferredDeleteSystem(Engine &ctx, DeferredDelete deferred_delete)
{
    ctx.destroyRenderableEntity(deferred_delete.e);
}

inline void generateLevelSystem(Engine &ctx, EpisodeState &episode_state)
{
    if (!episode_state.reachedExit) {
        return;
    }

    episode_state.reachedExit = false;

    PhysicsSystem::reset(ctx);
    LevelType level_type = extractLevelType(ctx.data().simFlags);
    level_type = generateLevel(ctx, level_type);
    ctx.get<AgentLevelTypeObs>(ctx.data().agent) = AgentLevelTypeObs {
        .type = level_type,
    };
}

// True if the agent touches a wall in the requested move direction.
inline bool touchingWall(Engine &ctx, const Position &pos, const Vector3 &moveDir){

    float moveDirTolerance = 0.01f;

    // Shrink the agent's box slightly so that touching surfaces do not count.
    float agentBoundsTolerance = 0.999f;

    AABB agent_extended_aabb = {
        .pMin = pos - (agentBoundsTolerance * consts::agentExtents * 0.5f)
        + moveDir * moveDirTolerance,
        .pMax = pos + (agentBoundsTolerance * consts::agentExtents * 0.5f)
        + moveDir * moveDirTolerance,
    };

    bool touchingWall = false;
    // The box is shifted slightly in the requested direction.
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, agent_extended_aabb, [&](Entity e) {
            if (ctx.get<EntityType>(e) != EntityType::Agent) {
                touchingWall = true;
            }
    });

    return touchingWall;
}

inline bool onGround(Engine &ctx, const Position &pos, const Scale &s){

    AABB agent_jump_aabb = {
        .pMin = pos - (Vector3{s.d0, s.d1, s.d2} * 0.5f),
        .pMax = pos + (Vector3{s.d0, s.d1, s.d2} * 0.5f),
    };

    // A thin box just under the agent's feet, inset from the edges.
    agent_jump_aabb.pMax.z = agent_jump_aabb.pMin.z;
    agent_jump_aabb.pMin.z -= 0.001f;
    Vector3 edgeEpsilon = Vector3{0.5f, 0.5f, 0.0f};
    agent_jump_aabb.pMin += edgeEpsilon;
    agent_jump_aabb.pMax -= edgeEpsilon;

    bool onGround = false;
    float contact_normal_threshold = 0.99f;
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, agent_jump_aabb, [&](Entity e) {
            if (ctx.get<EntityType>(e) != EntityType::Agent &&
                ctx.get<EntityType>(e) != EntityType::Lava &&
                ctx.get<ResponseType>(e) != ResponseType::StaticNoCollide) {
                // Count only real contact with an upward-facing surface, not
                // a bounding-box overlap.
                if(ctx.get<AgentTouched>(e).touched && 
                ctx.get<AgentTouched>(e).normal.normalize().dot(math::up) > contact_normal_threshold) {
                    onGround = true;
                }
            }
    });

    return onGround;
}

inline bool withinTruss(Engine &ctx, const Position &pos, const Scale &s) {
    AABB agent_aabb = {
        .pMin = pos - (Vector3{s.d0, s.d1, s.d2} * 0.5f),
        .pMax = pos + (Vector3{s.d0, s.d1, s.d2} * 0.5f),
    };

    // Grow the box slightly so that touching a truss counts.
    Vector3 edgeEpsilon = Vector3{0.01f, 0.01f, 0.01f};
    agent_aabb.pMin -= edgeEpsilon;
    agent_aabb.pMax += edgeEpsilon;

    bool withinTruss = false;

    PhysicsSystem::findEntitiesWithinAABB(
            ctx, agent_aabb, [&](Entity e) {
            if (ctx.get<EntityType>(e) == EntityType::Truss) {
                withinTruss = true;
            }
    });

    return withinTruss;
}

// Jump: if the agent stands on a surface, apply a strong upward force for one
// time step.
inline void jumpSystem(Engine &ctx,
                           Action &action,
                           Position &pos,
                           Rotation &rot,
                           Scale &s,
                           ExternalForce &external_force,
                           JumpPerformed &jump_performed)
{
    Action a = ((ctx.data().simFlags & SimFlags::NoActionDelay) ==
        SimFlags::NoActionDelay) ? action : ctx.singleton<ActionLatch>().action;
    jump_performed.performed = 0;

    if (a.interact != 1) {
        return;
    };

    if (!onGround(ctx, pos, s)) {
        return;
    }

    PhysicsParameters physicsParams = getPhysicsParams(ctx);

    float jumpForce = physicsParams.jumpForce / ctx.singleton<EpisodeState>().deltaT;
    // The agent does not tip over, so this is the jump force along +z.
    external_force.z += rot.rotateVec({ 0.0f, 0.0f, jumpForce }).z;
    jump_performed.performed = 1;
    ctx.singleton<EpisodeState>().curJumps += 1;
}

// Turns the discrete Action into forces for the physics simulation.
inline void movementSystem(Engine &ctx,
                           Entity e,
                           Action &action,
                           Position &pos,
                           Rotation &rot,
                           Orientation &orientation,
                           Scale &s,
                           ClimbState &climb_state,
                           ExternalForce &external_force,
                           ExternalTorque &external_torque,
                           FrameLinearVelocity &frame_linear_vel)
{

    Action a = ((ctx.data().simFlags & SimFlags::NoActionDelay) ==
        SimFlags::NoActionDelay) ? action : ctx.singleton<ActionLatch>().action;

    frame_linear_vel.velocity = {0.0f, 0.0f, 0.0f};

    PhysicsParameters physicsParams = getPhysicsParams(ctx);

    float move_max = physicsParams.moveForce;

    // Climbing: stop when the agent jumps, leaves the truss or is on the ground.
    Vector3 leavingForce = Vector3 {0, 0, 0};
    if(climb_state.constraintEntity != Entity::none() &&
        (a.interact == 1 || !withinTruss(ctx, pos, s) || onGround(ctx, pos, s)))
    {
        // A jump pushes the agent away from the truss.
        if (a.interact == 1) {
            Vector3 fromTruss = climb_state.normal;
            fromTruss = fromTruss.normalize();
            leavingForce = fromTruss * consts::agent_truss_force;
            leavingForce.z = consts::agent_truss_vertical_force;
        }
        ctx.destroyEntity(climb_state.constraintEntity);
        climb_state.constraintEntity = Entity::none();
        climb_state.truss = Entity::none();
        climb_state.isClimbing = false;
        climb_state.normal = Vector3::zero();
    }

    // Move in one of consts::numMoveAngleBuckets directions.
    float move_amount = a.moveAmount *
        (move_max / (consts::numMoveAmountBuckets - 1));

    move_amount *= 0.04f / ctx.singleton<EpisodeState>().deltaT;

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numMoveAngleBuckets);

    float move_angle = float(a.moveAngle) * move_angle_per_bucket;
    float f_x = move_amount * sinf(move_angle);
    float f_y = move_amount * cosf(move_angle);

    // The direction is relative to the agent's orientation.
    Vector3 f_move = Vector3 {f_x, f_y, 0};
    f_move = orientation.r.rotateVec(f_move);

    float turn_dir = (a.rotate == 0) ? 0.f : (a.rotate == 1) ? 1.f : -1.f;


    if ((ctx.data().simFlags & SimFlags::UnlockAndRandomizeAgentOrientation) == 
                           SimFlags::UnlockAndRandomizeAgentOrientation) {
        // Turning rotates the frame of the agent's actions and observations by 5 degrees.
        Quat quat_turn = Quat::angleAxis(turn_dir * 5.f * math::pi / 180.0f, math::up).normalize();
        orientation.r = (quat_turn * orientation.r).normalize();
    }

    // While climbing, the agent moves only vertically, held by a joint to the truss.
    if(climb_state.isClimbing) {

        // Replace the previous joint.
        if (climb_state.constraintEntity != Entity::none()) {
            ctx.destroyEntity(climb_state.constraintEntity);
        }
        assert(climb_state.truss != Entity::none());
        // The joint keeps the agent on the truss.
        constexpr float separation = -1.25f;
        constexpr Quat attach1 = { 1, 0, 0, 0 };
        const Quat other_rot = ctx.get<Rotation>(climb_state.truss);
        const Vector3 other_pos = ctx.get<Position>(climb_state.truss);
        const Quat attach2 = (other_rot.inv() * rot).normalize();
        const Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;
        // Moving toward the truss climbs up, moving away climbs down; other
        // directions do not climb.
        const Vector3 moveVector = f_move;
        float move_sign = moveVector.dot(other_pos - pos);
        if (abs(move_sign) < 0.9f) {
            move_sign = 0.0f;
        }
        if (move_sign != 0.0f) {
            move_sign /= abs(move_sign);
        }
        const float move_z = 2.0f * move_sign * move_amount *
            math::sqr(ctx.singleton<EpisodeState>().deltaT);
        const Vector3 climbingPos = pos + Vector3 { 0, 0, move_z };
        const Vector3 r2 = other_rot.inv().rotateVec(climbingPos +
            0.5f * math::up - other_pos);
        climb_state.constraintEntity = PhysicsSystem::makeFixedJoint(ctx, e, climb_state.truss, attach1, attach2, r1, r2, separation);
        f_move = Vector3::zero();
    }
    external_force = f_move + leavingForce;
    external_torque = Vector3 { 0, 0, 0};
}

// Grab (interact == 2): cast a short ray in front of the agent and join it to
// the grabbable entity it hits.
inline void grabSystem(Engine &ctx,
                       Entity e,
                       Position pos,
                       Rotation rot,
                       Action action,
                       GrabState &grab)
{
    if (action.interact != 2) {
        return;
    }

    // While holding an entity, the grab action releases it.
    if (grab.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab.constraintEntity);
        grab.constraintEntity = Entity::none();
        return;
    }

    auto &bvh = ctx.singleton<broadphase::BVH>();
    float hit_t;
    Vector3 hit_normal;

    Vector3 ray_o = pos + 0.5f * math::up;
    Vector3 ray_d = rot.rotateVec(math::fwd);

    Entity grab_entity =
        bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.0f);

    if (grab_entity == Entity::none()) {
        return;
    }

    auto response_type = ctx.get<ResponseType>(grab_entity);
    if (response_type != ResponseType::Dynamic) {
        return;
    }

    auto entity_type = ctx.get<EntityType>(grab_entity);
    if (entity_type == EntityType::Agent) {
        return;
    }

    Vector3 other_pos = ctx.get<Position>(grab_entity);
    Quat other_rot = ctx.get<Rotation>(grab_entity);

    Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;

    Vector3 hit_pos = ray_o + ray_d * hit_t;
    Vector3 r2 =
        other_rot.inv().rotateVec(hit_pos - other_pos);

    Quat attach1 = { 1, 0, 0, 0 };
    Quat attach2 = (other_rot.inv() * rot).normalize();

    float separation = hit_t - 1.25f;

    grab.constraintEntity = PhysicsSystem::makeFixedJoint(ctx,
        e, grab_entity, attach1, attach2, r1, r2, separation);
}

// Moves each door down into the ground when open, and back up when closed.
inline void setDoorPositionSystem(Engine &,
                                  Position &pos,
                                  OpenState &open_state)
{
    if (open_state.isOpen) {
        if (pos.z > -6.5f) {
            pos.z += -consts::doorSpeed * consts::deltaT;
        }
    }
    else if (pos.z < 0.0f) {
        pos.z += consts::doorSpeed * consts::deltaT;
    }

    if (pos.z >= 0.0f) {
        pos.z = 0.0f;
    }
}

inline void enemyActSystem(Engine &ctx,
                           Position pos,
                           Rotation rot,
                           EnemyState enemy_state,
                           ExternalForce &ext_force,
                           ExternalTorque &ext_torque)
{
    Vector3 agent_pos = ctx.get<Position>(ctx.data().agent);

    Vector3 to_agent = agent_pos - pos;
    float dist_to_agent = to_agent.length();

    if (dist_to_agent <= 1e-5f || enemy_state.isDead) {
        ext_force = Vector3::zero();
        ext_torque = Vector3::zero();
        return;
    }

    if (enemy_state.isChicken) {
        ext_force = enemy_state.moveForce * rot.rotateVec(math::fwd) * 1.0f;
        // A chicken walks forward and turns at random.
        ext_torque = Vector3 {
            0.f,
            0.f,
            (ctx.data().rng.sampleUniform()*2.0f - 1.0f) *
             enemy_state.moveTorque * 2.0f,
        };
    } else {
        to_agent /= dist_to_agent;
        Vector3 to_agent_local = rot.inv().rotateVec(to_agent);
        float theta = -atan2f(to_agent_local.x, to_agent_local.y);
        ext_force = enemy_state.moveForce * rot.rotateVec(math::fwd);
        // Other enemies turn toward the agent.
        ext_torque = Vector3 {
            0.f,
            0.f,
            theta * enemy_state.moveTorque,
        };
    }
}

inline void enemyPostPhysicsSystem(Engine &ctx,
                                   Position pos,
                                   Velocity &vel,
                                   EnemyState &enemy_state)
{
    vel = { Vector3::zero(), Vector3::zero() };

    if (enemy_state.isChicken) {
        return;
    } else {
        Vector3 agent_pos = ctx.get<Position>(ctx.data().agent);
        Vector3 to_agent = agent_pos - pos;

        float dist_to_agent = to_agent.length();

        if (dist_to_agent > 2.5f * consts::agentRadius) {
            return;
        }
        ctx.singleton<EpisodeState>().isDead = true;
    }
}

// A button is pressed while the agent touches it or a dynamic entity is on it.
inline void buttonSystem(Engine &ctx,
                         Position pos,
                         AgentTouched &touched,
                         ButtonState &state,
                         EntityExtents extents)
{
    // A press button (not hold) stays pressed once it is pressed.
    if(!state.isHold && state.isPressed) {
        return;
    }

    AABB button_aabb = {
        .pMin = pos - Vector3 {
            extents.x * 0.5f,
            extents.y * 0.5f,
            extents.z * 0.5f,
        },
        .pMax = pos + Vector3 {
            extents.x * 0.5f,
            extents.y * 0.5f,
            extents.z * 0.5f,
        },
    };

    bool button_pressed = touched.touched;
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, button_aabb, [&](Entity e) {
        if (ctx.get<ResponseType>(e) != ResponseType::Dynamic) {
            return;
        }

        button_pressed = true;
    });

    state.isPressed = button_pressed;
}

// A pattern is matched while an entity is on top of it.
inline void patternSystem(Engine &ctx,
                          Position pos,
                          PatternMatchState &state)
{
    const float pattern_width = ctx.data().buttonWidth;
    const float pattern_press_size = 0.4f * pattern_width;

    AABB pattern_aabb {
        .pMin = pos + Vector3 {
            -pattern_press_size,
            -pattern_press_size,
            0.f,
        },
        .pMax = pos + Vector3 {
            pattern_press_size,
            pattern_press_size,
            0.3f
        },
    };

    bool pattern_matched = false;
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, pattern_aabb, [&](Entity e) {
        auto response_type_ref = ctx.getSafe<ResponseType>(e);

        if (!response_type_ref.valid() ||
                response_type_ref.value() != ResponseType::Dynamic) {
            return;
        }

        pattern_matched = true;
    });

    state.isMatched = pattern_matched;
}

// Open a door when all its linked buttons are pressed. A non-persistent door
// closes again when they are released.
inline void doorOpenSystem(Engine &ctx,
                           OpenState &open_state,
                           const DoorProperties &door_props,
                           ButtonListElem door_button_list,
                           PatternListElem door_pattern_list,
                           ChickenListElem door_chicken_list)
{
    bool all_pressed = true;

    Entity cur_button = door_button_list.next;
    while (cur_button != Entity::none()) {
        if (!ctx.get<ButtonState>(cur_button).isPressed) {
            all_pressed = false;
            break;
        }

        cur_button = ctx.get<ButtonListElem>(cur_button).next;
    }

    Entity cur_pattern = door_pattern_list.next;
    while (cur_pattern != Entity::none() && all_pressed) {
        if (!ctx.get<PatternMatchState>(cur_pattern).isMatched) {
            all_pressed = false;
            break;
        }

        cur_pattern = ctx.get<PatternListElem>(cur_pattern).next;
    }

    Entity cur_chicken = door_chicken_list.next;
    while (cur_chicken != Entity::none() && all_pressed) {
        if (!ctx.get<EnemyState>(cur_chicken).isDead) {
            all_pressed = false;
            break;
        }

        cur_chicken = ctx.get<ChickenListElem>(cur_chicken).next;
    }

    if (all_pressed) {
        open_state.isOpen = true;
    } else if (!door_props.isPersistent) {
        open_state.isOpen = false;
    }
}

inline void checkWaypointSystem(Engine &ctx, AgentTouched &touched, WaypointProperties &waypoint) {
    if (!touched.touched) {
        return;
    }

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    if (episode_state.isDead) {
        return;
    }

    // Only the active waypoint counts, and only within its jump limit.
    if (episode_state.activeWaypointIdx != waypoint.idx) {
        return;
    }
    // jumps <= 0 means no jump limit.
    if (waypoint.jumps > 0 && episode_state.curJumps > waypoint.jumps) {
        return;
    }

    episode_state.reachedWaypoint = true;

    // Waypoints count down to the exit, which has index 0.
    episode_state.activeWaypointIdx = episode_state.activeWaypointIdx > 0 ? 
                                      episode_state.activeWaypointIdx - 1 : 0;
}
inline void updateWaypointSystem(Engine &ctx, Entity e, WaypointProperties &idx) {
    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    Level &level = ctx.singleton<Level>();

    if (episode_state.activeWaypointIdx == idx.idx) {
        level.waypoint = e;
    }
}

inline void checkExitSystem(Engine &ctx, AgentTouched &touched, IsExit) {
    if (!touched.touched) {
        return;
    }

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    if (episode_state.isDead) {
        return;
    }

    episode_state.reachedExit = true;
}

inline void clearTouchedSystem(Engine& ctx, AgentTouched &touched) {
    (void)ctx;
    touched.touched = false;
    touched.normal = Vector3::zero();
}

inline float maxAbs(const float a, const float b) {
    return (std::abs(a) > std::abs(b)) ? a : b;
}

inline Vector3 maxAbsVec3(const Vector3 a, const Vector3 b) {
    return Vector3 {
        maxAbs(a.x, b.x),
        maxAbs(a.y, b.y),
        maxAbs(a.z, b.z),
    };
}

inline void viewBoxSystem(Engine &ctx, Position& p, const IsViewBox){
    p = ctx.get<Position>(ctx.data().agent);
}

inline void orientationArrowSystem(Engine &ctx, Position& p, Rotation& r, const IsOrientationArrow){
    p = ctx.get<Position>(ctx.data().agent) + Vector3(0, 0, consts::agentExtents.z * 0.5f + 0.1f);
    r = ctx.get<Orientation>(ctx.data().agent).r;
    p += r.rotateVec(Vector3(0, consts::agentExtents.y * 0.25f, 0));
}

inline void trussSystem(Engine &ctx, const Entity e, const AgentTouched touched, const IsTruss&){
    // The agent climbs only when it touches the truss from the side.
    if (!touched.touched || std::abs(touched.normal.z) > 0.01f) {
        return;
    }
    auto &agentClimbState = ctx.get<ClimbState>(ctx.data().agent);
    agentClimbState.isClimbing = true;
    agentClimbState.truss = e;
    agentClimbState.normal = touched.normal;
}

inline void impulseObjectSystem(Engine &ctx, const AgentTouched &touched, const ImpulseObjectState impulse_object_state) {
    if (!touched.touched) {
        return;
    }

    // Trampolines and conveyors: per component, the agent's force is at least
    // the object's bounce. bounce is a velocity; dividing by deltaT turns it
    // into a force for a unit-mass agent.
    auto &agentForce = ctx.get<ExternalForce>(ctx.data().agent);
    agentForce = maxAbsVec3(agentForce, impulse_object_state.bounce / ctx.singleton<EpisodeState>().deltaT);
}

inline void lavaSystem(Engine &ctx, AgentTouched &touched, LavaProperties lava_properties)
{
    if (!touched.touched) {
        return;
    }

    auto &agentHealth = ctx.get<Health>(ctx.data().agent);
    agentHealth.amount -= lava_properties.damage;
    if (agentHealth.amount <= 0) {
        ctx.singleton<EpisodeState>().isDead = true;
    }
}

inline bool conditionsSatisfied(Engine &ctx, ConditionsList conditions)
{
    bool all_pressed = true;
    for(uint32_t i = 0; i < conditions.numConditions; i++)
    {
        // Only buttons can be conditions.
        all_pressed = all_pressed && ctx.get<ButtonState>(conditions.conditions[i]).isPressed;
    }
    return all_pressed;
}

inline void blockMotionSystem(Engine &ctx, Position &pos, Velocity &vel, AgentTouched &touched,
    ConditionsList conditions, MotionState &motion)
{
    if(!conditionsSatisfied(ctx, conditions) ||
        (motion.loopsCompleted > motion.loops && motion.loops != -1)) {
        vel.linear = Vector3::zero();
        vel.angular = Vector3::zero();
        return;
    }


    // At either end of the path, reverse or restart. For rotating blocks
    // start_pos equals end_pos, so this keeps them in place.
    float p = (pos - motion.start_pos).dot(motion.end_pos - motion.start_pos);
    Vector3 default_vel = (motion.end_pos - motion.start_pos) / motion.time;
    if (p <= 0.0f) {
        vel.linear = default_vel;
        if (p < 0.0f) {
            motion.loopsCompleted++;
        }
    }
    else if (p > (motion.end_pos - motion.start_pos).length2()) {
        if(motion.reverses) {
            vel.linear = -default_vel;
        } else {
            motion.loopsCompleted++;
            // Jump back to the start if loops remain.
            if (motion.loopsCompleted <= motion.loops || motion.loops == -1) {
                pos = motion.start_pos;
            }
            return;
        }
    }
    Vector3 v = vel.linear;
    vel.angular = motion.angular_vel;

    // An agent on the block moves with it.
    if (touched.touched) {
        ctx.get<FrameLinearVelocity>(ctx.data().agent).velocity = v;

        Vector3 agent_vel = ctx.get<Velocity>(ctx.data().agent).linear;
        Vector3 agent_vel_block_relative = agent_vel - v;

        Vector3 inputAirResVel = fmin(agent_vel_block_relative.length(), 2 * v.length()) * agent_vel_block_relative.normalize();

        Vector3 air_res_block_relative = airResistanceForce(inputAirResVel);

        AirResForce &air_res_force = ctx.get<AirResForce>(ctx.data().agent);
        // If the agent touches two moving blocks, both write here; either value is fine.
        air_res_force.airResistance = air_res_block_relative;
        air_res_force.isSet = true;
    }
}

// Zero the agent's angular velocity after each step, so that it does not spin.
inline void agentZeroVelSystem(Engine &ctx,
                               Velocity &vel,
                               Action &action)
{
    (void)ctx;
    (void)action;

    vel.angular = Vector3::zero();
    return;
}


// The agent dies when it falls below the level.
inline void agentFallSystem(Engine &ctx,
                               Position &pos,
                               Action &action)
{
    (void)action;
    const AABB &level_aabb = ctx.singleton<LevelAABB>();

    // More than two body heights below the level's bounds.
    if (pos.z < level_aabb.pMin.z - (2.0f * consts::agentExtents.z))
    {
        ctx.singleton<EpisodeState>().isDead = true;
    }
}

inline void dynamicObjectResetSystem(Engine &ctx,
                               Position &pos,
                               EntityExtents &extents,
                               ResponseType &response_type)
{

    if (response_type != ResponseType::Dynamic) {
        return;
    }

    const AABB &level_aabb = ctx.singleton<LevelAABB>();

    // A dynamic block that falls more than five block heights below the
    // level's bounds ends the episode (equivalent to respawning the block for a
    // policy with no memory).
    if (pos.z < level_aabb.pMin.z - (5.0f * extents.z))
    {
        ctx.singleton<EpisodeState>().isDead = true;
    }
}

// Divide components of a vector by the lengths in the AABB.
inline Vector3 normalizeByAABB(const Vector3 v, const AABB &aabb) {
    return Vector3 {
        v.x / (aabb.pMax.x - aabb.pMin.x),
        v.y / (aabb.pMax.y - aabb.pMin.y),
        v.z / (aabb.pMax.z - aabb.pMin.z),
    };
}

inline AABB AABBToAgentCenter(AABB aabb, Position pos, Quat rot) {
    Vector3 rotatedPMin = rot.rotateVec(aabb.pMin - pos);
    Vector3 rotatedPMax = rot.rotateVec(aabb.pMax - pos);
    // Rotation can swap the corners; recompute min and max.
    AABB rotatedAABB = {
        {
            std::min(rotatedPMin.x, rotatedPMax.x),
            std::min(rotatedPMin.y, rotatedPMax.y),
            std::min(rotatedPMin.z, rotatedPMax.z)
        },
        {
            std::max(rotatedPMin.x, rotatedPMax.x),
            std::max(rotatedPMin.y, rotatedPMax.y),
            std::max(rotatedPMin.z, rotatedPMax.z)
        }
    };
    return rotatedAABB;
}

inline Vector3 extentsToAgentView(Vector3 extents, Quat rot) {
    // Transform all the corners and then compute the extents.
    Vector3 corners[8] = {
        Vector3 { extents.x, extents.y, extents.z },
        Vector3 { extents.x, extents.y, -extents.z },
        Vector3 { extents.x, -extents.y, extents.z },
        Vector3 { extents.x, -extents.y, -extents.z },
        Vector3 { -extents.x, extents.y, extents.z },
        Vector3 { -extents.x, extents.y, -extents.z },
        Vector3 { -extents.x, -extents.y, extents.z },
        Vector3 { -extents.x, -extents.y, -extents.z },
    };
    Vector3 eMin = Vector3 { FLT_MAX, FLT_MAX, FLT_MAX };
    Vector3 eMax = Vector3 { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (const auto corner : corners) {
        Vector3 rotatedCorner = rot.rotateVec(corner);
        eMin = Vector3::min(eMin, rotatedCorner);
        eMax = Vector3::max(eMax, rotatedCorner);
    }
    return (eMax - eMin);
}

static inline EntityLabel typeToLabel(EntityType type)
{
    if constexpr (!consts::useBasicLabels) {
        return (EntityLabel) type;
    } else {
        // Basic labels: none, avoid (lava) or normal.
        if (type == EntityType::None) {
            return (EntityLabel) EntitySemantics::None;
        } else if (type == EntityType::Lava) {
            return (EntityLabel) EntitySemantics::Avoid;
        }
        return (EntityLabel) EntitySemantics::Normal;
    }
}

// Collect the agent's egocentric observations for the policy.
inline void collectObservationsSystem(
    Engine &ctx,
    Position pos,
    Orientation orientation,
    Velocity vel,
    Progress progress,
    Action action,
    const GrabState &grab,
    AgentTxfmObs &agent_txfm_obs,
    AgentInteractObs &agent_interact_obs,
    AgentExitObs &agent_exit_obs,
    StepsRemainingObservation &steps_remaining_obs,
    EntityPhysicsStateObsArray &entity_physics_obs,
    EntityTypeObsArray &entity_type_obs
    )
{
    // Observations are in the agent's orientation frame.
    Quat to_view = orientation.r.inv();

    const Level &level = ctx.singleton<Level>();

    const bool useNormalizedAABBObs = ((ctx.data().simFlags & SimFlags::UseNormalizedAABBObs) ==
                                        SimFlags::UseNormalizedAABBObs);

    AABB level_aabb = ctx.singleton<LevelAABB>();

    // Measure from the agent's chest height.
    Vector3 adjustedAgentPos = pos + consts::agentHeightCorrection;

    AABB ignore_aabb = AABB {
        .pMin = Vector3::zero(),
        .pMax = Vector3::zero(),
    };

    // Latch this step's action; it is observed and, with action delay, applied next step.
    Action &latchAction = ctx.singleton<ActionLatch>().action;
    latchAction = action;
    auto health = ctx.get<Health>(ctx.data().agent);

    // The room position, level box and heading are not observed (zeros).
    agent_txfm_obs = AgentTxfmObs {
        .localRoomPos = Vector3::zero(),
        .levelAABB = ignore_aabb,
        .theta = 0.f,
        .linearVelocity = to_view.rotateVec(vel.linear),
        .activeWaypointIdx = (float)ctx.singleton<EpisodeState>().activeWaypointIdx,
        .action = latchAction,
        .health = health.amount
    };

    agent_interact_obs = AgentInteractObs {
        .isGrabbing = grab.constraintEntity != Entity::none() ? 1 : 0,
    };

    // Offset to the active waypoint (the exit once all waypoints are reached).
    Vector3 to_exit = ctx.get<Position>(level.waypoint) - adjustedAgentPos;
    PolarObs toExitPolar = xyzToPolar(to_view.rotateVec(to_exit));


    // Normalized observations.
    PolarObs normalizedToExitPolar = toExitPolar;
    normalizedToExitPolar.r = 1.0f;

    float normalizedMinDistToExit = progress.minDistToExit / progress.startDistToExit;

    agent_exit_obs = {
        .toExitPolar = useNormalizedAABBObs ? normalizedToExitPolar : toExitPolar,
        .minDistToExit = useNormalizedAABBObs ? normalizedMinDistToExit : progress.minDistToExit,
    };

    const bool disableSteps = ((ctx.data().simFlags & SimFlags::DisableStepObs) ==
                                        SimFlags::DisableStepObs);

    uint32_t steps_remaining = ctx.data().episodeLen - ctx.singleton<EpisodeState>().curStep;
                                        
    // Always record the steps remaining, even when the policy does not observe them.
    ctx.singleton<StepsRemaining>().t = steps_remaining;

    steps_remaining_obs.t = disableSteps ? 0 : steps_remaining;

    const bool disablePhysObs = ((ctx.data().simFlags & SimFlags::DisablePhysObs) ==
                                        SimFlags::DisablePhysObs);

    // Entity observations: entities inside the level box, or inside the view box
    // around the agent when the normalized AABB observations are on.
    auto viewExtents = Vector3{consts::obsAABBFactor, consts::obsAABBFactor, consts::obsAABBFactor};
    AABB view_aabb = {
        .pMin = pos - viewExtents,
        .pMax = pos + viewExtents,
    };
    ctx.singleton<DynamicBlockPos>().pos = Vector3::zero();
    CountT cur_obs_idx = 0;
    const AABB &entity_aabb = useNormalizedAABBObs ? view_aabb : level_aabb;
    PhysicsSystem::findEntitiesWithinAABB(ctx, entity_aabb, [&]
    (Entity e)
    {
        EntityType type = ctx.get<EntityType>(e);

        if (type == EntityType::None ||
                type == EntityType::Agent) {
            return;
        }

        Vector3 entity_pos = ctx.get<Position>(e);
        Quat entity_rot = ctx.get<Rotation>(e);
        Vector3 entity_extents = ctx.get<EntityExtents>(e);

        // Record the position of the dynamic object (if several, the last one wins).
        if (ctx.get<ResponseType>(e) == ResponseType::Dynamic) {
            ctx.singleton<DynamicBlockPos>().pos = entity_pos;
        }

        if (disablePhysObs || (cur_obs_idx == consts::maxObservationsPerAgent)) {
            return;
        }

        Vector3 linear_velocity;
        Vector3 angular_velocity;
        {
            auto vel_ref = ctx.getSafe<Velocity>(e);

            if (!vel_ref.valid()) {
                linear_velocity = Vector3::zero();
                angular_velocity = Vector3::zero();
            } else {
                linear_velocity = vel_ref.value().linear;
                angular_velocity = vel_ref.value().angular;
            }
        }

        entity_physics_obs.obs[cur_obs_idx] = EntityPhysicsStateObs {
            .positionPolar = xyzToPolar(to_view.rotateVec(entity_pos - adjustedAgentPos)),
            .velocityPolar = xyzToPolar(to_view.rotateVec(linear_velocity)),
            .angVelocityPolar = xyzToPolar(to_view.rotateVec(angular_velocity)),
            .extents = entity_extents,
            .entityRotation = quatToEuler(
                (to_view * entity_rot).normalize()),
        };

        entity_type_obs.obs[cur_obs_idx] = EntityTypeObs {
            .entityType = typeToLabel(type),
        };

        cur_obs_idx += 1;
    });

    // Sort the observations by decreasing y (the agent's forward axis). The
    // simple O(n^2) sort is fine for at most consts::maxObservationsPerAgent entries.
    if (!disablePhysObs)
    {
        for (int i = 0; i < cur_obs_idx; ++i)
        {
            for (int j = i; j < cur_obs_idx; ++j)
            {
                float iYPos = polarToXYZ(entity_physics_obs.obs[i].positionPolar).y;
                float jYPos = polarToXYZ(entity_physics_obs.obs[j].positionPolar).y;

                if (jYPos > iYPos)
                {
                    EntityPhysicsStateObs p = entity_physics_obs.obs[i];
                    entity_physics_obs.obs[i] = entity_physics_obs.obs[j];
                    entity_physics_obs.obs[j] = p;

                    EntityTypeObs t = entity_type_obs.obs[i];
                    entity_type_obs.obs[i] = entity_type_obs.obs[j];
                    entity_type_obs.obs[j] = t;
                }
            }
        }
    }

    for (; cur_obs_idx < consts::maxObservationsPerAgent; cur_obs_idx++) {
        entity_physics_obs.obs[cur_obs_idx] = EntityPhysicsStateObs {
            .positionPolar = { 0.f, 0.f, 0.f },
            .velocityPolar = { 0.f, 0.f, 0.f },
            .extents = Vector3::zero(),
            .entityRotation = Vector3::zero(),
        };

        entity_type_obs.obs[cur_obs_idx] = EntityTypeObs {
            .entityType = typeToLabel(EntityType::None),
        };
    }
}

// Traces consts::numLidarSamples lidar rays per agent. On the GPU, a warp runs
// each invocation and each thread traces some of the rays.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        LidarDepth &lidar_depth,
                        LidarHitType &lidar_hit_type)
{

    Vector3 pos = ctx.get<Position>(e) + consts::agentHeightCorrection;
    Quat rot = ctx.get<Orientation>(e).r;

    auto &bvh = ctx.singleton<broadphase::BVH>();

    const bool disableLidar = ((ctx.data().simFlags & SimFlags::DisableLidar) ==
                                        SimFlags::DisableLidar); 

    auto traceRay = [&](int32_t idx, float maxDist) {
        // Without the full sphere, use the lower half of a 2N-point lattice.
        int i = consts::useFullLidarSphere ?
            idx : idx + consts::numLidarSamples;

        const int numSamples = consts::useFullLidarSphere ?
            consts::numLidarSamples : 2 * consts::numLidarSamples;

        // Spherical Fibonacci lattice, in the agent's orientation frame.
        const float PHI = sqrtf(5) * 0.5 + 0.5;
        float phi = (2 * pi * i) / PHI;
        float theta = acosf(1.f - (2.f * i) / numSamples);
        Vector3 ray_dir = Vector3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)).normalize();

        ray_dir = rot.rotateVec(ray_dir);

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity = bvh.traceRay(pos, ray_dir, &hit_t,
            &hit_normal, maxDist);

        Entity lidarRay;
        Entity lidarMark;
        float missDistance = consts::agentExtents.z * 0.5f;

        if (consts::visualizeLidar && idx % 2 == 0) {
            // Draw every other ray and its hit point.
            lidarRay = ctx.data().lidarRays[idx / 2];
            lidarMark = ctx.data().lidarMarks[idx / 2];

            Vector3 rotation_axis = cross(up, ray_dir).normalize();
            float rotation_angle = acosf(dot(up, ray_dir));
            ctx.get<Rotation>(lidarRay) = Quat::angleAxis(rotation_angle, rotation_axis);

            Vector3 hit_pos = pos + (hit_entity == Entity::none() ? missDistance : hit_t) * ray_dir;
            Vector3 mean_pos = (pos + hit_pos) * 0.5f;
            ctx.get<Position>(lidarRay) = mean_pos;
            ctx.get<Position>(lidarMark) = hit_pos;
        }

        // A miss (or disabled lidar) reads as depth 0 and type None.
        if (hit_entity == Entity::none() || disableLidar) {
            lidar_depth.samples[idx] = 0.f;
            lidar_hit_type.samples[idx] = typeToLabel(EntityType::None);
            if (consts::visualizeLidar) {
                ctx.get<Scale>(lidarRay) = Diag3x3{0.05f, 0.05f, missDistance};
                ctx.get<Scale>(lidarMark) = Diag3x3{0.0f, 0.0f, 0.f};
                ctx.get<ObjectID>(lidarRay) = ObjectID { (int32_t)SimObject::Cube };
            }
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar_depth.samples[idx] = hit_t;
            lidar_hit_type.samples[idx] = typeToLabel(entity_type);

            if (consts::visualizeLidar) {
                ctx.get<Scale>(lidarRay) = Diag3x3{0.05f, 0.05f, hit_t};
                ctx.get<ObjectID>(lidarRay) = ObjectID { (int32_t)SimObject::CyanCube };
                ctx.get<ObjectID>(lidarMark) = ObjectID { (int32_t)SimObject::PurpleCube };
                ctx.get<Scale>(lidarMark) = Diag3x3{0.2f, 0.2f, 0.2f};
            }
        }
    };

    // With the normalized AABB observations, rays stop at the view box.
    float maxDist = ((ctx.data().simFlags & SimFlags::UseNormalizedAABBObs) ==
            SimFlags::UseNormalizedAABBObs) ? consts::obsAABBFactor : 200.0f;

#ifdef MADRONA_GPU_MODE
    // Each of the 32 threads in the warp traces every 32nd ray.
    int32_t idx = threadIdx.x % 32;
    for (int i = 0; i <= (consts::numLidarSamples - 1) / 32; ++i) {
        int rayIdx = idx + i * 32;
        if (rayIdx < consts::numLidarSamples) {
            traceRay(rayIdx, maxDist);
        }
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i, maxDist);
    }
#endif
}

inline void dense1withTrajRewardSystem(Engine &ctx,
                                        Position pos,
                                        Progress &progress,
                                        Reward &out_reward)
{
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    AABB agent_aabb = {
    .pMin = pos - (consts::agentExtents * 0.5f),
    .pMax = pos + (consts::agentExtents * 0.5f),
    };

    Vector3 exit_pos = ctx.get<Position>(lvl.exit);
    Vector3 exit_extents = ctx.get<EntityExtents>(lvl.exit);

    AABB exit_aabb = {
    .pMin = exit_pos - (exit_extents * 0.5f),
    .pMax = exit_pos + (exit_extents * 0.5f),
    };

    float dist_to_exit = sqrtf(agent_aabb.distance2(exit_aabb));

    float min_dist = progress.minDistToExit;
    float start_dist = progress.startDistToExit;

    float reward = 0.f;
    if (dist_to_exit < min_dist) {
    // Pay for new progress, normalized by the starting distance.
    float diff = (min_dist - dist_to_exit) / start_dist;
    reward += diff * ctx.data().rewardPerDist;

    progress.minDistToExit = dist_to_exit;
    }

    if (episode_state.reachedExit) {
        reward += 10.f;
    }
    
    // With a reference trajectory, add a reward for staying close to it.
    const auto &worldTraj = ctx.singleton<WorldTrajectory>();

    if (worldTraj.exists) {
        float alpha = 3.f;  // scale of the trajectory reward
        float min_traj_dist = FLT_MAX;
        int closest_pt_idx = -1;
        int num_base = consts::maxTrajectoryObservations;

        // Nearest trajectory point.
        for (int sample_i = 0; sample_i < num_base; sample_i++) {
            float traj_dist = (pos - worldTraj.observations[sample_i]).length();
            if (traj_dist < min_traj_dist) {
                min_traj_dist = traj_dist;
                closest_pt_idx = sample_i;
            }
        }
        
        if (closest_pt_idx >= 0) {
            // Falls off with the distance to that point.
            reward += alpha * (1.0f / (1.0f + min_traj_dist));
        }
    }
    out_reward.v = reward;   
}

inline void handTunedTrajRewardSystem(Engine &ctx,
                                        Reward &out_reward)
{
    const TrajectorySection &trajSection = ctx.singleton<TrajectorySection>();

    // TODO: placeholder reward: the sum of the offsets to the closest,
    // previous and next trajectory points.
    float reward = trajSection.closest.length() + trajSection.previous.length() + trajSection.next.length();
    
    out_reward.v = reward;   
}


inline void denseNoGoalRewardSystem(Engine &ctx,
                               Position pos,
                               Progress &progress,
                               Reward &out_reward)
{
    const auto &lvl = ctx.singleton<Level>();

    AABB agent_aabb = {
        .pMin = pos - (consts::agentExtents * 0.5f),
        .pMax = pos + (consts::agentExtents * 0.5f),
    };

    Vector3 exit_pos = ctx.get<Position>(lvl.exit);
    Vector3 exit_extents = ctx.get<EntityExtents>(lvl.exit);

    AABB exit_aabb = {
        .pMin = exit_pos - (exit_extents * 0.5f),
        .pMax = exit_pos + (exit_extents * 0.5f),
    };

    float dist_to_exit = sqrtf(agent_aabb.distance2(exit_aabb));

    float min_dist = progress.minDistToExit;
    float start_dist = progress.startDistToExit;

    float reward = 0.f;
    if (dist_to_exit < min_dist) {
        // Pay for new progress, normalized by the starting distance.
        float diff = (min_dist - dist_to_exit) / start_dist;
        reward += diff * ctx.data().rewardPerDist;

        progress.minDistToExit = dist_to_exit;
    }

    out_reward.v = reward;
}

inline void sparseRewardSystem(Engine &ctx,
                               Position pos,
                               Progress &progress,
                               Reward &out_reward)
{
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    AABB agent_aabb = {
        .pMin = pos - (consts::agentExtents * 0.5f),
        .pMax = pos + (consts::agentExtents * 0.5f),
    };

    Vector3 waypoint_pos = ctx.get<Position>(lvl.waypoint);
    Vector3 waypoint_extents = ctx.get<EntityExtents>(lvl.waypoint);

    AABB waypoint_aabb = {
        .pMin = waypoint_pos - (waypoint_extents * 0.5f),
        .pMax = waypoint_pos + (waypoint_extents * 0.5f),
    };

    float dist_to_waypoint = sqrtf(agent_aabb.distance2(waypoint_aabb));

    float reward = 0.f;

    // Each waypoint pays 10 / numWaypoints (set in level_gen.cpp). Reaching the
    // exit pays it too, even with waypoints skipped, since some waypoints can
    // be unreachable.
    if (episode_state.reachedWaypoint || episode_state.reachedExit) {
        reward += 10.0f / (float)episode_state.numWaypoints;
        // Measure progress from here toward the next waypoint.
        progress.minDistToExit = dist_to_waypoint;
        progress.startDistToExit = dist_to_waypoint;
    }

    out_reward.v = reward;
}


inline void dense1RewardSystem(Engine &ctx,
                               Position pos,
                               Progress &progress,
                               Reward &out_reward)
{
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    AABB agent_aabb = {
        .pMin = pos - (consts::agentExtents * 0.5f),
        .pMax = pos + (consts::agentExtents * 0.5f),
    };

    Vector3 waypoint_pos = ctx.get<Position>(lvl.waypoint);
    Vector3 waypoint_extents = ctx.get<EntityExtents>(lvl.waypoint);

    AABB waypoint_aabb = {
        .pMin = waypoint_pos - (waypoint_extents * 0.5f),
        .pMax = waypoint_pos + (waypoint_extents * 0.5f),
    };

    float dist_to_waypoint = sqrtf(agent_aabb.distance2(waypoint_aabb));

    float min_dist = progress.minDistToExit;
    float start_dist = progress.startDistToExit;

    float reward = 0.f;
    if (dist_to_waypoint < min_dist) {
        // Pay for new progress, normalized by the starting distance.
        float diff = (min_dist - dist_to_waypoint) / start_dist;
        reward += diff * ctx.data().rewardPerDist;

        progress.minDistToExit = dist_to_waypoint;
    }
    // Each waypoint pays 10 / numWaypoints (set in level_gen.cpp). Reaching the
    // exit pays it too, even with waypoints skipped, since some waypoints can
    // be unreachable.
    if (episode_state.reachedWaypoint || episode_state.reachedExit) {
        reward += 10.0f / (float)episode_state.numWaypoints;
        // Measure progress from here toward the next waypoint.
        progress.minDistToExit = dist_to_waypoint;
        progress.startDistToExit = dist_to_waypoint;
    }

    out_reward.v = reward;
}

// Dense reward for the lava-button room.
inline void lavaButtonRewardSystem(Engine &ctx,
                               Position pos,
                               Progress &progress,
                               Reward &out_reward)
{
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    // These levels have a single button; the query keeps the last one found.
    Vector3 button_pos;
    bool button_is_pressed;
    ctx.iterateQuery(ctx.data().buttonQuery, [
        &button_pos, &button_is_pressed
    ](Position cur_button_pos, ButtonState cur_button_state) {
        button_pos = cur_button_pos;
        button_is_pressed = cur_button_state.isPressed;
    });

    // These levels have a single door.
    bool door_is_open;
    ctx.iterateQuery(ctx.data().doorQuery, [&door_is_open](OpenState &openState)
    {
        door_is_open = openState.isOpen;
    });

    float reward = 0.f;
    // Once the door is open, reward progress toward the exit.
    if (door_is_open) {
        float dist_to_exit = pos.distance(ctx.get<Position>(lvl.exit));

        float min_dist = progress.minDistToExit;

        if (dist_to_exit < min_dist) {
            float diff = min_dist - dist_to_exit;
            reward += diff * ctx.data().rewardPerDist;

            progress.minDistToExit = dist_to_exit;
        }
        reward += 0.02f;
    } else {
        float dist_to_button = pos.distance(button_pos);
            // Otherwise reward progress toward the button.
            float min_dist = progress.minDistToButton;

            if (dist_to_button < min_dist) {
                float diff = min_dist - dist_to_button;
                reward += diff * ctx.data().rewardPerDist;

                progress.minDistToButton = dist_to_button;
            }
            reward += 0.01f;
    }

    if (episode_state.isDead) {
        reward -= 1.f;
    }

    if (episode_state.reachedExit) {
        reward += 1.f;

        if (episode_state.curLevel == ctx.data().levelsPerEpisode) {
            reward += 10.f;
        }
    }

    out_reward.v = reward;
}

// Dense reward for block-button puzzles.
inline void dense2RewardSystem(Engine &ctx,
                               Position pos,
                               GrabState grab_state,
                               Progress &progress,
                               Reward &out_reward)
{
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    // These levels have a single button; the query keeps the last one found.
    Vector3 button_pos;
    bool button_is_pressed;
    ctx.iterateQuery(ctx.data().buttonQuery, [
        &button_pos, &button_is_pressed
    ](Position cur_button_pos, ButtonState cur_button_state) {
        button_pos = cur_button_pos;
        button_is_pressed = cur_button_state.isPressed;
    });

    bool is_grabbing = grab_state.constraintEntity != Entity::none();

    float reward = 0.f;
    if (button_is_pressed && progress.minDistToButton < 5.0f && !is_grabbing) {
        float dist_to_exit = pos.distance(ctx.get<Position>(lvl.exit));

        float min_dist = progress.minDistToExit;

        if (dist_to_exit < min_dist) {
            float diff = min_dist - dist_to_exit;
            reward += diff * ctx.data().rewardPerDist;

            progress.minDistToExit = dist_to_exit;
        }
        reward += 0.02f;
    } else {
        float dist_to_button = pos.distance(button_pos);
        if (is_grabbing) {
            // While grabbing the block, reward progress toward the button.
            float min_dist = progress.minDistToButton;

            if (dist_to_button < min_dist) {
                float diff = min_dist - dist_to_button;
                reward += diff * ctx.data().rewardPerDist;

                progress.minDistToButton = dist_to_button;
            }
            reward += 0.01f;
        } else {
            // No reward until the agent grabs the block; record the starting
            // distance on the first step.
            if (progress.minDistToButton == 0.f) {
                progress.minDistToButton = dist_to_button;
            }
        }
    }

    if (episode_state.isDead) {
        reward -= 1.f;
    }

    if (episode_state.reachedExit) {
        reward += 1.f;

        if (episode_state.curLevel == ctx.data().levelsPerEpisode) {
            reward += 10.f;
        }
    }

    out_reward.v = reward;
}

inline void perGoalRewardSystem(Engine &ctx,
                                 Reward &out_reward)
{
    const auto &episode_state = ctx.singleton<EpisodeState>();

    float reward = 0.f;

    if (episode_state.isDead) {
        reward -= 1.f;
    }

    if (episode_state.reachedWaypoint) {
        reward += ctx.data().episodeLen - ctx.singleton<EpisodeState>().curStep;
    }

    out_reward.v = reward;
}


inline void perLevelRewardSystem(Engine &ctx,
                                 Reward &out_reward)
{
    const auto &episode_state = ctx.singleton<EpisodeState>();

    float reward = 0.f;

    if (episode_state.isDead) {
        reward -= 1.f;
    }

    if (episode_state.reachedExit) {
        reward += 1.f;

        if (episode_state.curLevel == ctx.data().levelsPerEpisode) {
            reward += 10.f;
        }
    }

    out_reward.v = reward;
}

inline void endOnlyRewardSystem(Engine &ctx,
                                Reward &out_reward)
{
    const auto &episode_state = ctx.singleton<EpisodeState>();

    float reward = 0.f;

    if (episode_state.reachedExit) {
        reward += 1.f;
    }

    out_reward.v = reward;
}

inline void updateEpisodeStateSystem(Engine &ctx,
                                     EpisodeState &episode_state)
{
    episode_state.curStep += 1;

    if (episode_state.reachedExit) {
        episode_state.curLevel += 1;
    }

    WorldReset force_reset = ctx.singleton<WorldReset>();
    CheckpointReset ckpt_reset = ctx.singleton<CheckpointReset>();

    bool success =
        episode_state.curLevel == ctx.data().levelsPerEpisode;

    episode_state.episodeFinished =
        force_reset.reset == 1 ||
        ckpt_reset.reset == 1 ||
        episode_state.isDead ||
        success
    ;

    if ((ctx.data().simFlags & SimFlags::IgnoreEpisodeLength) !=
            SimFlags::IgnoreEpisodeLength) {
        if (episode_state.curStep == ctx.data().episodeLen) {
            episode_state.episodeFinished = true;
        }
    }

    EpisodeResult &episode_result = ctx.singleton<EpisodeResult>();
    if (success) {
        episode_result.score =
            float(ctx.data().episodeLen - episode_state.curStep) /
            ctx.data().episodeLen;
    } else {
        episode_result.score = 0.f;
    }
}

// Set done = 1 on the last step of an episode (death, success or the step
// limit), and record whether the episode succeeded.
inline void writeDoneSystem(Engine &ctx,
                            Done &done)
{
    EpisodeState &episode_state = ctx.singleton<EpisodeState>();

    WorldReset force_reset = ctx.singleton<WorldReset>();
    CheckpointReset ckpt_reset = ctx.singleton<CheckpointReset>();

    bool success =
        episode_state.curLevel == ctx.data().levelsPerEpisode;

    bool episode_finished = false;
    if ((ctx.data().simFlags & SimFlags::IgnoreEpisodeLength) !=
            SimFlags::IgnoreEpisodeLength) {
        if (episode_state.curStep == ctx.data().episodeLen) {
            episode_finished = true;
        }
    }

    // done records only death, success or the step limit. During a checkpoint
    // or world reset, physics runs one discarded step that can look like death
    // or success, so done stays 0 then. A terminal state is never checkpointed
    // (the next world is generated in the same step), so this loses nothing.
    done.v = (episode_state.isDead || success || episode_finished) && 
    !(ckpt_reset.reset == 1) && !(force_reset.reset == 1);

    // Success: the agent reached all waypoints in order, then the exit.
    ctx.singleton<Success>().v = episode_state.reachedExit && episode_state.activeWaypointIdx == 0;
    
    episode_state.reachedWaypoint = false;
}

static TaskGraphNodeID queueRewardSystem(const Sim::Config &cfg,
                                         TaskGraphBuilder &builder,
                                         Span<const TaskGraphNodeID> deps)
{
    TaskGraphNodeID reward_sys;
    switch (cfg.rewardMode) {
    case RewardMode::DenseNoGoal: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
                denseNoGoalRewardSystem,
                Position,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::Dense1: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
                dense1RewardSystem,
                Position,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::Dense1Traj: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
                dense1withTrajRewardSystem,
                Position,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::Dense2: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             dense2RewardSystem,
                Position,
                GrabState,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::PerLevel: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             perLevelRewardSystem,
                Reward
            >>(deps);
    } break;
    case RewardMode::EndOnly: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             endOnlyRewardSystem,
                Reward
            >>(deps);
    } break;
    case RewardMode::LavaButton: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             lavaButtonRewardSystem,
                Position,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::PerGoal: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             perGoalRewardSystem,
                Reward
            >>(deps);
    } break;
    case RewardMode::HandTunedTraj: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             handTunedTrajRewardSystem,
                Reward
            >>(deps);
    } break;
    // Sparse waypoint reward only; PBGE adds its own return reward.
    case RewardMode::PolicyBasedGoExplore: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
                sparseRewardSystem,
                Position,
                Progress,
                Reward
            >>(deps);
    } break;
    default: MADRONA_UNREACHABLE();
    }

    return reward_sys;
}

// Sort an archetype's rows by world. Only the GPU backend needs this; the CPU
// backend keeps a separate table per world.
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraphNodeID queueSortByWorld(TaskGraphBuilder &builder,
                                 Span<const TaskGraphNodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

static TaskGraphNodeID sortEntities(TaskGraphBuilder &builder,
                                    Span<const TaskGraphNodeID> deps)
{
    auto sort_sys = queueSortByWorld<PhysicsEntity>(
        builder, deps);
    sort_sys = queueSortByWorld<ButtonEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<DoorEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<ExitEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<EnemyEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<LavaEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<MotionEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<LavaMotionEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<TrussMotionEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<ImpulseObjectEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<TrussEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<ViewBoxEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<OrientationArrowEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<LidarEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<PatternEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<CoopEntity>(
        builder, {sort_sys});

    return sort_sys;
}
#endif

static TaskGraphNodeID setupSimTasks(TaskGraphBuilder &builder,
                                     const Sim::Config &,
                                     Span<const TaskGraphNodeID> deps)
{
    /**
     * Order of the simulation tasks:
     * 1. Update deltaT.
     * 2. Turn actions into forces (movement, jumps, impulse objects).
     * 3. Clear all contacts.
     * 4. Run the physics system, which updates the contacts.
     * 5. Run the game logic.
     */


    // Sample deltaT in the range set in consts.hpp.
    auto update_sim_deltaT = builder.addToGraph<ParallelForNode<Engine,
        updateDeltaTSystem,
            EpisodeState
        >>(deps);

    // Build BVH for broadphase / raycasting
    auto broadphase_setup_sys = PhysicsSystem::setupBroadphaseTasks(
        builder, {update_sim_deltaT});

    // Trusses: start climbing when the agent touches one.
    auto truss_sys = builder.addToGraph<
        ParallelForNode<Engine, trussSystem, Entity, AgentTouched, IsTruss>
    >({broadphase_setup_sys});

    // Turn the actions into movement forces.
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Entity,
            Action,
            Position,
            Rotation,
            Orientation,
            Scale,
            ClimbState,
            ExternalForce,
            ExternalTorque,
            FrameLinearVelocity
        >>({truss_sys});

    // Jump actions
    auto jump_sys = builder.addToGraph<ParallelForNode<Engine,
        jumpSystem,
            Action,
            Position,
            Rotation,
            Scale,
            ExternalForce,
            JumpPerformed
        >>({move_sys});


    auto block_motion_sys = builder.addToGraph<ParallelForNode<Engine,
        blockMotionSystem,
            Position,
            Velocity,
            AgentTouched,
            ConditionsList,
            MotionState
        >>({jump_sys});

    // Grab action
    auto grab_sys = builder.addToGraph<ParallelForNode<Engine,
        grabSystem,
            Entity,
            Position,
            Rotation,
            Action,
            GrabState
        >>({block_motion_sys});

    auto enemy_act_sys = builder.addToGraph<ParallelForNode<Engine,
        enemyActSystem,
            Position,
            Rotation,
            EnemyState,
            ExternalForce,
            ExternalTorque
        >>({grab_sys});

    // Impulse objects (e.g., trampolines, conveyors)
    auto impulse_object_sys = builder.addToGraph<
        ParallelForNode<Engine, impulseObjectSystem, AgentTouched, ImpulseObjectState>
    >({enemy_act_sys});

    auto view_box_sys = builder.addToGraph<
        ParallelForNode<Engine, viewBoxSystem, Position, IsViewBox>
    >({impulse_object_sys});

    // Scripted door behavior
    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
            Position,
            OpenState
        >>({view_box_sys});

    // Clear all contacts
    auto clear_touched_sys = builder.addToGraph<ParallelForNode<Engine,
        clearTouchedSystem,
            AgentTouched
        >>({set_door_pos_sys});

    // Physics collision detection and solver
    auto substep_sys = PhysicsSystem::setupPhysicsStepTasks(builder,
        {clear_touched_sys}, consts::numPhysicsSubsteps);

    // Zero the agent's angular velocity after physics, for control.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = PhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    auto orientation_arrow_sys = builder.addToGraph<
        ParallelForNode<Engine, orientationArrowSystem, Position, Rotation, IsOrientationArrow>
    >({phys_done});

    // Check if the agent has fallen off the level
    auto agent_fall_sys = builder.addToGraph<ParallelForNode<Engine,
        agentFallSystem, Position, Action>>(
            {orientation_arrow_sys});

    auto dynamic_object_reset_sys = builder.addToGraph<ParallelForNode<Engine,
        dynamicObjectResetSystem, Position, EntityExtents, ResponseType>>(
            {agent_fall_sys});

    auto enemy_post_phys_sys = builder.addToGraph<ParallelForNode<Engine,
        enemyPostPhysicsSystem, Position, Velocity, EnemyState>>(
            {dynamic_object_reset_sys});

    // Check buttons
    auto button_sys = builder.addToGraph<ParallelForNode<Engine,
        buttonSystem,
            Position,
            AgentTouched,
            ButtonState,
            EntityExtents
        >>({enemy_post_phys_sys});

    // Check patterns
    auto pattern_sys = builder.addToGraph<ParallelForNode<Engine,
        patternSystem,
            Position,
            PatternMatchState
        >>({button_sys});

    // Set door to start opening if button conditions are met
    auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
        doorOpenSystem,
            OpenState,
            DoorProperties,
            ButtonListElem,
            PatternListElem,
            ChickenListElem
        >>({pattern_sys});

    auto check_exit_sys = builder.addToGraph<ParallelForNode<Engine,
        checkExitSystem,
            AgentTouched,
            IsExit
        >>({door_open_sys});

    auto check_waypoint_system = builder.addToGraph<ParallelForNode<Engine,
        checkWaypointSystem,
            AgentTouched,
            WaypointProperties
        >>({check_exit_sys});

    auto update_waypoint_system = builder.addToGraph<ParallelForNode<Engine,
        updateWaypointSystem,
            Entity,
            WaypointProperties
        >>({check_waypoint_system});

    auto lava_sys = builder.addToGraph<ParallelForNode<Engine,
        lavaSystem,
            AgentTouched,
            LavaProperties
        >>({update_waypoint_system});

    return lava_sys;
}

static TaskGraphNodeID setupRewardAndDoneTasks(TaskGraphBuilder &builder,
                                               const Sim::Config &cfg,
                                               Span<const TaskGraphNodeID> deps)
{
    // Check if the episode is over
    auto update_episode_state_sys = builder.addToGraph<ParallelForNode<Engine,
        updateEpisodeStateSystem,
            EpisodeState
        >>(deps);

    TaskGraphNodeID reward_sys =
        queueRewardSystem(cfg, builder, {update_episode_state_sys});

    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        writeDoneSystem,
            Done
        >>({reward_sys});

    return done_sys;
}

static TaskGraphNodeID setupResetAndGenTasks(TaskGraphBuilder &builder,
                                             const Sim::Config &cfg,
                                             Span<const TaskGraphNodeID> deps)
{
    (void)cfg;

    // Conditionally reset the world if the episode is over
    auto new_episode_sys = builder.addToGraph<ParallelForNode<Engine,
        newEpisodeSystem,
           EpisodeState
        >>(deps);

    auto cleanup_level = builder.addToGraph<ParallelForNode<Engine,
        cleanupLevelSystem,
            EpisodeState
        >>({new_episode_sys});

    cleanup_level = builder.addToGraph<ParallelForNode<Engine,
        deferredDeleteSystem,
            DeferredDelete
        >>({cleanup_level});

    cleanup_level = builder.addToGraph<
        ClearTmpNode<DeferredDeleteEntity>>({cleanup_level});

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    cleanup_level = builder.addToGraph<RecycleEntitiesNode>({cleanup_level});
#endif


    auto load_checkpoint_json_idx_sys = builder.addToGraph<ParallelForNode<Engine,
        loadCheckpointJsonIdxSystem,
            CheckpointReset
        >>({cleanup_level});

    auto gen_level_sys = builder.addToGraph<ParallelForNode<Engine,
        generateLevelSystem,
            EpisodeState
        >>({load_checkpoint_json_idx_sys});

    auto post_gen = gen_level_sys;
#ifdef MADRONA_GPU_MODE
    post_gen = sortEntities(builder, {post_gen});
#endif
    // When requested, load the checkpoint after the world is reset. With the
    // observations, this restores all the state the training code reads.
    auto load_checkpoint_sys = builder.addToGraph<ParallelForNode<Engine,
        loadCheckpointSystem,
            CheckpointReset
        >>({post_gen});

    // Save a checkpoint unless CheckpointSave is 0.
    auto checkpoint_sys = builder.addToGraph<ParallelForNode<Engine,
        checkpointSystem,
            CheckpointSave
        >>({load_checkpoint_sys});

    // Rebuild the BVH. It is needed only after a reset, but the task graph
    // cannot queue nodes conditionally.
    auto post_reset_broadphase = PhysicsSystem::setupBroadphaseTasks(
        builder, {checkpoint_sys});

    return post_reset_broadphase;
}

static TaskGraphNodeID setupPostGenTasks(TaskGraphBuilder &builder,
                                         const Sim::Config &cfg,
                                         Span<const TaskGraphNodeID> deps)
{

    // Lidar.
#ifdef MADRONA_GPU_MODE
    // CustomParallelForNode<..., 32, 1> launches a warp of 32 threads for each
    // invocation.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            LidarDepth,
            LidarHitType
        >>(deps);


    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, deps);
    }

#ifndef MADRONA_GPU_MODE
    // Already sorted above.
    (void)lidar;
#endif

    // Finally, collect observations for the next step.
    return builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Orientation,
            Velocity,
            Progress,
            Action,
            GrabState,
            AgentTxfmObs,
            AgentInteractObs,
            AgentExitObs,
            StepsRemainingObservation,
            EntityPhysicsStateObsArray,
            EntityTypeObsArray
        >>({lidar});
}

static void setupInitTasks(TaskGraphBuilder &builder, const Sim::Config &cfg)
{
#ifdef MADRONA_GPU_MODE
    auto sort = queueSortByWorld<Agent>(builder, {});
    sort = sortEntities(builder, {sort});
#endif

    auto post_gen = setupResetAndGenTasks(builder, cfg, {
#ifdef MADRONA_GPU_MODE
        sort,
#endif
    });

    setupPostGenTasks(builder, cfg, {post_gen});
}

static void setupStepTasks(TaskGraphBuilder &builder, const Sim::Config &cfg)
{
    auto sim_done = setupSimTasks(builder, cfg, {});
    auto reward_and_dones = setupRewardAndDoneTasks(builder, cfg, {sim_done});
    auto post_gen = setupResetAndGenTasks(builder, cfg, {reward_and_dones});
    setupPostGenTasks(builder, cfg, {post_gen});
}

// Build the init and step task graphs.
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder, cfg);

    TaskGraphBuilder &step_builder = taskgraph_mgr.init(TaskGraphID::Step);
    setupStepTasks(step_builder, cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // The physics system needs an upper bound on the number of entities in
    // the BVH.
    constexpr CountT max_total_entities =
        consts::maxObjectsPerLevel + 10;

    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -consts::gravity * math::up, max_total_entities);

    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    curWorldEpisode = 0;

    episodeLen = cfg.episodeLen;
    levelsPerEpisode = cfg.levelsPerEpisode;
    doorWidth = cfg.doorWidth;
    buttonWidth = cfg.buttonWidth;
    rewardPerDist = cfg.rewardPerDist;
    slackReward = cfg.slackReward;

    simFlags = cfg.simFlags;

    jsonLevels = cfg.jsonLevels;

    agent = createAgent(ctx);


    if (consts::visualizeLidar) {
        for(CountT i=0; i<consts::numLidarSamples / 2; i++) {
            lidarRays[i] = ctx.makeRenderableEntity<LidarEntity>();
            lidarMarks[i] = ctx.makeRenderableEntity<LidarEntity>();
            ctx.get<Scale>(lidarMarks[i]) = Diag3x3{0.1f, 0.1f, 0.1f};
            ctx.get<Rotation>(lidarMarks[i]) = Quat{1,0,0,0};
        }
    }

    Entity orientation_entity = ctx.makeRenderableEntity<OrientationArrowEntity>();
    ctx.get<Position>(orientation_entity) = Vector3::zero();
    ctx.get<Scale>(orientation_entity) =  Diag3x3 { 0.1, consts::agentExtents.y * 0.5f, 0.1 };
    ctx.get<ObjectID>(orientation_entity) = ObjectID { (int32_t)SimObject::Cube };
    ctx.get<Rotation>(orientation_entity) = Quat { 1, 0, 0, 0 };

    resetAgent(ctx, Vector3::zero(), Quat{1,0,0,0}, Vector2{0.0f, 0.0f}, Vector3::zero(), Vector3::zero(), cfg.maxHealth);

    ctx.singleton<Level>() = {
        .exit = ctx.makeRenderableEntity<ExitEntity>(),
        .waypoint = ctx.makeRenderableEntity<PhysicsEntity>()
    };

    ctx.singleton<EpisodeState>() = initEpisodeState();

    simEntityQuery = ctx.query<Entity, EntityType>();
    buttonQuery = ctx.query<Position, ButtonState>();
    conditionQuery = ctx.query<Entity, ButtonState>();
    patternQuery = ctx.query<Entity, PatternMatchState>();
    doorQuery = ctx.query<OpenState>();

    ctx.singleton<CheckpointReset>().reset = 0;
    ctx.singleton<CheckpointSave>().save = 1;

    ctx.singleton<DebugMaxAirRes>().maxAirRes = Vector3(0,0,0);

    ctx.singleton<Success>().v = 0;

    ctx.singleton<JSONIndex>().index = -1;
    // A valid index is needed before the training code writes one on the
    // first step.
    ctx.singleton<CurrentJSONIndex>().index = 0;

    ctx.singleton<ActionLatch>().action = {0,0,0,0};
    ctx.singleton<DynamicBlockPos>().pos = Vector3(0,0,0);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}