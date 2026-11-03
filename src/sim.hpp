#pragma once
#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/rand.hpp>

#include "consts.hpp"
#include "types.hpp"
#include "sim_flags.hpp"

namespace madPuzzle {

class Engine;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Reset,
    Goal,
    Action,
    Reward,
    Done,
    Health,
    AgentPos,
    AgentTxfmObs,
    AgentInteractObs,
    AgentLevelTypeObs,
    AgentJumpPerformed,
    AgentExitObs,
    EntityPhysicsStateObsArray,
    EntityTypeObsArray,
    LidarDepth,
    LidarHitType,
    StepsRemainingObs,
    Checkpoint,
    CheckpointReset,
    CheckpointSave,
    AgentPolicy,
    EpisodeResult,
    JsonIndex,
    TimeStep,
    TrajectorySection,
    DynamicBlockPos,
    StepsRemaining,
    Success,
    NumExports,
};

// Stores values for the ObjectID component that links entities to
// render / physics assets.
enum class SimObject : uint32_t {
    Agent,
    Cube,
    RedCube,
    GreenCube,
    BlueCube,
    PinkCube,
    PurpleCube,
    YellowCube,
    CyanCube,
    OrangeCube,
    Sphere,
    RedSphere,
    GreenSphere,
    BlueSphere,
    Wedge,
    RedWedge,
    GreenWedge,
    BlueWedge,
    Cylinder,
    RedCylinder,
    Plane,
    NumObjects,
};

enum class TaskGraphID : uint32_t {
    Init,
    Step,
    NumGraphs,
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        bool autoReset;
        SimFlags simFlags;
        RewardMode rewardMode;
        RandKey initRandKey;
        uint32_t episodeLen;
        uint32_t levelsPerEpisode;
        float doorWidth;
        float buttonWidth;
        float rewardPerDist;
        float slackReward;
        float maxHealth;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        RewardHyperParams *rewardHyperParams;
        const madrona::render::RenderECSBridge *renderBridge;
        JSONLevel *jsonLevels;
        int32_t *enumCounts;
    };

    // Per-world initialization data. This environment needs none.
    struct WorldInit {};

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    // Called once per world during initialization. Config is shared by all
    // worlds; WorldInit holds per-world data (created in src/mgr.cpp).
    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &);

    // Base random key; each episode's key is split off from it.
    madrona::RandKey initRandKey;

    // Start a new episode automatically when one ends.
    bool autoReset;

    uint32_t episodeLen;
    uint32_t levelsPerEpisode;
    float doorWidth;
    float buttonWidth;
    float rewardPerDist;
    float slackReward;
    SimFlags simFlags;
    JSONLevel *jsonLevels;

    

    // True when rendering is on (with or without the viewer).
    bool enableRender;

    // Current episode within this world.
    uint32_t curWorldEpisode;
    // The random seed that generated this world.
    RandKey curEpisodeRNDCounter;
    // Random number generator state.
    madrona::RNG rng;

    // Agent entities. They live across all episodes; a reset moves them back
    // to the start of the level.
    Entity agent;
    Entity viewBox;


    Entity lidarRays[consts::numLidarSamples];
    Entity lidarMarks[consts::numLidarSamples];

    // Queries for the collectObservations system.
    Query<Entity, EntityType> simEntityQuery;
    Query<Position, ButtonState> buttonQuery;
    Query<Position, EntityExtents, WaypointProperties> waypointQuery;
    Query<Entity, ButtonState> conditionQuery;
    Query<OpenState> doorQuery;
    Query<Entity, PatternMatchState> patternQuery;
    Query<Position, CoopState> coopQuery;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // Helpers that create renderable entities whether or not rendering is on.
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"