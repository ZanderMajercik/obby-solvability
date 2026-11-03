#include "types.hpp"
#include "sim.hpp"

namespace madPuzzle {

using namespace madrona;

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);
    PhysicsSystem::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerComponent<Action>();

    registry.registerComponent<AgentTxfmObs>();
    registry.registerComponent<AgentInteractObs>();
    registry.registerComponent<AgentLevelTypeObs>();
    registry.registerComponent<AgentExitObs>();

    registry.registerComponent<EntityPhysicsStateObsArray>();
    registry.registerComponent<EntityTypeObsArray>();
    registry.registerComponent<LidarDepth>();
    registry.registerComponent<LidarHitType>();

    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<AgentPolicy>();

    registry.registerComponent<Orientation>();

    registry.registerComponent<GrabState>();
    registry.registerComponent<AirResForce>();
    registry.registerComponent<ClimbState>();
    registry.registerComponent<Progress>();
    registry.registerComponent<Health>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<PatternMatchState>();
    registry.registerComponent<CoopState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorProperties>();
    registry.registerComponent<DoorRooms>();
    registry.registerComponent<StepsRemainingObservation>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<EntityExtents>();

    registry.registerComponent<DeferredDelete>();

    registry.registerComponent<ButtonListElem>();
    registry.registerComponent<PatternListElem>();
    registry.registerComponent<ChickenListElem>();

    registry.registerComponent<IsExit>();
    registry.registerComponent<LavaProperties>();
    registry.registerComponent<WaypointProperties>();
    registry.registerComponent<IsOrientationArrow>();
    registry.registerComponent<IsLidar>();
    registry.registerComponent<EnemyState>();
    registry.registerComponent<ImpulseObjectState>();
    registry.registerComponent<MotionState>();
    registry.registerComponent<IsTruss>();
    registry.registerComponent<IsViewBox>();
    registry.registerComponent<ConditionsList>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<Level>();
    registry.registerSingleton<LevelAABB>();
    registry.registerSingleton<EpisodeState>();
    registry.registerSingleton<EpisodeResult>();
    registry.registerSingleton<ActionLatch>();
    registry.registerSingleton<JumpPerformed>();
    registry.registerSingleton<DynamicBlockPos>();
    registry.registerSingleton<StepsRemaining>();
    registry.registerSingleton<TimeStep>();
    registry.registerSingleton<DebugMaxAirRes>();
    registry.registerSingleton<Success>();
    
    // Index of the JSON level description to load.
    registry.registerSingleton<JSONIndex>();
    registry.registerSingleton<CurrentJSONIndex>();

    // Checkpoint state.
    registry.registerSingleton<Checkpoint>();
    registry.registerSingleton<CheckpointReset>();
    registry.registerSingleton<CheckpointSave>();

    // Trajectory state.
    registry.registerSingleton<WorldTrajectory>();
    registry.registerSingleton<TrajectorySection>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();
    registry.registerArchetype<ExitEntity>();
    registry.registerArchetype<EnemyEntity>();
    registry.registerArchetype<LavaEntity>();
    registry.registerArchetype<MotionEntity>();
    registry.registerArchetype<LavaMotionEntity>();
    registry.registerArchetype<TrussMotionEntity>();
    registry.registerArchetype<PatternEntity>();
    registry.registerArchetype<CoopEntity>();
    registry.registerArchetype<ImpulseObjectEntity>();
    registry.registerArchetype<TrussEntity>();
    registry.registerArchetype<ViewBoxEntity>();
    registry.registerArchetype<OrientationArrowEntity>();
    registry.registerArchetype<LidarEntity>();

    registry.registerArchetype<DeferredDeleteEntity>();

    registry.exportSingleton<TrajectorySection>(
        (uint32_t)ExportID::TrajectorySection);
    registry.exportSingleton<Checkpoint>(
        (uint32_t)ExportID::Checkpoint);
    registry.exportSingleton<CheckpointReset>(
        (uint32_t)ExportID::CheckpointReset);
    registry.exportSingleton<CheckpointSave>(
        (uint32_t)ExportID::CheckpointSave);
    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportSingleton<TimeStep>(
        (uint32_t)ExportID::TimeStep);
    registry.exportSingleton<JSONIndex>(
        (uint32_t)ExportID::JsonIndex);
    registry.exportSingleton<EpisodeResult>(
        (uint32_t)ExportID::EpisodeResult);
    registry.exportSingleton<DynamicBlockPos>(
        (uint32_t)ExportID::DynamicBlockPos);
    registry.exportSingleton<StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportSingleton<Success>(
        (uint32_t)ExportID::Success);

    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);

    registry.exportColumn<Agent, Position>(
        (uint32_t)ExportID::AgentPos);
    registry.exportColumn<Agent, AgentTxfmObs>(
        (uint32_t)ExportID::AgentTxfmObs);
    registry.exportColumn<Agent, AgentInteractObs>(
        (uint32_t)ExportID::AgentInteractObs);
    registry.exportColumn<Agent, AgentLevelTypeObs>(
        (uint32_t)ExportID::AgentLevelTypeObs);
    registry.exportColumn<Agent, AgentExitObs>(
        (uint32_t)ExportID::AgentExitObs);
    registry.exportColumn<Agent, StepsRemainingObservation>(
        (uint32_t)ExportID::StepsRemainingObs);
    registry.exportColumn<Agent, JumpPerformed>(
        (uint32_t)ExportID::AgentJumpPerformed);

    registry.exportColumn<Agent, EntityPhysicsStateObsArray>(
        (uint32_t)ExportID::EntityPhysicsStateObsArray);
    registry.exportColumn<Agent, EntityTypeObsArray>(
        (uint32_t)ExportID::EntityTypeObsArray);

    registry.exportColumn<Agent, LidarDepth>(
        (uint32_t)ExportID::LidarDepth);
    registry.exportColumn<Agent, LidarHitType>(
        (uint32_t)ExportID::LidarHitType);

    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);

    registry.exportColumn<Agent, AgentPolicy>(
        (uint32_t)ExportID::AgentPolicy);
}

}