#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madPuzzle {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

#ifdef MADRONA_CUDA_SUPPORT
static inline uint64_t numTensorBytes(const Tensor &t)
{
    uint64_t num_items = 1;
    uint64_t num_dims = t.numDims();
    for (uint64_t i = 0; i < num_dims; i++) {
        num_items *= t.dims()[i];
    }

    return num_items * (uint64_t)t.numBytesPerItem();
}
#endif

static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = 1,
        .maxInstancesPerWorld = 1000,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    JSONLevel *jsonLevelsBuffer;
    int32_t* enumCountsBuffer;
    PhysicsLoader physicsLoader;
    WorldReset *worldResetBuffer;
    JSONIndex *jsonIndexBuffer;
    CheckpointSave *worldSaveCheckpointBuffer;
    CheckpointReset *worldLoadCheckpointBuffer;
    Action *agentActionsBuffer;
    RewardHyperParams *rewardHyperParams;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                WorldReset *reset_buffer,
                JSONLevel *json_level_buffer,
                int32_t *enum_counts_buffer,
                JSONIndex *json_index_buffer,
                CheckpointSave *checkpoint_save_buffer,
                CheckpointReset *checkpoint_load_buffer,
                Action *action_buffer,
                RewardHyperParams *reward_hyper_params,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          jsonLevelsBuffer(json_level_buffer),
          enumCountsBuffer(enum_counts_buffer),
          physicsLoader(std::move(phys_loader)),
          worldResetBuffer(reset_buffer),
          jsonIndexBuffer(json_index_buffer),
          worldSaveCheckpointBuffer(checkpoint_save_buffer),
          worldLoadCheckpointBuffer(checkpoint_load_buffer),
          agentActionsBuffer(action_buffer),
          rewardHyperParams(reward_hyper_params),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void init() = 0;
    virtual void step() = 0;

    inline void renderStep()
    {
        if (renderMgr.has_value()) {
            renderMgr->readECS();
        }

        if (cfg.enableBatchRenderer) {
            renderMgr->batchRender();
        }
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &) = 0;
    virtual void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &) = 0;

    virtual void gpuStreamLoadCheckpoints(
        cudaStream_t strm, void **buffers, Manager &) = 0;
    virtual void gpuStreamGetCheckpoints(
        cudaStream_t strm, void **buffers, Manager &) = 0;
#endif

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    virtual Tensor rewardHyperParamsTensor() const = 0;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   JSONLevel *json_level_buffer,
                   int32_t *enum_count_buffer,
                   JSONIndex *json_index_buffer,
                   CheckpointSave *checkpoint_save_buffer,
                   CheckpointReset *checkpoint_load_buffer,
                   Action *action_buffer,
                   RewardHyperParams *reward_hyper_params,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, json_level_buffer, enum_count_buffer, json_index_buffer,
               checkpoint_save_buffer, checkpoint_load_buffer, action_buffer,
               reward_hyper_params,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final 
    {
        free(rewardHyperParams);
    }

    inline virtual void init() final
    {
        cpuExec.runTaskGraph(TaskGraphID::Init);
        renderStep();
    }

    inline virtual void step() final
    {
        cpuExec.runTaskGraph(TaskGraphID::Step);
        renderStep();
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t, void **, Manager &)
    {
        assert(false);
    }

    virtual void gpuStreamStep(cudaStream_t, void **, Manager &)
    {
        assert(false);
    }

    virtual void gpuStreamLoadCheckpoints(
        cudaStream_t, void **, Manager &)
    {
        assert(false);
    }

    virtual void gpuStreamGetCheckpoints(
        cudaStream_t, void **, Manager &)
    {
        assert(false);
    }
#endif

    virtual Tensor rewardHyperParamsTensor() const final
    {
        return Tensor(rewardHyperParams, TensorElementType::Float32,
            {
                cfg.numPBTPolicies,
                sizeof(RewardHyperParams) / sizeof(float),
            }, Optional<int>::none());
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph initGraph;
    MWCudaLaunchGraph stepGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   JSONLevel *json_level_buffer,
                   int32_t *enum_count_buffer,
                   JSONIndex *json_index_buffer,
                   CheckpointSave *checkpoint_save_buffer,
                   CheckpointReset *checkpoint_load_buffer,
                   Action *action_buffer,
                   RewardHyperParams *reward_hyper_params,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, json_level_buffer, enum_count_buffer, json_index_buffer, checkpoint_save_buffer,
               checkpoint_load_buffer, action_buffer, reward_hyper_params,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          initGraph(gpuExec.buildLaunchGraph(TaskGraphID::Init)),
          stepGraph(gpuExec.buildLaunchGraph(TaskGraphID::Step))
    {}

    inline virtual ~CUDAImpl() final
    {
        REQ_CUDA(cudaFree(rewardHyperParams));
    }

    inline virtual void init() final
    {
        gpuExec.run(initGraph);
        renderStep();
    }

    inline virtual void step() final
    {
        gpuExec.run(stepGraph);
        renderStep();
    }

    inline void copyFromSim(cudaStream_t strm, void *dst, const Tensor &src)
    {
        uint64_t num_bytes = numTensorBytes(src);

        REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
            cudaMemcpyDeviceToDevice, strm));
    }

    inline void copyToSim(cudaStream_t strm, const Tensor &dst, void *src)
    {
        uint64_t num_bytes = numTensorBytes(dst);

        REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
            cudaMemcpyDeviceToDevice, strm));
    }

#ifdef MADRONA_CUDA_SUPPORT
    inline void ** copyOutObservations(cudaStream_t strm,
                                       void **buffers,
                                       Manager &mgr)
    {
        // Observations
        copyFromSim(strm, *buffers++, mgr.agentTxfmObsTensor());
        copyFromSim(strm, *buffers++, mgr.agentInteractObsTensor());
        copyFromSim(strm, *buffers++, mgr.agentLevelTypeObsTensor());
        copyFromSim(strm, *buffers++, mgr.agentExitObsTensor());
        copyFromSim(strm, *buffers++, mgr.lidarDepthTensor());
        copyFromSim(strm, *buffers++, mgr.lidarHitTypeTensor());
        copyFromSim(strm, *buffers++, mgr.stepsRemainingObsTensor());
        copyFromSim(strm, *buffers++, mgr.entityPhysicsStateObsTensor());
        copyFromSim(strm, *buffers++, mgr.entityTypeObsTensor());

        return buffers;
    }

    virtual void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &mgr)
    {
        HeapArray<WorldReset> resets_staging(cfg.numWorlds);
        for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
            resets_staging[i].reset = 1;
        }

        cudaMemcpyAsync(worldResetBuffer, resets_staging.data(),
                   sizeof(WorldReset) * cfg.numWorlds,
                   cudaMemcpyHostToDevice, strm);
        gpuExec.runAsync(initGraph, strm);
        copyOutObservations(strm, buffers, mgr);

        REQ_CUDA(cudaStreamSynchronize(strm));
    }

    virtual void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &mgr)
    {
        copyToSim(strm, mgr.actionTensor(), *buffers++);
        copyToSim(strm, mgr.resetTensor(), *buffers++);

        if (cfg.numPBTPolicies > 0) {
            copyToSim(strm, mgr.policyAssignmentsTensor(), *buffers++);
            copyToSim(strm, mgr.rewardHyperParamsTensor(), *buffers++);
        }

        gpuExec.runAsync(stepGraph, strm);

        buffers = copyOutObservations(strm, buffers, mgr);

        copyFromSim(strm, *buffers++, mgr.rewardTensor());
        copyFromSim(strm, *buffers++, mgr.doneTensor());
        copyFromSim(strm, *buffers++, mgr.episodeResultTensor());
    }

    virtual void gpuStreamLoadCheckpoints(
        cudaStream_t strm, void **buffers, Manager &mgr)
    {
        copyToSim(strm, mgr.checkpointResetTensor(), *buffers++);
        copyToSim(strm, mgr.checkpointTensor(), *buffers++);


        gpuExec.runAsync(stepGraph, strm);

        copyOutObservations(strm, buffers, mgr);
    }

    virtual void gpuStreamGetCheckpoints(
        cudaStream_t strm, void **buffers, Manager &mgr)
    {
        copyFromSim(strm, *buffers, mgr.checkpointTensor());
    }
#endif

    virtual Tensor rewardHyperParamsTensor() const final
    {
        return Tensor(rewardHyperParams, TensorElementType::Float32,
            {
                cfg.numPBTPolicies,
                sizeof(RewardHyperParams) / sizeof(float),
            }, cfg.gpuID);
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static void loadRenderObjects(render::RenderManager &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    
    // The agent has its own mesh so that its face shows its heading.
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();

    // Cube colors
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::RedCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::GreenCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::BlueCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::PinkCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::PurpleCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::YellowCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::CyanCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::OrangeCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_render.obj").string();

    // Sphere colors
    render_asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_render.obj").string();
    render_asset_paths[(size_t)SimObject::RedSphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_render.obj").string();
    render_asset_paths[(size_t)SimObject::GreenSphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_render.obj").string();
    render_asset_paths[(size_t)SimObject::BlueSphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_render.obj").string();

    // Wedge colors
    render_asset_paths[(size_t)SimObject::Wedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_render.obj").string();
    render_asset_paths[(size_t)SimObject::RedWedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_render.obj").string();
    render_asset_paths[(size_t)SimObject::GreenWedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_render.obj").string();
    render_asset_paths[(size_t)SimObject::BlueWedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_render.obj").string();

    render_asset_paths[(size_t)SimObject::Cylinder] =
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").string();
    render_asset_paths[(size_t)SimObject::RedCylinder] =
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").string();

    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f }, // Orange
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,}, // Gray, default
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,}, // Agent texture
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f }, // White
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,}, // Plane texture
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f }, // Red
        { render::rgb8ToFloat(20, 230, 20),   -1, 0.8f, 1.0f }, // Green
        { render::rgb8ToFloat(20, 20, 230),   -1, 0.8f, 1.0f }, // Blue
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f }, // Yellow
        { render::rgb8ToFloat(230, 20, 230),   -1, 0.8f, 1.0f }, // Purple
        { render::rgb8ToFloat(20, 230, 230),   -1, 0.8f, 1.0f }, // Cyan
        { render::rgb8ToFloat(230, 20, 20),   2, 0.8f, 1.0f }, // Enemy
    });

    // Material of each object.
    render_assets->objects[(CountT)SimObject::Agent].meshes[0].materialIDX = 2;

    // Gray
    render_assets->objects[(CountT)SimObject::Cube].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObject::Sphere].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObject::Wedge].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObject::Cylinder].meshes[0].materialIDX = 1;

    // Red
    render_assets->objects[(CountT)SimObject::RedCube].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObject::RedSphere].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObject::RedWedge].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObject::RedCylinder].meshes[0].materialIDX = 5;

    // Green
    render_assets->objects[(CountT)SimObject::GreenCube].meshes[0].materialIDX = 6;
    render_assets->objects[(CountT)SimObject::GreenSphere].meshes[0].materialIDX = 6;
    render_assets->objects[(CountT)SimObject::GreenWedge].meshes[0].materialIDX = 6;
    // Blue
    render_assets->objects[(CountT)SimObject::BlueCube].meshes[0].materialIDX = 7;
    render_assets->objects[(CountT)SimObject::BlueSphere].meshes[0].materialIDX = 7;
    render_assets->objects[(CountT)SimObject::BlueWedge].meshes[0].materialIDX = 7;

    // Misc
    render_assets->objects[(CountT)SimObject::PinkCube].meshes[0].materialIDX = 9;
    render_assets->objects[(CountT)SimObject::PurpleCube].meshes[0].materialIDX = 9;
    render_assets->objects[(CountT)SimObject::YellowCube].meshes[0].materialIDX = 8;
    render_assets->objects[(CountT)SimObject::CyanCube].meshes[0].materialIDX = 10;
    render_assets->objects[(CountT)SimObject::OrangeCube].meshes[0].materialIDX = 0;

    // Plane
    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 4;

    render_mgr.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smileRot.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smileRot.png").string().c_str() },
    });

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    std::array<std::string, (size_t)SimObject::NumObjects - 1> asset_paths;


    // The agent's collision shape is a unit cube.
    asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();

    // Cube colors
    asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::RedCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::GreenCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::BlueCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::PinkCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::PurpleCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::YellowCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::CyanCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();
    asset_paths[(size_t)SimObject::OrangeCube] =
        (std::filesystem::path(DATA_DIR) / "unit_cube_collision.obj").string();

    // Sphere colors
    asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_collision.obj").string();
    asset_paths[(size_t)SimObject::RedSphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_collision.obj").string();
    asset_paths[(size_t)SimObject::GreenSphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_collision.obj").string();
    asset_paths[(size_t)SimObject::BlueSphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_collision.obj").string();

    // Wedge colors
    asset_paths[(size_t)SimObject::Wedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_collision.obj").string();
    asset_paths[(size_t)SimObject::RedWedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_collision.obj").string();
    asset_paths[(size_t)SimObject::GreenWedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_collision.obj").string();
    asset_paths[(size_t)SimObject::BlueWedge] =
        (std::filesystem::path(DATA_DIR) / "unit_wedge_collision.obj").string();
    
    // Cylinder
    asset_paths[(size_t)SimObject::Cylinder] =
        (std::filesystem::path(DATA_DIR) / "cylinder_collision.obj").string();
    asset_paths[(size_t)SimObject::RedCylinder] =
        (std::filesystem::path(DATA_DIR) / "cylinder_collision.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects - 1> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    char import_err_buffer[4096];
    auto imported_src_hulls = imp::ImportedAssets::importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObject::NumObjects);

    auto setupHull = [&](SimObject obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };


    setupHull(SimObject::Agent, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Cube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::RedCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::GreenCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::BlueCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::PinkCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::PurpleCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::YellowCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::CyanCube, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    // Mass and friction tuned for an unanchored, freely simulated cube.
    setupHull(SimObject::OrangeCube, 0.075f, {
        .muS = 1.0f,
        .muD = 1.5f,
    });

    setupHull(SimObject::Sphere, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::RedSphere, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::GreenSphere, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::BlueSphere, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Wedge, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::RedWedge, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::GreenWedge, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::BlueWedge, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::Cylinder, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::RedCylinder, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObject::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // Infinite inertia: the agent cannot tip over, so the policy keeps control.
    // The z inertia is also infinite; without it the agent glides.
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.y = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.z = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.simFlags = mgr_cfg.simFlags;
    sim_cfg.rewardMode = mgr_cfg.rewardMode;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);
    sim_cfg.episodeLen = mgr_cfg.episodeLen;
    sim_cfg.levelsPerEpisode = mgr_cfg.levelsPerEpisode;
    sim_cfg.buttonWidth = mgr_cfg.buttonWidth;
    sim_cfg.doorWidth = mgr_cfg.doorWidth;
    sim_cfg.rewardPerDist = mgr_cfg.rewardPerDist;
    sim_cfg.slackReward = mgr_cfg.slackReward;
    sim_cfg.maxHealth = mgr_cfg.maxHealth;

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        PhysicsLoader phys_loader(ExecMode::CUDA,
                                  (uint32_t)SimObject::NumObjects);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        if (mgr_cfg.numPBTPolicies > 0) {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)cu::allocGPU(
                sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
        } else {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)cu::allocGPU(
                sizeof(RewardHyperParams));

            RewardHyperParams default_reward_hyper_params {
                .distToExitScale = mgr_cfg.rewardPerDist,
            };

            REQ_CUDA(cudaMemcpy(sim_cfg.rewardHyperParams,
                &default_reward_hyper_params, sizeof(RewardHyperParams),
                cudaMemcpyHostToDevice));
        }

        sim_cfg.jsonLevels = (JSONLevel *)cu::allocGPU(
            sizeof(JSONLevel) * consts::maxJsonLevelDescriptions
        );

        // Storage for the level-type and entity-type counts.
        sim_cfg.enumCounts = (int32_t *)cu::allocGPU(
            sizeof(int32_t) * 2
        );

        int32_t numLabels = consts::useBasicLabels ?
            (int32_t) EntitySemantics::NumTypes : (int32_t) EntityType::NumTypes;
        int32_t enum_counts[2] = {(int32_t)LevelType::NumTypes, numLabels};

        REQ_CUDA(cudaMemcpy(sim_cfg.enumCounts,
            &enum_counts[0], sizeof(int32_t) * 2,
            cudaMemcpyHostToDevice));

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        CheckpointSave *checkpoint_save_buffer = 
            (CheckpointSave *)gpu_exec.getExported((uint32_t)ExportID::CheckpointSave);

        CheckpointReset *checkpoint_load_buffer = 
            (CheckpointReset *)gpu_exec.getExported((uint32_t)ExportID::CheckpointReset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        JSONIndex *json_index_buffer =
            (JSONIndex *)gpu_exec.getExported((uint32_t)ExportID::JsonIndex);

        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            sim_cfg.jsonLevels,
            sim_cfg.enumCounts,
            json_index_buffer,
            checkpoint_save_buffer,
            checkpoint_load_buffer,
            agent_actions_buffer,
            sim_cfg.rewardHyperParams,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        PhysicsLoader phys_loader(ExecMode::CPU,
                                  (uint32_t)SimObject::NumObjects);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        if (mgr_cfg.numPBTPolicies > 0) {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)malloc(
                sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
        } else {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)malloc(
                sizeof(RewardHyperParams));

            *(sim_cfg.rewardHyperParams) = {
                .distToExitScale = mgr_cfg.rewardPerDist,
            };
        }

        sim_cfg.jsonLevels = (JSONLevel *)malloc(
            sizeof(JSONLevel) * consts::maxJsonLevelDescriptions
        );

        sim_cfg.enumCounts = (int32_t *)malloc(
            sizeof(int32_t) * 2
        );

        // Number of level types and entity types, exported to the training code.
        sim_cfg.enumCounts[0] = (int32_t)LevelType::NumTypes;
        int32_t numLabels = consts::useBasicLabels ?
            (int32_t) EntitySemantics::NumTypes : (int32_t) EntityType::NumTypes;

        sim_cfg.enumCounts[1] = numLabels;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
            (CountT)TaskGraphID::NumGraphs,
        };

        WorldReset *world_reset_buffer = 
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        CheckpointSave *checkpoint_save_buffer = 
            (CheckpointSave *)cpu_exec.getExported((uint32_t)ExportID::CheckpointSave);

        CheckpointReset *checkpoint_load_buffer = 
            (CheckpointReset *)cpu_exec.getExported((uint32_t)ExportID::CheckpointReset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        JSONIndex *json_index_buffer = 
            (JSONIndex *)cpu_exec.getExported((uint32_t)ExportID::JsonIndex);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            sim_cfg.jsonLevels,
            sim_cfg.enumCounts,
            json_index_buffer,
            checkpoint_save_buffer,
            checkpoint_load_buffer,
            agent_actions_buffer,
            sim_cfg.rewardHyperParams,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(cpu_exec),
        };

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

Manager::~Manager() {}

void Manager::init()
{
    const CountT num_worlds = impl_->cfg.numWorlds;

    // Reset and step once so that the observations of the first episode exist.
    for (CountT i = 0; i < num_worlds; i++) {
        triggerReset(i);
    }

    impl_->init();
}

void Manager::step()
{
    impl_->step();
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamInit(strm, buffers, *this);

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

void Manager::gpuStreamStep(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamStep(strm, buffers, *this);

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}


void Manager::gpuStreamLoadCheckpoints(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamLoadCheckpoints(strm, buffers, *this);

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

void Manager::gpuStreamGetCheckpoints(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamGetCheckpoints(strm, buffers, *this);

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}
#endif

Tensor Manager::checkpointResetTensor() const {
    return impl_->exportTensor(ExportID::CheckpointReset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(CheckpointReset) / sizeof(int32_t)
                               });
}

Tensor Manager::checkpointTensor() const {
    return impl_->exportTensor(ExportID::Checkpoint,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Checkpoint)
                               });
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(WorldReset) / sizeof(int32_t)
                               });
}

Tensor Manager::timestepTensor() const
{
    return impl_->exportTensor(ExportID::TimeStep,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(TimeStep) / sizeof(float)
                               });
}

Tensor Manager::jsonIndexTensor() const
{
    return impl_->exportTensor(ExportID::JsonIndex,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(JSONIndex) / sizeof(int32_t)
                               });
}

Tensor Manager::jsonLevelDescriptionsTensor() const {
    // The JSON level descriptions. The training code writes them once; they are
    // not part of the ECS and are read only when a level loads.
    return Tensor(impl_->jsonLevelsBuffer,
                  TensorElementType::Float32,
                  {consts::maxJsonLevelDescriptions,
                  sizeof(JSONLevel) / sizeof(JSONObject),
                   sizeof(JSONObject) / sizeof(float)},
                  Optional<int>::none());

}

Tensor Manager::enumCountsTensor() const {
    return Tensor(impl_->enumCountsBuffer,
                  TensorElementType::Int32,
                  {(CountT)2},
                  Optional<int>::none());
}

Tensor Manager::trajectorySectionTensor() const
{
    return impl_->exportTensor(ExportID::TrajectorySection,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(TrajectorySection) / sizeof(float)
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            sizeof(Action) / sizeof(int32_t),
        });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Reward) / sizeof(float),
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Done) / sizeof(int32_t),
                               });
}

Tensor Manager::agentTxfmObsTensor() const
{
    return impl_->exportTensor(ExportID::AgentTxfmObs,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(AgentTxfmObs) / sizeof(float),
                               });
}

Tensor Manager::agentInteractObsTensor() const
{
    return impl_->exportTensor(ExportID::AgentInteractObs,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(AgentInteractObs) / sizeof(int32_t),
                               });
}

Tensor Manager::agentLevelTypeObsTensor() const
{
    return impl_->exportTensor(ExportID::AgentLevelTypeObs,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(AgentLevelTypeObs) / sizeof(int32_t),
                               });
}

Tensor Manager::agentJumpPerformedTensor() const
{
    return impl_->exportTensor(ExportID::AgentJumpPerformed,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(JumpPerformed) / sizeof(int32_t),
                               });
}

Tensor Manager::dynamicBlockPosTensor() const
{
    return impl_->exportTensor(ExportID::DynamicBlockPos,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(DynamicBlockPos) / sizeof(float),
                               });
}

Tensor Manager::agentPosTensor() const
{
    return impl_->exportTensor(ExportID::AgentPos,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Position) / sizeof(float),
                               });
}

Tensor Manager::agentExitObsTensor() const
{
    return impl_->exportTensor(ExportID::AgentExitObs,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(AgentExitObs) / sizeof(float),
                               });
}

Tensor Manager::entityPhysicsStateObsTensor() const
{
    return impl_->exportTensor(ExportID::EntityPhysicsStateObsArray,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::maxObservationsPerAgent,
                                   sizeof(EntityPhysicsStateObs) / sizeof(float),
                               });
}

Tensor Manager::entityTypeObsTensor() const
{
    return impl_->exportTensor(ExportID::EntityTypeObsArray,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::maxObservationsPerAgent,
                                   sizeof(EntityTypeObs) / sizeof(float),
                               });
}

Tensor Manager::lidarDepthTensor() const
{
    return impl_->exportTensor(ExportID::LidarDepth,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numLidarSamples,
                                   1,
                               });
}

Tensor Manager::lidarHitTypeTensor() const
{
    return impl_->exportTensor(ExportID::LidarHitType,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numLidarSamples,
                                   1,
                               });
}

Tensor Manager::stepsRemainingObsTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemainingObs,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(StepsRemainingObservation) /
                                       sizeof(int32_t),
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1
                               });
}

Tensor Manager::successTensor() const
{
    return impl_->exportTensor(ExportID::Success,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Success) / sizeof(int32_t)
                               });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setJsonIndex(int32_t world_idx,
                           int32_t index)
                           {
    JSONIndex jsonIndex {
        index,
    };

    auto *json_index_ptr = impl_->jsonIndexBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(json_index_ptr, &jsonIndex, sizeof(JSONIndex),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *json_index_ptr = jsonIndex;
    }
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate,
                        int32_t interact)
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        .interact = interact
    };

    const CountT num_agents = 1;

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * num_agents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

void Manager::setSaveCheckpoint(int32_t world_idx, int32_t value) 
{
    CheckpointSave save {
        value,
    };

    auto *save_ptr = impl_->worldSaveCheckpointBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(save_ptr, &save, sizeof(CheckpointSave),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *save_ptr = save;
    }
}

void Manager::triggerLoadCheckpoint(int32_t world_idx) 
{
    CheckpointReset load {
        1,
    };

    auto *load_ptr = impl_->worldLoadCheckpointBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(load_ptr, &load, sizeof(CheckpointReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *load_ptr = load;
    }
}
render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}


Tensor Manager::episodeResultTensor() const
{
    return impl_->exportTensor(ExportID::EpisodeResult,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(EpisodeResult) / sizeof(float),
                               });
}

Tensor Manager::policyAssignmentsTensor() const
{
    return impl_->exportTensor(ExportID::AgentPolicy,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(AgentPolicy) / sizeof(int32_t)
                               });
}

Tensor Manager::rewardHyperParamsTensor() const
{
    return impl_->rewardHyperParamsTensor();
}

TrainInterface Manager::trainInterface() const
{
    auto pbt_inputs = std::to_array<NamedTensorInterface>({
        { "policy_assignments", policyAssignmentsTensor().interface() },
        { "reward_hyper_params", rewardHyperParamsTensor().interface() },
    });

    return TrainInterface {
        {
            .actions = actionTensor().interface(),
            .resets = resetTensor().interface(),
            .pbt = impl_->cfg.numPBTPolicies > 0 ?
                pbt_inputs : Span<const NamedTensorInterface>(nullptr, 0),
        },
        {
            .observations = {
                { "agent_txfm", agentTxfmObsTensor().interface() },
                { "agent_interact", agentInteractObsTensor().interface() },
                { "agent_level_type", agentLevelTypeObsTensor().interface() },
                { "agent_exit", agentExitObsTensor().interface() },
                { "lidar_depth", lidarDepthTensor().interface() },
                { "lidar_hit_type", lidarHitTypeTensor().interface() },
                { "steps_remaining", stepsRemainingObsTensor().interface() },
                { "entity_physics_states", entityPhysicsStateObsTensor().interface() },
                { "entity_types", entityTypeObsTensor().interface() },
            },
            .rewards = rewardTensor().interface(),
            .dones = doneTensor().interface(),
            .pbt = {
                { "episode_results", episodeResultTensor().interface() },
            },
        },
        TrainCheckpointingInterface {
            .triggerLoad = checkpointResetTensor().interface(),
            .checkpointData =  checkpointTensor().interface(),
        },
    };
}

}