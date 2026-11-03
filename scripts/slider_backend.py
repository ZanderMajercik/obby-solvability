"""
HTTP backend of the Roblox Studio design tool (the robloxTestHarness plugin).

The plugin loads levels and saved queries, edits obstacle parameters, and asks
the backend to search a level for a solution (Go-Explore or PBGE), to measure
success rates of a policy, or to finetune a policy. Saved
levels, queries and player recordings are JSON files in the folders given by
--obby-save-load-folder, --query-save-load-folder and
--player-trajectory-save-load-folder.
"""
import argparse
import json
import threading
import time

import matplotlib.pyplot as plt

import os

import random
import numpy as np
import torch
import wandb
from flask import Flask, jsonify, request
from madrona_puzzle_bench import SimManager
from madrona_puzzle_bench_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface, LearningState, EMANormalizer
)
from madrona_scripts.infer import setup_sim, run_infer, ACTION_DIMS
from madrona_scripts.simplePPO import PPOTrainingWrapper
from obby.examples.distribution_generators import Constant
from obby.grammar.obby import Obby
from obby.grammar.part import Part
from obby.grammar.behaviors import *
from madrona_scripts.policy import make_policy, setup_obs
from madrona_scripts.go_explore_search_solvable import explore_from_set_sim
from madrona_scripts.pbge import policy_based_explore_from_set_sim

from generateLevels import jsonDescriptionsToTensor, randomize_spawn_blocks, select_specific_spawn_block, json_parts_to_xyz_bounds
from parameters.params_train import TrainParameters1
from parameters.params_transforms import TransformParameters, generateLimitationStepFunc
from slider_backend_defs import Settings, Parameter, TraceInfo, TrajectoryVisualization

from pathlib import Path
torch.manual_seed(0)
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--host', type=str, default="localhost")
arg_parser.add_argument('--port', type=int, default=8080)
arg_parser.add_argument('--ckpt-path', type=str, nargs='+')
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--num-sim-worlds', type=int, default=1024)
arg_parser.add_argument('--episode-len', type=int, default=300)
arg_parser.add_argument('--num-samples', type=int, default=32)
arg_parser.add_argument('--transform-params', type=str, nargs='+', default=None)
arg_parser.add_argument('--obby-save-load-folder', type=str, default="SavedObbies")
arg_parser.add_argument('--player-trajectory-save-load-folder', type=str, default="SavedPlayerTrajectories")
arg_parser.add_argument('--query-save-load-folder', type=str, default="Queries")
arg_parser.add_argument('--method', type=str, choices=["Ours", "PBGE"], default="Ours")
args = arg_parser.parse_args()

settings = Settings(
    num_worlds=args.num_sim_worlds,
    trials_per_level=100,
    episode_len=args.episode_len,
    num_samples=args.num_samples,
    seed=0,
    transform_params=None,
    ckpt_path=None,
    gpu_sim=args.gpu_sim,
    gpu_id=0
)
ckpt_to_profile = {}
obs = None
sim = None
train_wrapper = None
levels = {}
active_inference_tasks = {}
task_lock = threading.Lock()
sem = threading.Semaphore()


def _extract_part_distribution(param_dist):
    """Returns (distribution name, slider step, min, max, default value) for an obby parameter distribution."""
    # The distribution type is read from the class name.
    distribution = param_dist.__class__.__name__
    step, param_min, param_max, default_val = 0, 0, 0, 0
    if distribution == "Uniform":
        step = (param_dist.max - param_dist.min) / 50
        param_min = param_dist.min
        param_max = param_dist.max
        default_val = (param_min + param_max) / 2
    elif distribution == "UniformInt":
        step = 1
        param_min = param_dist.min
        param_max = param_dist.max
        default_val = (param_min + param_max) // 2
    elif distribution == "Binary":
        param_min = 0
        param_max = 1
        step = 1
        default_val = 0
    elif distribution == "Constant":
        param_min = param_dist.min
        param_max = param_dist.max
        step = 0
        default_val = param_dist.value
    return distribution, step, param_min, param_max, default_val

def _level_json_from_dict(generator, obstacle_param_dict, diffs, name=""):
    """
    Returns the level JSON for generator [generator] with the DSL parameters
    [obstacle_param_dict] (all Constant), after applying the designer's
    per-part edits [diffs] (deleted, moved, resized, re-typed or created parts).
    """
    for param_dist in obstacle_param_dict.values():
        assert isinstance(param_dist, Constant)

    constant_generator = type(generator)(**obstacle_param_dict)
    constant_obby = Obby(name, unit=4)
    constant_obstacle = constant_generator.sample(name, "Checkpoint 0",
                                                  (0, 0, 1.5), (0, 0, 0))
    constant_obby.spawn((0, 0, 0), 3, (0, 0, 0))
    constant_obby.append_parts(constant_obstacle.generate())

    # Remove all deleted parts.
    constant_obby.parts = [p for p in constant_obby.parts \
                           if p.name not in diffs.keys() or not diffs[p.name]["deleted"]]
    
    def createBehavior(behavior_dict):
        match behavior_dict["name"]:
            case "Spawn":
                return SpawnBehavior()
            case "Checkpoint":
                return CheckpointBehavior()
            case "Damage":
                return DamageBehavior(amount=behavior_dict["damage"])
            case "Velocity":
                return VelocityBehavior(velocity=[v / constant_obby.unit for v in behavior_dict["velocity"]])
            case "Carry":
                return CarryBehavior()
            case "PersistentTranslate":
                return PersistentTranslateBehavior(
                    time=behavior_dict["time"],
                    loop=behavior_dict["loop"],
                    easing_style=behavior_dict["easing_style"],
                    easing_direction=behavior_dict["easing_direction"],
                    reverses=behavior_dict["reverses"],
                    start_position=[b / constant_obby.unit for b in behavior_dict["start_position"]],
                    end_position=[b / constant_obby.unit for b in behavior_dict["end_position"]]
                )
            case "PersistentRotate":
                return PersistentRotateBehavior(
                    about=(0,0,0),
                    angular=behavior_dict["angular"],
                    axis=behavior_dict["axis"],
                    loop=behavior_dict["loop"]
                )
            case "Waypoint":
                return WaypointBehavior(index=behavior_dict["index"], jumps=behavior_dict["jumps"] if "jumps" in behavior_dict.keys() else -1)
            case "Avoid":
                return AvoidBehavior()
            case "NoCollide":
                return NoCollideBehavior()
        # Only the behaviors above are supported; others are dropped.
        print("BEHAVIOR NOT FOUND!!!")
        return None
    
    # Apply the per-part edits.
    for part in constant_obby.parts:
        if part.name in diffs.keys():
            part.position = tuple([x + y for x, y in zip(part.position, tuple([diffs[part.name]["position"][i] / constant_obby.unit for i in range(3)]))])
            part.orientation = tuple([x + y for x, y in zip(part.orientation, tuple([diffs[part.name]["orientation"][i] for i in range(3)]))])
            part.size = tuple([x * y for x,y in zip(part.size, tuple([diffs[part.name]["size"][i] for i in range(3)]))])
            part.behaviors = [createBehavior(b) for b in diffs[part.name]["behaviors"]]
            part.part_type = diffs[part.name]["type"]

    # Add the parts the designer created, before the last part.
    for part_name, diff in diffs.items():
        if diff["created"]:
            diff_part = Part(
                part_type=diff["type"],
                name=part_name + f"_{len(constant_obby.parts)}",
                position=tuple(diff["position"][i] / constant_obby.unit for i in range(3)),
                size=tuple(diff["size"][i] / constant_obby.unit for i in range(3)),
                orientation=diff["orientation"],
                behaviors=[createBehavior(b) for b in diff["behaviors"]]
            )

            constant_obby.parts.insert(len(constant_obby.parts) - 1, diff_part)
    
    return constant_obby.jsonify()


def _level_json_from_parameters(level_type: str, parameters: list[Parameter]):
    """
    Returns the level JSON for level type [level_type] with the slider
    values [parameters].
    """
    new_level_params = TrainParameters1(settings.seed)
    generators = new_level_params.get_generators()

    level_dist = generators[level_type]
    dsl_params = {p.param_name: Constant(p.curr_val) for p in parameters if p.dsl_param}

    # Older save files store per-block moves as non-DSL "<block>_transform_<axis>"
    # parameters instead of diffs. Rebuild diffs from them, so that the level
    # starts from the right DSL JSON and the block edits still apply after a DSL
    # parameter changes. Newer save files have no such parameters.
    local_translation_params = {p.param_name: Constant(p.curr_val) for p in parameters if not p.dsl_param}
    diffs = {}
    for transform_name, transform in local_translation_params.items():
        block_name = transform_name[:transform_name.index("_transform")]
        if block_name not in diffs.keys():
            # Older save files only allowed moves.
            diffs[block_name] = {"position": [0,0,0],
                                 "orientation": [0,0,0],
                                 "size": [1,1,1],
                                 "deleted": False,
                                 "created": False}
        # Older save files use a unit scale of 4.
        if "_x" in transform_name: diffs[block_name]["position"][0] = transform.value * 4
        if "_y" in transform_name: diffs[block_name]["position"][1] = transform.value * 4
        if "_z" in transform_name: diffs[block_name]["position"][2] = transform.value * 4

    json_out = _level_json_from_dict(level_dist, dsl_params, diffs, name="")
    return json.loads(json_out)


def _setup():
    """
    Loads all the levels and generators from the obby DSL,
        and creates the simulator.
    """
    global settings
    global ckpt_to_profile

    settings.device = torch.device(
        f'cuda:{settings.gpu_id}') if settings.gpu_sim else torch.device('cpu')
    
    # One limitation profile (TransformParameters) per checkpoint.
    assert args.transform_params is None or len(args.ckpt_path) == len(args.transform_params)
    def _make_profile(profile_json=None):
        transform_params = TransformParameters.default_params(
            settings.num_worlds, settings.device)
        if profile_json is not None:
            transform_params.from_json(profile_json)
        return transform_params

    if args.transform_params is None:
        ckpt_to_profile = {ckpt: _make_profile() for ckpt in args.ckpt_path}
    else:
        ckpt_to_profile = {
            ckpt: _make_profile(profile_json) for (ckpt, profile_json) in zip(args.ckpt_path, args.transform_params)
        }
    settings.transform_params = ckpt_to_profile[args.ckpt_path[0]]

    global sim
    sim = setup_sim(
        num_worlds=settings.num_worlds,
        episode_len=settings.episode_len,
        gpu_sim=settings.gpu_sim,
        gpu_id=settings.gpu_id,
        rand_seed=settings.seed,
        use_complex_level=False,
        use_normaabb_obs=False,
        disable_phys_obs=False,
        disable_step_obs=args.method == "Ours",
        unlock_agent_orientation=False,
        reward_mode="PolicyBasedGoExplore" if args.method == "PBGE" else "Dense1"
    )
    sim.init()

    # Limitation profile of the first checkpoint.
    transform_params = TransformParameters.default_params(
        settings.num_worlds, settings.device)
    if args.transform_params is not None:
        transform_params.from_json(args.transform_params[0])
    settings.transform_params = transform_params


    if settings.ckpt_path:
        settings.policy = load_policy(settings.ckpt_path)
        settings.value_normalizer = load_value_normalizer(settings.ckpt_path)

    # Load every level type and its slider parameters from the DSL.
    train_params = TrainParameters1(settings.seed)
    generators = train_params.get_generators()
    for level in generators:
        level_params = []
        level_dist = generators[level]
        for level_param_name in vars(level_dist):
            param_dist = getattr(level_dist, level_param_name)
            distribution, step, param_min, param_max, default_val = (_extract_part_distribution(param_dist))
            param = Parameter(param_name=level_param_name,
                              min_val=param_min,
                              max_val=param_max,
                              curr_val=default_val,
                              distribution=distribution,
                              step=step,
                              dsl_param=True,
                              enabled=True)
            level_params.append(param)

        levels[level] = {"parameters": level_params}


    # PBGE: a PPO training wrapper that finetunes the policy during the search.
    if args.method == "PBGE":
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "obby-solvability"),
            config=args,
            name="Finetune"
        )
    global train_wrapper
    train_wrapper = PPOTrainingWrapper(
        sim, 
        settings.device,
        settings.seed,
        train_params,
        transform_params,
        ["Plank"], # Must not be empty; each search sets the level.
        num_components=1,
        no_level_obs=settings.no_level_obs,
        num_channels=settings.num_channels,
        separate_value=settings.separate_values,
        use_intrinsic_loss=False,
        normalize_values=True,
        steps_per_update=40,
        single_instance=True,
        disable_random_orientation=True,
        random_spawn=False,
        random_waypoint=False,
        perturb=None,
        perturbation_seed=0,
        run_inference_during_training=False,
        return_based_weighting=False,
        save_frequency=10,
        log_frequency=10,
        eval_frequency=0,
        ckpt_dir=Path("DemoAgents/Finetune"),
        profile_report=False,
        starting_policy=None, # Set by each search.
        level_json=None,
        expert_traj=None,
        # Observation tensor for the bin coordinates; GoExplore reads it from
        # the training wrapper.
        bin_coord_obs = torch.zeros((settings.num_worlds, 6), device=settings.device) if args.method == "PBGE" else None
    )
    return


app = Flask(__name__)


@app.route("/requestAllLevels", methods=["GET"])
def request_all_levels():
    """Returns the names of all level types."""
    return {"status": "success", "levels": list(levels.keys())}

@app.route("/requestSavedLevels", methods=["GET"])
def request_saved_levels():
    if not os.path.exists(args.obby_save_load_folder):
        return jsonify({"error": f"Save folder '{args.obby_save_load_folder}' not found"}), 404

    # File names of the saved levels.
    return {"status": "success", "levels": sorted(
        [x for x in os.listdir(args.obby_save_load_folder) if os.path.isfile(os.path.join(args.obby_save_load_folder, x))])}

@app.route("/requestSavedQueries", methods=["GET"])
def request_saved_queries():
    if not os.path.exists(args.query_save_load_folder):
        return jsonify({"error": f"Query folder '{args.query_save_load_folder}' not found"}), 404

    # File names of the saved queries.
    return {"status": "success", "levels": sorted(
        [x for x in os.listdir(args.query_save_load_folder) if os.path.isfile(os.path.join(args.query_save_load_folder, x))])}


@app.route("/requestSavedPlayerData", methods=["GET"])
def request_saved_player_data():
    if not os.path.exists(args.player_trajectory_save_load_folder):
        return jsonify({"error": f"Save folder '{args.player_trajectory_save_load_folder}' not found"}), 404

    # File names of the saved player recordings.
    return {"status": "success", "player_data_files": sorted([f for f in os.listdir(args.player_trajectory_save_load_folder) if ".json" in f])}


@app.route("/requestAllAgents", methods=["GET"])
def request_all_agents():
    # A --ckpt-path entry can be a folder of .pth files.
    agent_list = []
    for p in args.ckpt_path:
        if os.path.isdir(p):
            agent_list += [os.path.join(p, f) for f in os.listdir(p) if ".pth" in f]
        else:
            agent_list.append(p)

    return {"status": "success", "agents": agent_list}

@app.route("/requestLevelType/<level_type>", methods=["GET"])
def request_level_type(level_type):
    """Returns the default slider parameters and the level JSON of [level_type]."""
    if level_type not in levels:
        return jsonify({"error": f"Level type '{level_type}' not found"}), 404

    level_data = levels[level_type]

    parameters = [param for param in level_data["parameters"]]
    level_json = _level_json_from_parameters(level_type=level_type,
                                    parameters=parameters)
    return jsonify({
        "parameters": [param.to_dict() for param in parameters],
        "level_json": level_json
    })

@app.route("/requestSavedLevel/<level_path>", methods=["GET"])
def request_saved_level(level_path):
    """
    Returns a saved level: its type, DSL parameters, full level JSON (with the
    designer's edits) and the DSL-only level JSON.
    """
    level_path_file = os.path.join(args.obby_save_load_folder, level_path)

    if not (os.path.exists(level_path_file)):
        return jsonify({"error": f"Level path '{level_path_file}' not found"}), 404

    with open(level_path_file, "r") as f:
        level_data = json.loads(f.read())

    parameters = [Parameter.from_dict(param) for param in level_data["parameters"]]

    level_json = None
    if "level_json" in level_data.keys():
        level_json = level_data["level_json"]
    else:
        # Older save files have only the parameters.
        level_json = _level_json_from_parameters(level_type=level_data["level_type"],
                                        parameters=parameters)
    
    # Keep only the DSL parameters of this level type.
    dsl_parameters = levels[level_data["level_type"]]["parameters"]
    def is_dsl_param(x):
        for p in dsl_parameters:
            if x.param_name == p.param_name:
                return True
        return False

    parameters = [p for p in parameters if is_dsl_param(p)]

    return jsonify({
        "level_type": level_data["level_type"],
        "parameters": [p.to_dict() for p in parameters],
        "level_json": level_json,
        "dsl_json": _level_json_from_parameters(level_type=level_data["level_type"],
                                                parameters=parameters)
    })

@app.route("/requestSavedQuery/<query_path>", methods=["GET"])
def request_saved_query(query_path):
    """
    Returns a saved query: the saved level fields plus its step and jump limits,
    start step and goal bounds.
    """
    query_path_file = os.path.join(args.query_save_load_folder, query_path)

    if not (os.path.exists(query_path_file)):
        return jsonify({"error": f"Level path '{query_path_file}' not found"}), 404

    with open(query_path_file, "r") as f:
        level_data = json.loads(f.read())

    parameters = [Parameter.from_dict(param) for param in level_data["parameters"]]

    level_json = None
    if "level_json" in level_data.keys():
        level_json = level_data["level_json"]
    else:
        # Older save files have only the parameters.
        level_json = _level_json_from_parameters(level_type=level_data["level_type"],
                                        parameters=parameters)
    
    # Keep only the DSL parameters of this level type.
    dsl_parameters = levels[level_data["level_type"]]["parameters"]
    def is_dsl_param(x):
        for p in dsl_parameters:
            if x.param_name == p.param_name:
                return True
        return False

    parameters = [p for p in parameters if is_dsl_param(p)]

    bounds = None
    if "boundsPosition" in level_data.keys():
        bounds = {
            "position": level_data["boundsPosition"], 
            "size": level_data["boundsExtents"] 
        }

    return jsonify({
        "level_type": level_data["level_type"],
        "parameters": [p.to_dict() for p in parameters],
        "level_json": level_json,
        "dsl_json": _level_json_from_parameters(level_type=level_data["level_type"],
                                                parameters=parameters),
        "max_traj_steps": level_data["max_traj_steps"] if "max_traj_steps" in level_data.keys() else 3000,
        "max_traj_jumps": level_data["max_traj_jumps"] if "max_traj_jumps" in level_data.keys() else 3000,
        "traj_start_step": level_data["traj_start_step"] if "traj_start_step" in level_data.keys() else 0,
        "bounds_position": bounds["position"] if bounds is not None else None,
        "bounds_size": bounds["size"] if bounds is not None else None
    })


def processTrajectoryFromPlayerData(player_data):
    level_data = player_data[0]["level_payload"]

    level_json = level_data["level_json"]
    
    success_trace_viz_list = []
    failure_trace_viz_list = []

    curr_trajectory = []
    print([player_data.index(d) for d in player_data if not d["alive"]])
    for datapoint in player_data:
        success = datapoint["success"]
        if datapoint["alive"]:
            curr_trajectory.append(datapoint)
        else:
            if len(curr_trajectory) > 0:
                traj_viz = TrajectoryVisualization(
                    x=[d["characterPosition"][0] for d in curr_trajectory],
                    y=[d["characterPosition"][1] for d in curr_trajectory],
                    z=[d["characterPosition"][2] for d in curr_trajectory],
                    t=[(d["characterTimestep"] if "characterTimestep" in d.keys() else d["delta"]) for d in curr_trajectory],
                    reward=[0 for d in curr_trajectory],
                    value=[0 for d in curr_trajectory],
                    action=[[0,0,0,0]for d in curr_trajectory],
                    action_dist=[[0 for j in range(sum(ACTION_DIMS))] for d in curr_trajectory],
                    success=success,
                    block_x=[0 for _ in curr_trajectory],
                    block_y=[0 for _ in curr_trajectory],
                    block_z=[0 for _ in curr_trajectory]
                )
                if success:
                    success_trace_viz_list.append(traj_viz)
                else:
                    failure_trace_viz_list.append(traj_viz)
                curr_trajectory = []

    total_trajectories = len(success_trace_viz_list) + len(failure_trace_viz_list)
    print("Total trajectories: ", total_trajectories, "Successes: ", len(success_trace_viz_list), "Failures: ", len(failure_trace_viz_list))

    return {
        "level_type": level_data["level_type"],
        "parameters": level_data["parameters"],
        "level_json": level_json,
        'success_rate': 1.0,
        'total_trials': total_trajectories,
        'traces': [],
        'trace_viz': [trace.to_dict() for trace in success_trace_viz_list],
        'success_trace_viz': [trace.to_dict() for trace in success_trace_viz_list],
        'failure_trace_viz': [trace.to_dict() for trace in failure_trace_viz_list],
        'human_trace_viz': [],
    }

@app.route("/requestSavedPlayerDatafile/<player_data_path>", methods=["GET"])
def request_saved_player_data_file(player_data_path):
    """Returns the success and failure traces of a saved player recording."""
    player_data_path_file = os.path.join(args.player_trajectory_save_load_folder, player_data_path)

    if not (os.path.exists(player_data_path_file)):
        return jsonify({"error": f"Level path '{player_data_path_file}' not found"}), 404

    player_data = []
    with open(player_data_path_file, "r") as f:
        player_data = json.loads(f.read())


    return jsonify(processTrajectoryFromPlayerData(player_data))



file_timestamp = "0"
current_player_data = []
@app.route("/recordPlayerData", methods=["POST"])
def record_player_data():
    global file_timestamp
    global current_player_data
    player_data_json = request.get_json()
    if len(player_data_json) == 0:
        print("No player data. Ensure Roblox is in play mode")
        return "No player data written"


    if not os.path.exists(args.player_trajectory_save_load_folder):
        os.mkdir(args.player_trajectory_save_load_folder)

    # The plugin sends a recording in chunks. A chunk that starts within 0.5 s
    # of pressing record starts a new file; later chunks append to it.
    if player_data_json[0]["delta"] < 0.5:
        t = time.localtime()
        file_timestamp = f"{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}"
        current_player_data = []

    level_payload = player_data_json[0]["level_payload"]

    current_player_data += player_data_json
    with open(os.path.join(args.player_trajectory_save_load_folder, level_payload["level_type"] + file_timestamp + ".json"), "w") as f:
        f.write(json.dumps(current_player_data))

    return "Wrote player data"

@app.route("/saveCurrentParameters", methods=["POST"])
def save_current_parameters():
    """Saves the posted level (type, parameters, level JSON) as a new file in the save folder."""
    if not request.json:
        return jsonify({"error": f"Need to send parameters"}), 404

    if not os.path.exists(args.obby_save_load_folder):
        os.mkdir(args.obby_save_load_folder)

    level_type = request.json["level_type"]
    
    t = time.localtime()
    saveFilePath = level_type + f"{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.json"

    with open(os.path.join(args.obby_save_load_folder, saveFilePath), "w") as f:
        f.write(json.dumps(request.json))

    return "WROTE JSON"

@app.route("/sendCurrentParameters", methods=["POST"])
def send_current_parameters():
    """Returns the level JSON for the posted level type and slider parameters."""
    if not request.json:
        return jsonify({"error": f"Need to send parameters"}), 404

    level_type = request.json["level_type"]
    parameters = request.json["parameters"]
    parameters = [Parameter.from_dict(param) for param in parameters]
    json_out = _level_json_from_parameters(level_type=level_type,
                                  parameters=parameters)
    return jsonify({
        "level_json": json_out
    })

def load_value_normalizer(ckpt_path):
    value_normalizer = EMANormalizer(0.999,
                                     disable=False)
    weights = LearningState.load_value_normalizer_weights(ckpt_path)
    value_normalizer.load_state_dict(weights, strict=False)
    value_normalizer.train(False)
    value_normalizer = value_normalizer.to(settings.device)
    return value_normalizer

# Returns the policy loaded from [ckpt_path]; does not change settings.
def load_policy(ckpt_path):
    _, num_obs_features = setup_obs(sim, sim.enum_counts_tensor().to_torch().to(
    settings.device), settings.no_level_obs)
    policy = make_policy(num_obs_features, None, settings.num_channels,
                            settings.separate_values, ACTION_DIMS)
    weights = LearningState.load_policy_weights(ckpt_path)
    policy.load_state_dict(weights, strict=False)
    policy = policy.to(settings.device)
    return policy

@app.route("/updateBackendAgent", methods=['GET', 'POST'])
def update_policy():
    """Loads the posted agent (checkpoint path) as the current policy; "" clears it."""
    if not request.json:
        return jsonify({"error": "Need to send agent"})
    
    global settings
    global ckpt_to_profile
    
    agent = request.json.get("agent", "")

    if agent == "":
        settings.ckpt_path = ""
        settings.policy = None
        settings.value_normalizer = None
        # Keep existing transform params unchanged
        return jsonify({})

    settings.ckpt_path = agent
    settings.policy = load_policy(settings.ckpt_path)
    settings.value_normalizer = load_value_normalizer(settings.ckpt_path)

    # Use the agent's limitation profile; a new agent gets the default one.
    if agent not in ckpt_to_profile.keys():
        print(agent, ckpt_to_profile.keys())
        print("Creating new profile")
        ckpt_to_profile[agent] = TransformParameters.default_params(
            settings.num_worlds, settings.device)
    settings.transform_params = ckpt_to_profile[agent]


    print(settings.transform_params.action_transform.p_repeat[0])

    return jsonify({})

solution_streaming_info = {
    "should_stop" : False,
    "current_go_explore": None,
    "current_best_bin": None
}
@app.route('/cancelCurrentOperation', methods=['GET'])
def cancel():
    global solution_streaming_info
    solution_streaming_info["should_stop"] = True
    for i in range(5):
        if not solution_streaming_info["should_stop"]: return "Canceled"
        time.sleep(0.1)
    solution_streaming_info["should_stop"] = False
    return "TIMEOUT did not cancel"

previous_valid_bins = None

@app.route('/resetGoExplore', methods=['GET'])
def reset_go_explore():
    global solution_streaming_info
    global previous_valid_bins
    solution_streaming_info["should_stop"] = False
    solution_streaming_info["current_go_explore"] = None    
    solution_streaming_info["current_best_bin"] = None
    previous_valid_bins = None
    return "Reset GoExplore"

@app.route('/findSuccessfulTrajectory', methods=['POST'])
def findSuccessfulTrajectory():
    t = time.time()
    global previous_valid_bins
    global solution_streaming_info
    global train_wrapper
    previous_valid_bins = None
    data = request.get_json()

    # Save every query to the query folder.
    if not os.path.exists(args.query_save_load_folder):
        os.mkdir(args.query_save_load_folder)
    file_t = time.localtime()
    queryFilePath = f"{file_t.tm_mday}_{file_t.tm_hour}_{file_t.tm_min}_{file_t.tm_sec}.json"
    with open(os.path.join(args.query_save_load_folder, queryFilePath), "w") as f:
        f.write(json.dumps(data))

    level_type = data['level_type']
    parameters = [Parameter.from_dict(param) for param in data['parameters']]

    train_params = TrainParameters1(settings.seed)
    generators = train_params.get_generators()
    level_dist = generators[level_type]
    obby_name = f"Random{data['level_type']}_s{settings.seed}"

    curr_obstacle_param_dict = {p.param_name: Constant(p.curr_val) for p in parameters if p.dsl_param}
    curr_level_json = json.dumps(data["level_json"])

    sample_level_jsons = [curr_level_json]

    # When resuming from an archive slot, keep the level that is in the simulator.
    if "starting_slot_idx" not in data.keys():
        json_levels_tensor = sim.json_level_descriptions_tensor().to_torch()
        json_indices = sim.json_index_tensor().to_torch().to(settings.device)

        # Every world gets the same level.
        sample_level_jsons.extend([sample_level_jsons[0] for i in range(json_levels_tensor.shape[0] - len(sample_level_jsons))])

        json_indices[..., 0] = 0
        json_levels_cpu = jsonDescriptionsToTensor(
            json_levels_tensor.shape, sample_level_jsons, json_levels_physics=None)
        json_levels_tensor[:] = json_levels_cpu.to(settings.device)

    bounds = None
    if "boundsPosition" in data.keys():
        bounds = {
            "position": data["boundsPosition"], 
            "size": data["boundsExtents"] 
        }


    has_animated_object = False
    has_dynamic_object = False
    has_limited_jumps = data["max_traj_jumps"] < 10 and data["max_traj_jumps"] > 0


    for _, desc in data["level_json"].items():
        for b in desc["behaviors"]:
            if b['name'] == "Unanchored":
                has_dynamic_object = True
            if "Translate" in b['name'] or "Rotate" in b['name']:
                has_animated_object = True

    # Search settings shared by Go-Explore and PBGE.
    num_exploration_steps = 40
    num_sim_steps_per_exploration_step = min(data["max_traj_steps"], 200 if args.method == "Ours" else 1000)
    # Bin grid: full x/y/z resolution; t, jump and dynamic-object bins only when
    # the level needs them. Each extra axis halves another one to bound the bin count.
    num_x_bins = 32
    num_y_bins = 32
    num_z_bins = 8
    num_t_bins = 16 if has_animated_object else 1
    num_j_bins = 1
    num_dyn_bins = 1

    if has_limited_jumps:
        if has_animated_object:
            num_x_bins //= 2
            num_y_bins //= 2
        num_j_bins = data["max_traj_jumps"]
    if has_dynamic_object:
        if has_animated_object:
            num_t_bins //= 2
        if has_limited_jumps:
            num_z_bins //= 2
        num_dyn_bins = 8

    # Probability of repeating the previous random action.
    repeat_action_probability = 0.95

    if args.method == "PBGE":
        train_wrapper.ckpt_dir=Path("DemoAgents/PBGE")
        # PBGE starts each search from a fresh policy.
        train_wrapper.policy = make_policy(train_wrapper.num_obs_features, None, train_wrapper.num_channels,
                            train_wrapper.separate_value, ACTION_DIMS)
        # Runs in the background and stores its GoExplore object in
        # solution_streaming_info itself.
        policy_based_explore_from_set_sim(sim,
                    level_json=data["level_json"],
                    num_worlds=settings.num_worlds,
                    num_steps=num_exploration_steps,
                    num_x_bins=num_x_bins,
                    num_y_bins=num_y_bins,
                    num_z_bins=num_z_bins,
                    num_t_bins=num_t_bins,
                    num_j_bins=num_j_bins,
                    num_dyn_bins=num_dyn_bins,
                    exploration_steps=num_sim_steps_per_exploration_step,
                    max_traj_steps=data["max_traj_steps"],
                    max_traj_jumps=data["max_traj_jumps"],
                    bounds=bounds,
                    repeat_p=repeat_action_probability,
                    break_when_solved=True,
                    store_successful_trajs=True,
                    gpu_sim=settings.gpu_sim,
                    gpu_id=settings.gpu_id,
                    seed=settings.seed,
                    dev=settings.device,
                    action_policy=train_wrapper.policy,
                    use_policy_selection=False,
                    solution_streaming_info=solution_streaming_info,
                    train_wrapper=train_wrapper,
                    run_in_background=True)
    elif args.method == "Ours":
        # Starts the search in the background and returns its GoExplore object.
        go_explore = explore_from_set_sim(sim,
                    level_json=data["level_json"],
                    num_worlds=settings.num_worlds,
                    num_steps=num_exploration_steps,
                    num_x_bins=num_x_bins,
                    num_y_bins=num_y_bins,
                    num_z_bins=num_z_bins,
                    num_t_bins=num_t_bins,
                    num_j_bins=num_j_bins,
                    num_dyn_bins=num_dyn_bins,
                    exploration_steps=num_sim_steps_per_exploration_step,
                    max_traj_steps=data["max_traj_steps"],
                    max_traj_jumps=data["max_traj_jumps"],
                    bounds=bounds,
                    repeat_p=repeat_action_probability,
                    # break_when_solved=False is not supported here.
                    break_when_solved=True, 
                    store_successful_trajs=True,
                    gpu_sim=settings.gpu_sim,
                    gpu_id=settings.gpu_id,
                    seed=settings.seed,
                    dev=settings.device,
                    action_policy=settings.policy,
                    use_policy_selection=True,
                    solution_streaming_info=solution_streaming_info,
                    record_trajectories=False,
                    starting_slot_idx = -1 if "starting_slot_idx" not in data.keys() else data["starting_slot_idx"],
                    track_block_trajectories=True,
                    run_in_background=True)
        solution_streaming_info["current_go_explore"] = go_explore
    print("Total time:", time.time() - t)
    return jsonify({})

def _slot_block_trajectories(go_explore):
    """Per-step dynamic-block positions of every archive slot: the paper GoExplore
    keeps them in slot_block_trajectories, PBGE's in slot_dynamic_block_pos."""
    if getattr(go_explore, "slot_block_trajectories", None) is not None:
        return go_explore.slot_block_trajectories
    return go_explore.slot_dynamic_block_pos


@app.route('/getTrajectoryAtSlot', methods=['POST'])
def get_trajectory_at_slot():
    data = request.get_json()
    slot_idx = data["slot_idx"]
    go_explore = solution_streaming_info["current_go_explore"]

    valid_step = go_explore.slot_trajectories_step[slot_idx]
    
    positions = go_explore.slot_pos_trajectories[slot_idx, :valid_step, ...]
    actions = go_explore.slot_action_trajectories[slot_idx, :valid_step, ...]
    block_positions = _slot_block_trajectories(go_explore)[slot_idx, :valid_step, ...]

    trace = TrajectoryVisualization(
        x=positions[:, 0].tolist(),
        y=positions[:, 2].tolist(),
        z=(-positions[:, 1]).tolist(),
        t=[i for i in range(len(positions))],
        reward=[0 for i in range(len(positions))],
        value=[0 for i in range(len(positions))],
        action=[actions[i, ...].tolist() for i in range(len(actions))],
        action_dist=[[0 for j in range(sum(ACTION_DIMS))] for i in range(len(positions))],
        success=False,
        block_x=block_positions[:, 0].tolist(),
        block_y=block_positions[:, 2].tolist(),
        block_z=(-block_positions[:, 1]).tolist()
    )
    
    return jsonify({
        'trace_viz': [trace.to_dict()],
    })
    
@app.route('/getCurrentTrajectory', methods=['POST'])
def get_current_trajectory():
    global previous_valid_bins
    global solution_streaming_info
    data = request.get_json()
    go_explore = solution_streaming_info["current_go_explore"]
    # No search is running (none started, or it was cancelled).
    if go_explore is None:
        print("No GoExplore object found")
        return jsonify({
            'success_rate': 0.0,
            'total_trials': 0,
            'traces': [],
            'trace_viz': [],
            'success_trace_viz': [],
            'failure_trace_viz': [],
            'human_trace_viz': [],
            'should_continue': True
        })

    valid_slot_indices = torch.where(go_explore._slot_to_coarse != -1)[0]
    num_valid_slots = int(torch.sum(go_explore._slot_to_coarse != -1))

    if num_valid_slots == go_explore.POOL_SIZE:
        print("ARCHIVE FULL!")

    valid_bins = torch.unique(go_explore._slot_to_coarse[:num_valid_slots])

    if previous_valid_bins is not None and torch.any(~torch.isin(previous_valid_bins, valid_bins)):
        print("Lost a valid bin!")
        print("prev valid bins", previous_valid_bins)
        print("valid_bins", valid_bins)
    previous_valid_bins = valid_bins

    assert len(valid_bins) == num_valid_slots

    bin_is_new = torch.ones_like(valid_bins, dtype=torch.bool)
    if previous_valid_bins is not None:
        bin_is_new = ~torch.isin(valid_bins, previous_valid_bins)
    previous_valid_bins = valid_bins

    slot_scores = go_explore.slot_rewards[:num_valid_slots] + 1
    slot_scores = torch.clamp(slot_scores, 0, go_explore.target_return)


    # Representative point of each archive slot: the last position and step of
    # its trajectory.
    slot_rep_steps = go_explore.slot_trajectories_step[:num_valid_slots]
    final_pos_index = slot_rep_steps.to(torch.int64).view(
        slot_rep_steps.shape[0], 1, 1
    ).expand(-1, 1, 3)
    slot_rep_positions = torch.gather(go_explore.slot_pos_trajectories[:num_valid_slots], dim=1, index=torch.maximum(final_pos_index - 1, torch.zeros_like(final_pos_index))).squeeze()
    if len(slot_rep_positions.shape) == 1:
        slot_rep_positions = slot_rep_positions.unsqueeze(0)
    slot_rep_steps = slot_rep_steps.unsqueeze(1)
    slot_rep_points = torch.cat((slot_rep_positions, slot_rep_steps), dim=1)


    goal_rep_points = None
    intermediate_goal_rep_points = None

    # PBGE: the bins that the worlds currently target as goals.
    if args.method == "PBGE":
        exploring_worlds_mask = go_explore.returning_world_trajs_final_step_idx == -1
        tensors_to_cat = []
        if torch.any(exploring_worlds_mask):
            tensors_to_cat.append(go_explore.exploring_world_goal_cells[exploring_worlds_mask].view(-1))
        # Worlds that return to a cell target the end of their stored trajectory.
        if torch.any(~exploring_worlds_mask):
            tensors_to_cat.append(go_explore.returning_world_trajs_bins[~exploring_worlds_mask, go_explore.returning_world_trajs_final_step_idx[~exploring_worlds_mask]].view(-1))

        goal_bins = torch.unique(
            torch.cat(tensors_to_cat)
        )
        goal_rep_points = go_explore.binning_function.rep_points[goal_bins[goal_bins > -1].int()]


    x_grid, y_grid, z_grid, t_grid, j_grid, dyn_grid = go_explore.binning_function.get_grid()
    bin_size = [abs(float(grid[0] - grid[-1])) / max(1, len(grid) - 1) for grid in (x_grid, y_grid, z_grid, t_grid)]

    valid_rep_points = go_explore.binning_function.rep_points[valid_bins]

    # Convert to Roblox coordinates (y up): (x, y, z) -> (x, z, -y).
    bin_history = go_explore.bin_history[valid_bins]
    bin_history[..., 1] = go_explore.bin_history[valid_bins, ..., 2]
    bin_history[..., 2] = -go_explore.bin_history[valid_bins, ..., 1]
    # Novelty per time chosen.
    bin_history[..., 6] /= torch.maximum(torch.ones_like(bin_history[..., 5]), bin_history[..., 5])

    sorted_indices = torch.argsort(slot_scores, descending=True)
    # The trajectory of the highest-scoring slot.
    valid_traj = go_explore.slot_pos_trajectories[sorted_indices[0]].unsqueeze(0)
    valid_action = go_explore.slot_action_trajectories[sorted_indices[0]].unsqueeze(0)
    valid_step = go_explore.slot_trajectories_step[sorted_indices[0]].unsqueeze(0)
    valid_block_pos = _slot_block_trajectories(go_explore)[sorted_indices[0]].unsqueeze(0)


    # Trajectory visualizations, split into successes and failures.
    trace_viz_list = []
    success_trace_viz_list = []
    failure_trace_viz_list = []

    for i in range(len(go_explore.successful_trajs_pos)):
        positions = go_explore.successful_trajs_pos[i]
        actions = go_explore.successful_trajs_action[i]
        block_positions = go_explore.successful_trajs_dynamic_block_pos[i]
        traj_viz = TrajectoryVisualization(
            x=positions[:, 0].tolist(),
            y=positions[:, 2].tolist(),
            z=(-positions[:, 1]).tolist(),
            t=[i for i in range(len(positions))],
            reward=[0 for i in range(len(positions))],
            value=[0 for i in range(len(positions))],
            action=[actions[i, ...].tolist() for i in range(len(actions))],
            action_dist=[[0 for j in range(sum(ACTION_DIMS))] for i in range(len(positions))],
            success=True,
            block_x=block_positions[:, 0].tolist(),
            block_y=block_positions[:, 2].tolist(),
            block_z=(-block_positions[:, 1]).tolist()
        )
        success_trace_viz_list.append(traj_viz)

    
    if valid_step.shape[0] > 0:
        # The best trajectory so far, reported as a failure until solved.
        positions = valid_traj[0, :valid_step[0], ...]
        actions = valid_action[0, :valid_step[0], ...]
        block_positions = valid_block_pos[0, :valid_step[0], ...]

        success = False

        # TODO: positions can contain NaNs; the cause is not known.
        if torch.any(torch.isnan(positions)):
            print("NAN in position tensor")
        if torch.any(torch.isnan(actions)):
            print("NAN in action tensor")

        traj_viz = TrajectoryVisualization(
            x=positions[:, 0].tolist(),
            y=positions[:, 2].tolist(),
            z=(-positions[:, 1]).tolist(),
            t=[i for i in range(len(positions))],
            reward=[0 for i in range(len(positions))],
            value=[0 for i in range(len(positions))],
            action=[actions[i, ...].tolist() for i in range(len(actions))],
            action_dist=[[0 for j in range(sum(ACTION_DIMS))] for i in range(len(positions))],
            success=False,
            block_x=block_positions[:, 0].tolist(),
            block_y=block_positions[:, 2].tolist(),
            block_z=(-block_positions[:, 1]).tolist()
        )
        if success:
            success_trace_viz_list.append(traj_viz)
        else:
            failure_trace_viz_list.append(traj_viz)


    # The search stops at its first solution, so at most one success is sent.
    num_successes_to_send = min(1, len(success_trace_viz_list))
    num_failures_to_send = min(1, len(failure_trace_viz_list))

    total_trajectories = num_successes_to_send + num_failures_to_send 

    return jsonify({
        'total_trials': total_trajectories,
        'trace_viz': [trace.to_dict() for trace in trace_viz_list],
        'success_trace_viz': [trace.to_dict() for trace in success_trace_viz_list[:num_successes_to_send]],
        'failure_trace_viz': [trace.to_dict() for trace in failure_trace_viz_list[:num_failures_to_send]],
        'bin_size': [bin_size[0], bin_size[2], bin_size[1], bin_size[3]],
        'bin_points': [[v[0], v[2], -v[1], v[3]] for v in valid_rep_points.cpu().tolist()],
        'ckpt_points': [[c[0], c[2], -c[1], c[3]] for c in slot_rep_points.cpu().tolist()],
        'slot_idx': valid_slot_indices.cpu().tolist(),
        'goal_rep_points': [[g[0], g[2], -g[1], g[3]] for g in goal_rep_points.cpu().tolist()] if goal_rep_points is not None else [],
        'intermediate_goal_rep_points': [[i[0], i[2], -i[1], i[3]] for i in intermediate_goal_rep_points.cpu().tolist()] if intermediate_goal_rep_points is not None else [],
        'bin_new': bin_is_new.cpu().tolist(),
        'human_trace_viz': [],
        'should_continue': not go_explore.finished
    })

@app.route("/requestFinetuning", methods=["POST"])
def request_finetuning():

    t = time.time()
    # Finetune the current policy on the posted level and its parameter sweeps.
    starting_policy = settings.policy

    print("Finetuning from policy", settings.ckpt_path)

    data = request.get_json()
    level_type = data['level_type']
    parameters = [Parameter.from_dict(param) for param in data['parameters']]

    train_params = TrainParameters1(settings.seed)
    generators = train_params.get_generators()
    level_dist = generators[level_type]
    obby_name = f"Random{data['level_type']}_s{settings.seed}"

    curr_obstacle_param_dict = {p.param_name: Constant(p.curr_val) for p in parameters if p.dsl_param}
    curr_level_json = _level_json_from_dict(level_dist, curr_obstacle_param_dict, data["diffs"], name="")


    json_levels_tensor = sim.json_level_descriptions_tensor().to_torch()
    json_indices = sim.json_index_tensor().to_torch().to(settings.device)

    # The current level, then one level per value of each enabled parameter.
    sample_level_jsons = [curr_level_json]
    deltas_per_param = {}
    modifiable_params = [p for p in parameters if p.distribution in ["Uniform", "UniformInt"] and p.enabled]
    for level_param in modifiable_params:
        if level_param.distribution == "UniformInt":
            deltas = np.arange(level_param.min_val, level_param.max_val + 1).tolist()
        else:
            deltas = np.linspace(level_param.min_val, level_param.max_val, num=settings.num_samples).tolist()
        deltas_per_param[level_param.param_name] = deltas
    for level_param in modifiable_params:
        for delta in deltas_per_param[level_param.param_name]:
            if level_param.dsl_param:
                perturb_obstacle_params = curr_obstacle_param_dict.copy()
                perturb_obstacle_params[level_param.param_name] = Constant(max(min(delta, level_param.max_val), level_param.min_val))
                level_json = _level_json_from_dict(level_dist, perturb_obstacle_params, data["diffs"], name="")
            # Only DSL parameters are swept; per-block parameters are not.
            sample_level_jsons.append(level_json)

    # Fill the remaining level slots with the current level.
    sample_level_jsons.extend([sample_level_jsons[0] for i in range(json_levels_tensor.shape[0] - len(sample_level_jsons))])
    # Random spawn block per level (randomize_spawn_blocks sets the probabilities).
    sample_level_jsons = [json.loads(p) for p in sample_level_jsons]
    randomize_spawn_blocks(sample_level_jsons, random.Random())
    sample_level_jsons = [json.dumps(p) for p in sample_level_jsons]
    
    # Split the worlds evenly across the levels.
    assert settings.num_worlds > len(sample_level_jsons), "More JSON descriptions than worlds!"
    worlds_per_level = settings.num_worlds // len(sample_level_jsons)

    # Assign each world its level.
    for i in range(len(sample_level_jsons)):
        json_indices[i * worlds_per_level : (i + 1) * worlds_per_level, 0] = i
    json_levels_tensor[:] = jsonDescriptionsToTensor(
        json_levels_tensor.shape, sample_level_jsons, json_levels_physics=None).to(settings.device)

    limitation_step_func = generateLimitationStepFunc(
        sim, 
        train_wrapper.obs, 
        train_wrapper.action_transform, 
        train_wrapper.obs_transform, 
        settings.device
    )

    train_wrapper.ckpt_path = Path("DemoAgents/Finetune")

    # Checkpoints go to DemoAgents/Finetune.
    train(
        settings.device,
        SimInterface(
            step = limitation_step_func,
            obs = train_wrapper.obs,
            actions = train_wrapper.actions,
            dones = train_wrapper.dones,
            rewards = train_wrapper.rewards,
            resets = train_wrapper.resets,
            checkpoints = train_wrapper.checkpoints,
            checkpoint_resets = train_wrapper.checkpoint_resets,
        ),
        TrainConfig(
            run_name="Finetune",
            num_updates=100,
            steps_per_update=train_wrapper.steps_per_update,
            num_bptt_chunks=1,
            lr=0.0002,
            gamma=0.998,
            gae_lambda=0.95,
            ppo = PPOConfig(
                num_mini_batches=1,
                clip_coef=0.2,
                value_loss_coef=0.5,
                use_intrinsic_loss=train_wrapper.use_intrinsic_loss,
                value_loss_intrinsic_coef=0.5,
                intrinsic_loss_coef=0.5,
                entropy_coef=0.001,
                max_grad_norm=0.5,
                num_epochs=2,
                clip_value_loss=True,
                no_advantages=False,
            ),
            value_normalizer_decay=0.999,
            mixed_precision=False,
            normalize_advantages=True,
            normalize_values=True,
        ),
        starting_policy,
        train_wrapper,
        None
    )

    finetune_wallclock_time = time.time() - t
    print("Time elapsed(after finetune)", finetune_wallclock_time)

    return jsonify(f"Finetuned in {finetune_wallclock_time} seconds.")

@app.route("/requestSuccessRates", methods=["POST"])
def request_success_rates():
    """
    Runs the current policy on the posted level and on a sweep of each enabled
    parameter, and returns the success rate and trajectories of each level.
    """
    t = time.time()
    json_levels_tensor = sim.json_level_descriptions_tensor().to_torch()
    json_indices = sim.json_index_tensor().to_torch().to(settings.device)

    data = request.get_json()
    level_type = data['level_type']
    parameters = [Parameter.from_dict(param) for param in data['parameters']]

    train_params = TrainParameters1(settings.seed)
    generators = train_params.get_generators()
    level_dist = generators[level_type]
    obby_name = f"Random{data['level_type']}_s{settings.seed}"

    curr_obstacle_param_dict = {p.param_name: Constant(p.curr_val) for p in parameters if p.dsl_param}
    curr_level_json = _level_json_from_dict(level_dist, curr_obstacle_param_dict, data["diffs"], name="")

    # Sweep values of each enabled parameter: every integer for UniformInt,
    # num_samples evenly spaced values for Uniform.
    deltas_per_param = {}
    modifiable_params = [p for p in parameters if p.distribution in ["Uniform", "UniformInt"] and p.enabled]
    for level_param in modifiable_params:
        if level_param.distribution == "UniformInt":
            deltas = np.arange(level_param.min_val, level_param.max_val + 1).tolist()
        else:
            deltas = np.linspace(level_param.min_val, level_param.max_val, num=settings.num_samples).tolist()
        deltas_per_param[level_param.param_name] = deltas

    sample_level_jsons = [curr_level_json]
    for level_param in modifiable_params:
        for delta in deltas_per_param[level_param.param_name]:
            if level_param.dsl_param:
                perturb_obstacle_params = curr_obstacle_param_dict.copy()
                perturb_obstacle_params[level_param.param_name] = Constant(max(min(delta, level_param.max_val), level_param.min_val))
                level_json = _level_json_from_dict(level_dist, perturb_obstacle_params, data["diffs"], name="")
            # Only DSL parameters are swept; per-block parameters are not.
            sample_level_jsons.append(level_json)

    # Split the worlds evenly across the levels.
    assert settings.num_worlds > len(sample_level_jsons), "More JSON descriptions than worlds!"
    worlds_per_level = settings.num_worlds // len(sample_level_jsons)
    remainder = settings.num_worlds % len(sample_level_jsons)
    print("Worlds per level:", worlds_per_level)

    # Fill the remaining level slots with the first level.
    sample_level_jsons.extend([sample_level_jsons[0] for i in range(json_levels_tensor.shape[0] - len(sample_level_jsons))])



    # Assign each world its level.
    for i in range(len(sample_level_jsons)):
        json_indices[i * worlds_per_level : (i + 1) * worlds_per_level, 0] = i
    json_levels_tensor[:] = jsonDescriptionsToTensor(
        json_levels_tensor.shape, sample_level_jsons, json_levels_physics=None).to(settings.device)


    # One trial per world.
    min_worlds_per_level = int(worlds_per_level)
    num_trials = 1
    print("Trials per level:", num_trials)

    # Run the policy on every world.
    sem.acquire()
    obs, _ = setup_obs(sim,
                       sim.enum_counts_tensor().to_torch().to(settings.device),
                       settings.no_level_obs)
    print(settings.ckpt_path, ckpt_to_profile[settings.ckpt_path], settings.transform_params.action_transform.p_random[0])
    infer_data = run_infer(
        sim,
        settings.transform_params,
        settings.policy,  # one policy or a list of policies
        obs,
        settings.num_worlds,
        num_trials,
        settings.episode_len,
        success_counts=None,
        pre_coordinate_transform_func=None,
        use_complex_level=False,
        use_normaabb_obs=False,
        unlock_agent_orientation=False,
        store_positions=True,
        use_best_action=False,
        record_log=None,
        device=settings.device,
        logging=True,
        value_normalizer=settings.value_normalizer
    )
    sem.release()

    print("Time elapsed(after infer)", time.time() - t)

    # Success rate of each level: the worlds that ran it are consecutive.
    all_positions = infer_data['all_positions']
    num_copies = np.where(np.arange(len(sample_level_jsons)) < remainder,
                          np.ceil(worlds_per_level).astype(int),
                          np.floor(worlds_per_level).astype(int))

    success_per_world = infer_data['success_per_world']
    curr_obstacle_success = np.sum(success_per_world[:num_copies[0]]) / (
            num_copies[0] * num_trials)
    trace_infos = []

    delta_idx = 1
    total_successes = 0
    for level_param in modifiable_params:
        success_rates = []
        for _ in deltas_per_param[level_param.param_name]:
            level_world_min = np.sum(num_copies[:delta_idx])
            level_world_max = np.sum(num_copies[:delta_idx + 1])
            total_trials = (level_world_max - level_world_min) * num_trials
            successes = np.sum(success_per_world[level_world_min:level_world_max])
            total_successes += successes
            success_rates.append(float(successes / total_trials))
            delta_idx += 1
        trace_infos.append(
            TraceInfo(
                param_name=level_param.param_name,
                values=deltas_per_param[level_param.param_name],
                success_rates=success_rates
            )
        )

    # Trajectory visualizations of the current level, split into successes and failures.
    trace_viz_list = []
    success_trace_viz_list = []
    failure_trace_viz_list = []
    avg_traj_length = 0
    for i in range(num_copies[0]):
        rewards = all_positions[i, :, 0]
        positions = all_positions[i, :, 1:4]
        dones = all_positions[i, :, 5]
        timesteps = all_positions[i, :, 6]
        values = all_positions[i, :, 7]
        # Action distribution: move (2), direction (8), rotate (3), jump (2).
        action_dists = all_positions[i, :, 8:23]


        traj_ends = np.where(dones==1)[0]
        traj_start = 0
        for traj_end in traj_ends:
            full_traj = positions[traj_start:traj_end, :]
            success = bool(rewards[traj_end] - 9.9999 > 0)
            traj_viz = TrajectoryVisualization(
                x=positions[traj_start:traj_end, 0].tolist(),
                y=positions[traj_start:traj_end, 2].tolist(),
                z=(-positions[traj_start:traj_end, 1]).tolist(),
                t=timesteps[traj_start:traj_end].tolist(),
                # A step's reward arrives with the next state, so rewards are shifted by 1.
                reward=rewards[traj_start+1:traj_end+1].tolist(),
                value=values[traj_start:traj_end].tolist(),
                action=[[0,0,0,0]for i in range(traj_end-traj_start)],
                action_dist=action_dists[traj_start:traj_end].tolist(),
                success=success
            )
            avg_traj_length += traj_end - traj_start
            traj_start = traj_end
            if success:
                success_trace_viz_list.append(traj_viz)
            else:
                failure_trace_viz_list.append(traj_viz)
            trace_viz_list.append(traj_viz)

    trace_list = [trace.to_dict() for trace in trace_infos]

    print("Success trace viz list:", len(success_trace_viz_list))
    print("Failure trace viz list:", len(failure_trace_viz_list))
    print("Average trajectory length: ", avg_traj_length / (len(success_trace_viz_list) + len(failure_trace_viz_list)))
    print("Total time:", time.time() - t)

    num_successes_to_send = len(success_trace_viz_list)
    num_failures_to_send = len(failure_trace_viz_list)

    return jsonify({
        'success_rate': curr_obstacle_success,
        'total_trials': int(num_copies[0] * num_trials),
        'traces': trace_list,
        'trace_viz': [trace.to_dict() for trace in trace_viz_list],
        'success_trace_viz': [trace.to_dict() for trace in success_trace_viz_list[:num_successes_to_send]],
        'failure_trace_viz': [trace.to_dict() for trace in failure_trace_viz_list[:num_failures_to_send]],
        'human_trace_viz': []
    })


# Allow cross-origin requests from Roblox Studio.
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    _setup()
    app.run(debug=False, host=args.host, port=args.port)
