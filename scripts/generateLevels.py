import copy
import json
import math
import os
from random import random
import struct

import numpy as np
import torch
from madrona_scripts.level_physics import Physics
from madrona_scripts.parameters.enums import *
from madrona_scripts.parameters.params_train import TrainParameters1
from obby.examples.distribution_generators import *
from obby.grammar.part import Part
from obby.grammar.obstacles import *
from obby.grammar.behaviors import *
from obby.utils.vector import vadd, vmul

# A block that can hold a spawn point or a waypoint: no damage, no checkpoint, not a cylinder.
def is_safe(part):
    for behavior in part["behaviors"]:
        if behavior["name"] == "Damage" or behavior["name"] == "Checkpoint" or part["type"] == "Cylinder":
            return False
    return True

# Add one waypoint per level, at a random point near a random safe block.
def add_randomized_waypoints(json_levels: list[dict], block_selector):
    for json_level in json_levels:
        safe_blocks = [json_level[k] for k in json_level.keys() if is_safe(json_level[k]) and json_level[k]["type"] != "SpawnLocation"]
        # A random point within rough reachable bounds of a random safe block.
        chosen_block = block_selector.choice(safe_blocks)
        expansion = 2
        max_height = 8
        eps = 0.01
        x_block, z_block = chosen_block["size"][0] * 0.5, chosen_block["size"][2] * 0.5
        y = block_selector.uniform(0, max_height)

        x = block_selector.uniform(-(expansion + x_block), expansion + x_block)
        z = block_selector.uniform(-(expansion + z_block), expansion + z_block)

        x, y, z = vadd(chosen_block["position"], (x, y, z))

        # Some of these waypoints are not reachable.

        waypoint_part = Part(
            part_type="Block",
            name="Waypoint" + f"_{len(json_level)}",
            position=(x,y,z),
            size=(1,1,1),
            orientation=(0,0,0),
            behaviors=[
                # The only waypoint in the level.
                WaypointBehavior(index=1, jumps=-1), NoCollideBehavior()
            ]
        )

        waypoint_name, waypoint_dict = waypoint_part.jsonify()
        json_level[waypoint_name] = waypoint_dict

# Load trajectory file and extract waypoints from agent positions
def load_trajectory_waypoints(traj_file_path, num_waypoints=5):
    """
    Load a trajectory file (.pt or .out format) and extract waypoints.
    
    Args:
        traj_file_path: Path to the trajectory file (.pt format from go_explore_search_solvable.py
                        or .out/.bin binary format from viewer recording)
        num_waypoints: Number of waypoints to extract (default 5)
    
    Returns:
        List of (x, y, z) positions for waypoints
    """
    import torch
    
    _, file_extension = os.path.splitext(traj_file_path)
    
    if file_extension.lower() == '.pt':
        # Handle PyTorch .pt file format
        traj_data = torch.load(traj_file_path, map_location='cpu')
    
        # Extract trajectories (list of position tensors)
        trajectories = traj_data.get("trajectories", [])
        
        if not trajectories:
            print(f"Warning: No trajectories found in {traj_file_path}")
            return []
    
        # Use the first trajectory.
        traj = trajectories[0]  # Shape: [num_steps, 3] where 3 is (x, y, z)
    
        # Subsample num_waypoints positions evenly across the trajectory
        num_steps = traj.shape[0]
        if num_steps < num_waypoints:
            print(f"Warning: Trajectory has only {num_steps} steps, using all as waypoints")
            indices = range(num_steps)
        else:
            # Evenly sample indices across the trajectory
            indices = torch.linspace(0, num_steps - 1, num_waypoints).long()
    
        # Extract waypoints
        waypoints = []
        for idx in indices:
            pos = traj[idx]
            waypoints.append((float(pos[0]), float(pos[1]), float(pos[2])))
    
        return waypoints
    
    elif file_extension.lower() in ['.out', '.bin']:
        # Handle binary .out/.bin file format
        # Binary format:
        # - First 4 bytes: level seed (int32)
        # - Followed by Trajectory structs (52 bytes each):
        #   int32_t move_amount, move_angle, r, interact
        #   float state_x, state_y, state_z, theta, health, reward
        #   uint32_t world_idx
        #   float deltaT
        #   uint32_t episode
        
        seed_struct = struct.Struct('<i')  # 4 bytes for level seed
        traj_struct = struct.Struct('<iiiiffffffIfI')  # 52 bytes per trajectory entry
        
        with open(traj_file_path, 'rb') as f:
            data = f.read()
        
        if len(data) < seed_struct.size:
            print(f"Warning: File {traj_file_path} is too small to contain trajectory data")
            return []
        
        # Parse trajectory entries and extract positions
        positions = []
        offset = seed_struct.size  # Skip the level seed
        
        while offset + traj_struct.size <= len(data):
            rec = traj_struct.unpack_from(data, offset)
            # Extract state_x, state_y, state_z (indices 4, 5, 6)
            state_x, state_y, state_z = rec[4], rec[5], rec[6]
            positions.append((state_x, state_y, state_z))
            offset += traj_struct.size
        
        if not positions:
            print(f"Warning: No trajectory entries found in {traj_file_path}")
            return []
        
        # Subsample num_waypoints positions evenly across the trajectory
        num_steps = len(positions)
        if num_steps < num_waypoints:
            print(f"Warning: Trajectory has only {num_steps} steps, using all as waypoints")
            indices = range(num_steps)
        else:
            # Evenly sample indices across the trajectory
            indices = np.linspace(0, num_steps - 1, num_waypoints, dtype=int)
        
        # Extract waypoints
        waypoints = [positions[idx] for idx in indices]
        
        return waypoints
    
    else:
        print(f"Warning: Unsupported file extension '{file_extension}' for trajectory file {traj_file_path}")
        print("Supported formats: .pt (PyTorch), .out/.bin (binary)")
        return []

# Inject waypoints from trajectory file into JSON level descriptions
def inject_trajectory_waypoints(json_levels: list[dict], traj_file_path, num_waypoints=5):
    """
    Load waypoints from a trajectory file and inject them into JSON level descriptions.
    
    Args:
        json_levels: List of JSON level dictionaries to modify
        traj_file_path: Path to the trajectory file
        num_waypoints: Number of waypoints to extract and inject (default 5)
    """
    # Load waypoints from trajectory
    waypoints = load_trajectory_waypoints(traj_file_path, num_waypoints)
    
    if not waypoints:
        print("Warning: No waypoints extracted from trajectory, skipping injection")
        return
    
    print(f"Injecting {len(waypoints)} waypoints from trajectory into levels")
    print(f"  Waypoint positions (Madrona coords): {waypoints}")
    
    # Convert waypoints from Madrona coordinates to DSL coordinates
    # This is needed because json_parts_to_tensor calls fix_coordinates() which
    # converts DSL -> Madrona. Since trajectory positions are already in Madrona,
    # we need to apply the inverse transform first.
    waypoints_dsl = [inverse_transform_coordinates(wp) for wp in waypoints]
    print(f"  Waypoint positions (DSL coords): {waypoints_dsl}")
    
    # Inject waypoints into each level
    for json_level in json_levels:
        for i, (x, y, z) in enumerate(waypoints_dsl):
            # Index waypoints from goal (index 0) to start (index n-1)
            # waypoints list goes from start to goal, so reverse the indexing
            waypoint_index = len(waypoints) - i
            
            waypoint_part = Part(
                part_type="Block",
                name=f"Waypoint_traj_{i}_{len(json_level)}",
                position=(x, y, z),
                size=(1, 1, 1),
                orientation=(0, 0, 0),
                behaviors=[
                    WaypointBehavior(index=waypoint_index, jumps=-1), 
                    NoCollideBehavior()
                ]
            )
            
            waypoint_name, waypoint_dict = waypoint_part.jsonify()
            json_level[waypoint_name] = waypoint_dict

# Move the goal (Checkpoint) of a json level to a random block.
def randomize_goal_blocks(json_level: dict):
    keys = json_level.keys()
    rand_goal_idx = UniformInt(0, 997, seed=0)
    goal_idx = rand_goal_idx.sample()
    for idx, k in enumerate(keys):
        d = json_level[k]
        if len(d["behaviors"]) > 0 and d["behaviors"][0]["name"] == "Checkpoint":
            d["behaviors"] = []
        if idx == (goal_idx % len(keys)):
            # Remove all previous behaviors and make the Checkpoint block.
            d["behaviors"] = [{"name": "Checkpoint", "conditions": []}]

# Move the spawn point of each json level to a random safe block.
def randomize_spawn_blocks(json_levels: list[dict], spawn_block_selector):
    def _randomize_spawn_blocks(json_level):
        keys = json_level.keys()

        safe_blocks = list(filter(lambda k: is_safe(json_level[k]), list(json_level.keys())))

        # Remove the original spawn point.
        for k in keys:
            d = json_level[k]
            if len(d["behaviors"]) > 0 and d["behaviors"][0]["name"] == "Spawn":
                d["behaviors"] = []

        spawn_block = spawn_block_selector.choice(safe_blocks)
        json_level[spawn_block]["behaviors"].insert(0, {"name": "Spawn", "conditions": []})

    for json_level in json_levels:
        _randomize_spawn_blocks(json_level)

# Select a specific spawn block by index for a collection of json levels.
def select_specific_spawn_block(json_levels: list[dict], spawn_block_index: int) -> int:
    """
    Select a specific spawn block by index for all levels.
    
    Args:
        json_levels: List of JSON level dictionaries to modify
        spawn_block_index: Desired spawn block index (will be clamped to valid range)
    
    Returns:
        The actual block index used (after clamping to valid range)
    """
    actual_index = None
    
    for json_level in json_levels:
        keys = json_level.keys()

        # Filter for safe blocks
        def is_safe(part):
            for behavior in part["behaviors"]:
                if behavior["name"] == "Damage" or behavior["name"] == "Checkpoint" or part["type"] == "Cylinder":
                    return False
            return True
        safe_blocks = list(filter(lambda k: is_safe(json_level[k]), list(json_level.keys())))
        
        # Keep safe blocks in their natural creation order (roughly distance from origin)
        # Don't sort alphabetically as it destroys the spatial ordering
        
        # Check if the requested index is valid and clamp to valid range
        block_index = spawn_block_index
        if block_index >= len(safe_blocks):
            print(f"  Warning: spawn-block-index {block_index} >= {len(safe_blocks)} safe blocks, using index {len(safe_blocks)-1}")
            block_index = len(safe_blocks) - 1
        elif block_index < 0:
            print(f"  Warning: spawn-block-index {block_index} < 0, using index 0")
            block_index = 0

        # Store the actual index used (should be same for all levels)
        if actual_index is None:
            actual_index = block_index

        # First get rid of the original spawn block
        for k in keys:
            d = json_level[k]
            if len(d["behaviors"]) > 0 and d["behaviors"][0]["name"] == "Spawn":
                d["behaviors"] = []

        # Select the specific spawn block by index
        selected_spawn_block = safe_blocks[block_index]
        print(f"  Selected spawn block {block_index}: {selected_spawn_block}")
        json_level[selected_spawn_block]["behaviors"].insert(0, {"name": "Spawn", "conditions": []})
    
    return actual_index if actual_index is not None else 0

# Apply a physical perturbation to a single block.
def _apply_perturbation_to_block(obj_desc: dict, rng: random.Random):
    """
    Apply random perturbations to a block's physical parameters.
    
    Args:
        obj_desc: The block/part dictionary to modify
        rng: Random number generator for reproducibility
    """
    obj_desc["position"] = [(x + 2 * rng.uniform(-1, 1)) for x in obj_desc["position"]]
    obj_desc["orientation"] = [(x + 10 * rng.uniform(-1, 1)) for x in obj_desc["orientation"]]
    obj_desc["size"] = [(x * (0.5 + rng.uniform(0, 1))) for x in obj_desc["size"]]


def select_specific_perturbation_block(json_levels: list[dict], perturbation_block_index: int, seed: int = 0) -> int:
    """
    Select and perturb a specific block by index for all levels.
    
    Args:
        json_levels: List of JSON level dictionaries to modify
        perturbation_block_index: Index of block to perturb (will be clamped to valid range)
        seed: Random seed for perturbations
    
    Returns:
        The actual block index used (after clamping to valid range)
    """
    rng = random.Random(seed)
    actual_index = None
    
    for json_level in json_levels:
        keys = json_level.keys()
        
        # Get all blocks in their natural creation order (roughly distance from origin)
        # Don't sort alphabetically as it destroys the spatial ordering
        # NOTE: This returns the blocks in insertion order, and may not line up
        # with the block index recorded during generation.
        all_blocks = list(keys)
        
        # Check if the requested index is valid and clamp to valid range
        block_index = perturbation_block_index
        if block_index >= len(all_blocks):
            print(f"  Warning: perturbation-block-index {block_index} >= {len(all_blocks)} total blocks, using index {len(all_blocks)-1}")
            block_index = len(all_blocks) - 1
        elif block_index < 0:
            print(f"  Warning: perturbation-block-index {block_index} < 0, using index 0")
            block_index = 0

        # Store the actual index used (should be same for all levels)
        if actual_index is None:
            actual_index = block_index

        # Select the specific block to perturbate
        selected_block = all_blocks[block_index]
        
        # Apply perturbation to only this block
        _apply_perturbation_to_block(json_level[selected_block], rng)
    
    return actual_index if actual_index is not None else 0


def perturb_all_part_positions(json_level: dict, seed: int = 0):
    """
    Apply random position perturbations to all blocks in a level.
    
    Args:
        json_level: JSON level dictionary to modify
        seed: Random seed for perturbations
    """
    rng = random.Random(seed)

    # Process each part
    for name in json_level.keys():
        _apply_perturbation_to_block(json_level[name], rng)

def transform_coordinates(v: list):
    """Convert from DSL convention to Madrona convention."""
    conv_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    return conv_mat.dot(v)

def inverse_transform_coordinates(v: tuple):
    """Convert from Madrona convention back to DSL convention.
    Madrona: (+X right, +Z up, +Y forward)
    DSL: (+X right, +Y up, -Z forward)
    Transform: (x, y, z) -> (x, z, -y)
    """
    x, y, z = v
    return (x, z, -y)

# Convert object dict with position/orientation/size, 
# and all dependent fields in behaviors, to the Madrona coordinate system.
def fix_coordinates(obj_desc: dict):
    """
    Convert from DSL convention (+X right, +Y up, -Z forward)
    to Madrona convention (+X right, +Z up, +Y forward)
    @param obj_desc: The part dictionary, converts in place
    """
    # Update all coordinate-dependent fields
    for field in ["position", "orientation", "size"]:
        if field in obj_desc:
            obj_desc[field] = transform_coordinates(obj_desc[field])
            if field == "size":
                obj_desc[field] = [abs(x) for x in obj_desc[field]]

    # Update all behavior-dependent fields
    for behavior in obj_desc["behaviors"]:
        for sub_behavior in ["start_position", "end_position", "velocity", "base_position", "base_orientation"]:
            if sub_behavior in behavior:
                behavior[sub_behavior] = transform_coordinates(behavior[sub_behavior])
    return

# Convert a single description of physics parameters to tensor.
def json_physics_to_tensor(
        json_level_physics: dict,
        physics_tensor_shape: tuple
):
    """
    Converts physics info (gravity, move force, jump force, air resistance)
    from a JSON dictionary to a tensor.
    """
    physics_tensor = torch.zeros(physics_tensor_shape)
    physics_tensor[0][0] = json_level_physics["gravity"]
    physics_tensor[0][1] = json_level_physics["move_force"]
    physics_tensor[0][2] = json_level_physics["jump_force"]
    physics_tensor[0][3] = json_level_physics["air_resistance"]
    physics_tensor[0][4:7] = torch.tensor(json_level_physics["spawn_position"])
    physics_tensor[0][7:10] = torch.tensor(json_level_physics["spawn_velocity"])
    return physics_tensor

def orientation_to_rot_mat(orientation):
    """
    Converts ZXY Euler angles (yaw, pitch, roll) to a 3x3 rotation matrix.

    Args:
        yaw (float): Rotation around the Z-axis (in radians).
        pitch (float): Rotation around the X-axis (in radians).
        roll (float): Rotation around the Y-axis (in radians).

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # Madrona X, Y, Z euler angles
    pitch, roll, yaw = [np.radians(angle) for angle in orientation]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    # Rotation matrix around Z-axis (Yaw)
    Rz = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    # Rotation matrix around X-axis (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, cos_pitch, -sin_pitch],
        [0, sin_pitch, cos_pitch]
    ])

    # Rotation matrix around Y-axis (Roll)
    Ry = np.array([
        [cos_roll, 0, sin_roll],
        [0, 1, 0],
        [-sin_roll, 0, cos_roll]
    ])


    # Combine rotations in ZXY order: R = Rz @ Ry @ Rx
    R = Rz @ Rx @ Ry
    return R

def part_bounds(pos, size, orientation):
    cx, cy, cz = pos
    sx, sy, sz = size

    corner_offsets = np.array([[-sx / 2, -sy / 2, -sz / 2],
                               [sx / 2, -sy / 2, -sz / 2],
                               [sx / 2, sy / 2, -sz / 2],
                               [-sx / 2, sy / 2, -sz / 2],
                               [-sx / 2, -sy / 2, sz / 2],
                               [sx / 2, -sy / 2, sz / 2],
                               [sx / 2, sy / 2, sz / 2],
                               [-sx / 2, sy / 2, sz / 2]])
    
    rotation_matrix = orientation_to_rot_mat(orientation)

    corner_offsets = np.dot(corner_offsets, rotation_matrix.T)

    corners = corner_offsets + np.array([cx, cy, cz])

    x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
    y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
    z_min, z_max = np.min(corners[:, 2]), np.max(corners[:, 2])

    return x_min, x_max, y_min, y_max, z_min, z_max


# Sort the parts by relevant state.
# For each part description, convert to tensor state that Madrona ingests.
def json_parts_to_xyz_bounds(json_level: dict):
    """
    Computes bounding box from all parts in a JSON level.
    """
    x_min, y_min, z_min = 9999, 9999, 9999
    x_max, y_max, z_max = -9999, -9999, -9999

    # Process each part
    for part_idx, name in enumerate(json_level.keys()):
        obj_desc = json_level[name]

        # Account for the swapped coordinate system
        fix_coordinates(obj_desc)

        px_min, px_max, py_min, py_max, pz_min, pz_max = part_bounds(
            obj_desc["position"],
            obj_desc["size"],
            obj_desc["orientation"]
        )

        x_min = min(px_min, x_min)
        x_max = max(px_max, x_max)
        y_min = min(py_min, y_min)
        y_max = max(py_max, y_max)
        z_min = min(pz_min, z_min)
        z_max = max(pz_max, z_max)

        # Factor in Rotate and Translate behaviors in bounds computation.
        for behavior_idx, behavior_dict in enumerate(obj_desc["behaviors"]):
            behavior_name = behavior_dict["name"]

            if "Rotate" in behavior_name: #"TouchRotate" or "PersistentRotate"
                size = obj_desc["size"]
                if behavior_dict["axis"] == 'right':
                    size[1] = max(size[1], size[2])
                    size[2] = max(size[1], size[2])
                elif behavior_dict["axis"] == 'up':
                    size[0] = max(size[0], size[1])
                    size[1] = max(size[0], size[1])
                elif behavior_dict["axis"] == 'forward':
                    size[0] = max(size[0], size[2])
                    size[2] = max(size[0], size[2])

                px_min, px_max, py_min, py_max, pz_min, pz_max = part_bounds(
                    obj_desc["position"],
                    size,
                    obj_desc["orientation"]
                )

                x_min = min(px_min, x_min)
                x_max = max(px_max, x_max)
                y_min = min(py_min, y_min)
                y_max = max(py_max, y_max)
                z_min = min(pz_min, z_min)
                z_max = max(pz_max, z_max)

            elif "Translate" in behavior_name:
                start_pos = behavior_dict["start_position"] + obj_desc["position"]
                end_pos = behavior_dict["end_position"]+ obj_desc["position"]
                px_min, px_max, py_min, py_max, pz_min, pz_max = part_bounds(start_pos, obj_desc["size"], obj_desc["orientation"])
                x_min = min(px_min, x_min)
                x_max = max(px_max, x_max)
                y_min = min(py_min, y_min)
                y_max = max(py_max, y_max)
                z_min = min(pz_min, z_min)
                z_max = max(pz_max, z_max)
                px_min, px_max, py_min, py_max, pz_min, pz_max = part_bounds(end_pos, obj_desc["size"], obj_desc["orientation"])
                x_min = min(px_min, x_min)
                x_max = max(px_max, x_max)
                y_min = min(py_min, y_min)
                y_max = max(py_max, y_max)
                z_min = min(pz_min, z_min)
                z_max = max(pz_max, z_max)
            elif "Trampoline" in behavior_name:
                print("WARNING: Bounds computation for trampoline behavior not implemented.")

    return x_min, x_max, y_min, y_max, z_min, z_max

# Sort the parts by relevant state.
# For each part description, convert to tensor state that Madrona ingests.
def json_parts_to_tensor(json_level: dict, part_tensor_shape: tuple):
    """
    Converts all the parts in a JSON level to a tensor.
     A single level has max 256 objects, 80 floats/object
    """

    global rand_goal_idx

    # Sort the parts so that the parts with relevant states are first
    keys = json_level.keys()
    keys = sorted(keys, key=lambda x: json_level[x]["state"]["name"] != "None", reverse=True)

    # Collect all the states, map the state name to a unique index
    state_to_state_id = {}
    for name in keys:
        state_name = json_level[name]["state"]["name"]
        if state_name not in state_to_state_id:
            state_to_state_id[state_name] = len(state_to_state_id)

    # Now we can create the part tensor
    part_tensor = torch.zeros(part_tensor_shape)

    # Process each part
    for part_idx, name in enumerate(keys):
        obj_desc = json_level[name]

        # Account for the swapped coordinate system
        fix_coordinates(obj_desc)

        # Add state information
        object_state = torch.zeros(2) # index, value
        object_state[0] = state_to_state_id[obj_desc["state"]["name"]]
        object_state[1] = obj_desc["state"]["value"]

        # Check the object type
        sim_object = SimObject.Cube
        if obj_desc["type"] == "Wedge":
            sim_object = SimObject.Wedge
        if obj_desc["type"] == "Ball":
            sim_object = SimObject.Sphere
        if obj_desc["type"] == "Cylinder":
            sim_object = SimObject.Cylinder

        # Process each behavior. Max 4 behaviors, 12 params and 4 conditions each
        behavior_parameters = torch.zeros(4, 12)
        behavior_conditions = -torch.ones(4, 4)
        behavior_types = -torch.ones(4, 1)
        for behavior_idx, behavior_dict in enumerate(obj_desc["behaviors"]):
            # Type of behavior
            obj_behavior = BehaviorType.NONE

            # Condition(s) for this behavior
            for c_idx, cond in enumerate(behavior_dict["conditions"]):
                assert cond in state_to_state_id, \
                    f"Condition {cond} not found in states."
                behavior_conditions[behavior_idx][c_idx] = state_to_state_id[cond]

            # Handle the parameters and type
            behavior_name = behavior_dict["name"]
            if "Spawn" in behavior_name:
                obj_behavior = BehaviorType.SPAWN
            elif "Goal" in behavior_name or "Checkpoint" in behavior_name:
                obj_behavior = BehaviorType.CHECKPOINT
            elif "Damage" in behavior_name:
                obj_behavior = BehaviorType.DAMAGE
                behavior_parameters[behavior_idx][0] = behavior_dict["damage"]
            elif "Rotate" in behavior_name: #"TouchRotate" or "PersistentRotate"
                obj_behavior = BehaviorType.ROTATE
                behavior_parameters[behavior_idx][0] = behavior_dict["angular"]
                if behavior_dict["axis"] == 'right':
                    behavior_parameters[behavior_idx][1] = 0.0
                elif behavior_dict["axis"] == 'up':
                    behavior_parameters[behavior_idx][1] = 1.0
                elif behavior_dict["axis"] == 'forward':
                    behavior_parameters[behavior_idx][1] = 2.0
                else:
                    raise ValueError("Invalid rotation axis:", behavior_dict["axis"])
            elif "Translate" in behavior_name: #"TouchTranslate" or "PersistentTranslate"
                obj_behavior = BehaviorType.MOTION
                if "base_position" in behavior_dict:
                    pos_offset = behavior_dict["base_position"]
                    rotation_matrix = orientation_to_rot_mat(behavior_dict["base_orientation"])
                else:
                    pos_offset = obj_desc["position"]
                    rotation_matrix = orientation_to_rot_mat([0,0,0])
                start_pos = np.dot(np.array(behavior_dict["start_position"]), rotation_matrix.T) + np.array(pos_offset)
                end_pos = np.dot(np.array(behavior_dict["end_position"]), rotation_matrix.T) + np.array(pos_offset)

                behavior_parameters[behavior_idx][:3] = torch.tensor(start_pos)
                behavior_parameters[behavior_idx][3:6] = torch.tensor(end_pos)
                behavior_parameters[behavior_idx][6] = behavior_dict["time"]
                behavior_parameters[behavior_idx][7] = EasingStyle.Linear if behavior_dict["easing_style"] == "Linear" else -1.0
                behavior_parameters[behavior_idx][8] = EasingDirection.Out if behavior_dict["easing_direction"] == "Out" else -1.0
                behavior_parameters[behavior_idx][9] = 1.0 if behavior_dict["reverses"] else 0.0
                behavior_parameters[behavior_idx][10] = behavior_dict["loop"]
                behavior_parameters[behavior_idx][11] = behavior_dict["direction"] if "direction" in behavior_dict else 1
            elif "Conveyor" in behavior_name:
                obj_behavior = BehaviorType.CONVEYOR
                rotation_matrix = orientation_to_rot_mat(obj_desc["orientation"])
                rotated_velocity = np.dot(np.array(behavior_dict["velocity"]), rotation_matrix.T)
                behavior_parameters[behavior_idx][:3] = torch.tensor(rotated_velocity)
            elif "Trampoline" in behavior_name:
                obj_behavior = BehaviorType.TRAMPOLINE
                rotation_matrix = orientation_to_rot_mat(obj_desc["orientation"])
                rotated_velocity = np.dot(np.array(behavior_dict["velocity"]), rotation_matrix.T)
                behavior_parameters[behavior_idx][:3] = torch.tensor(rotated_velocity)
            elif "Truss" in behavior_name:
                obj_behavior = BehaviorType.TRUSS
            elif "PressState" in behavior_name:
                obj_behavior = BehaviorType.PRESS_STATE
            elif "HoldState" in behavior_name:
                obj_behavior = BehaviorType.HOLD_STATE
            elif "Waypoint" in behavior_name:
                obj_behavior = BehaviorType.WAYPOINT
                behavior_parameters[behavior_idx][0] = behavior_dict["index"]
                behavior_parameters[behavior_idx][1] = behavior_dict["jumps"] if "jumps" in behavior_dict.keys() else -1
            elif "Avoid" in behavior_name:
                # Avoid can be implemented in Madrona as Damage.
                obj_behavior = BehaviorType.DAMAGE
                behavior_parameters[behavior_idx][0] = 100
            elif "NoCollide" in behavior_name:
                obj_behavior = BehaviorType.NOCOLLIDE
            elif "Carry" in behavior_name:
                pass  # Default behavior of the Kinematic response type (see level_gen.cpp).
            elif "Scale" in behavior_name: #"TouchScale" or "PersistentScale"
                print("NOT IMPLEMENTED")
                exit(-1)
            elif "Friction" in behavior_name:
                if sim_object == SimObject.Wall: sim_object = SimObject.IceWall
                if sim_object == SimObject.Wedge: sim_object = SimObject.IceWedge
            elif "Unanchored" in behavior_name:
                obj_behavior = BehaviorType.UNANCHORED
                sim_object = SimObject.Cube

            behavior_types[behavior_idx] = int(obj_behavior)

        # Combine these behaviors into one tensor
        object_behavior = torch.hstack([behavior_types, behavior_conditions, behavior_parameters])

        # Fill in the part tensor
        part_tensor[part_idx][:3] = torch.tensor(obj_desc["position"])
        part_tensor[part_idx][3:6] = torch.tensor(obj_desc["size"])
        part_tensor[part_idx][6:9] = torch.tensor(obj_desc["orientation"])
        part_tensor[part_idx][9] = sim_object
        part_tensor[part_idx][10:12] = object_state
        part_tensor[part_idx][12:] = object_behavior.flatten()
    return part_tensor

def json_traj_to_tensor(episode, traj_tensor_shape: tuple, level_seed: int = 0):
    """
    Converts trajectory info from a json file containing dict of episodes into a tensor.
    """
    traj_tensor = torch.zeros(traj_tensor_shape, dtype=torch.float32)

    # Checks for directory or file
    if episode is None:
        return traj_tensor

    n_samples = 25 # can only store 25 samples due to storage constraints

    if episode:
        # pick n_samples from the episode
        pick_idxs = np.linspace(0, len(episode)-1, n_samples, dtype=int)
        last_idx = 0
        for out_i, src_i in enumerate(pick_idxs):
            
            entry: dict = episode[src_i]
            obs = entry["observations"]
            
            # write 3 observation values per sample into slots of size 3
            traj_tensor[0][3*out_i + 0] = obs["state_x"]
            traj_tensor[0][3*out_i + 1] = obs["state_y"]
            traj_tensor[0][3*out_i + 2] = obs["state_z"]
            last_idx += 3 * out_i + 2

        # metadata at the end
        traj_tensor[0][75] = 0 # set world idx to 0 for now
        traj_tensor[0][76] = level_seed
        traj_tensor[0][77] = 1  # exists
        
    return traj_tensor

# Merges two lists of json parts into a single JSON description.
# Uses the transformation of the first to orient the second etc.
def merge_json_parts(levels_1: list[str], levels_2: list[str]):
    """
    Given a list of JSON parts for two levels, merge them into one.
    levels_2 merges into levels_1, renaming any objects that conflict.
    """
    # Empty to start
    if len(levels_1) == 0:
        return levels_2
    # Combine the two lists of levels
    assert len(levels_1) == len(levels_2)
    merged_levels = []
    for parts_1, parts_2 in zip(levels_1, levels_2):
        json_1 = json.loads(parts_1)
        json_2 = json.loads(parts_2)
        # Find the checkpoint of json_1, convert to block
        checkpoints = [obj for obj in json_1 if json_1[obj]["class"] == "Checkpoint"]
        last_checkpoint = checkpoints[-1]
        json_1[last_checkpoint]["class"] = "Block"
        json_1[last_checkpoint]["type"] = "Block"
        json_1[last_checkpoint]["behaviors"] = []
        checkpoint_pos = json_1[last_checkpoint]["position"]
        checkpoint_size = json_1[last_checkpoint]["size"]
        # Get the largest index of the first level
        max_idx = max([json_1[obj]["idx"] for obj in json_1])

        # Place the second level at the first level's last checkpoint.
        transform = checkpoint_pos
        to_rename = []
        for obj in json_2:
            # Convert spawn points into blocks
            if json_2[obj]["class"] == "SpawnLocation":
                json_2[obj]["class"] = "Block"
                json_2[obj]["type"] = "Block"
                json_2[obj]["behaviors"] = []
            # Update position and index
            json_2[obj]["position"][0] += transform[0]
            json_2[obj]["position"][1] += transform[1]
            json_2[obj]["position"][2] += transform[2]
            json_2[obj]["idx"] += max_idx
            # Rename clashing names by appending a number until they are unique.
            if obj in json_1:
                new_name_idx = 2
                new_name = f"{obj}-{new_name_idx}"
                while new_name in json_1:
                    new_name_idx += 1
                    new_name = f"{obj}-{new_name_idx}"
                to_rename.append((obj, new_name))

        # Now perform the renaming
        for old_name, new_name in to_rename:
            json_2[new_name] = json_2.pop(old_name)
        # Sort by index
        json_merged = {**json_1, **json_2}
        json_merged = dict(sorted(json_merged.items(), key=lambda item: item[1]["idx"]))
        merged_levels.append(json.dumps(json_merged))

    return merged_levels

# Generates a JSON-style dict for each of num_levels json description slots.
# Calls the DSL functions directly, does not interact with the filesystems or an existing json string.
def generate_json_levels(
        num_levels: int,
        level_seed: int,
        level_types: list,
        level_selector,
        level_generation_params: TrainParameters1,
        rotation_distribution: Distribution
):  
    if level_types is None:
        raise ValueError("Level types must be specified.")
    # If no selector provided, fall back to random based on seed
    if level_selector is None:
        from random import Random
        level_selector = Random(level_seed)
    
    generators = level_generation_params.get_generators()

    json_levels_parts = ["" for i in range(num_levels)]

    # Needed for deterministic level generation.
    level_types = sorted(level_types)

    for i in range(num_levels):
        obbyName = level_selector.choice(level_types)
        obbyDist = generators[obbyName]
        obbyClassName = f"random_{obbyName}"
        obbyGeneratorName = f"Random{obbyName}"

        rotation_angle = rotation_distribution.sample()
        orientation = (0, rotation_angle, 0)
        start_pos = (1.5 * math.sin(rotation_angle * math.pi / 180), 0, 1.5 * math.cos(rotation_angle * math.pi / 180))

        # Create the obby object and the spawn.
        obby = Obby(obbyClassName + f"_s{level_seed}_l{i}", unit=4)
        curr_obby_parts = obbyDist.sample(obbyGeneratorName + f"_s{level_seed}", "Checkpoint 0", start_pos, orientation)
        obby.spawn((0,0,0), 3, orientation)
        obby.append_parts(curr_obby_parts.generate())

        json_levels_parts[i] = obby.jsonify()

    return json_levels_parts

# Builds the json descriptions tensor (physics, expert trajectory and parts per
# world) that can be copied directly into the simulator's tensor. Uses default
# physics when none is given.
def jsonDescriptionsToTensor(json_levels_shape, json_levels_parts, level_seed=None, json_levels_physics=None, json_levels_traj=None):
    n_worlds, n_objs, n_floats = json_levels_shape
    json_levels_tensor = torch.zeros((n_worlds, n_objs, n_floats))
    
    if not json_levels_physics:
        # Default physics for every world.
        json_levels_physics = [Physics().jsonify() for _ in range(n_worlds)]
        
    if json_levels_traj is not None:
        _, file_extension = os.path.splitext(json_levels_traj)
        
        if file_extension.lower() == '.json':
            # Handle JSON file format
            with open(json_levels_traj, 'r') as f:
                expert_demos = json.load(f)
        elif file_extension.lower() == '.pt':
            # .pt file of successful trajectories (go_explore_search_solvable.py --store-successful-trajs)
            traj_data = torch.load(json_levels_traj, map_location='cpu')
            trajectories = traj_data.get("trajectories", [])
            
            if trajectories:
                # Use the first trajectory and convert to expected format
                traj = trajectories[0]  # Shape: [num_steps, 3] where 3 is (x, y, z)
                entries = []
                for i in range(traj.shape[0]):
                    entry = {
                        "observations": {
                            "state_x": float(traj[i, 0]),
                            "state_y": float(traj[i, 1]),
                            "state_z": float(traj[i, 2]),
                        }
                    }
                    entries.append(entry)
                expert_demos = [entries]
            else:
                print(f"Warning: No trajectories found in {json_levels_traj}, using empty trajectory")
                expert_demos = [None]
        elif file_extension.lower() in ['.out', '.bin']:
            # Handle binary .out/.bin file format
            # Binary format:
            # - First 4 bytes: level seed (int32)
            # - Followed by Trajectory structs (52 bytes each)
            seed_struct = struct.Struct('<i')  # 4 bytes for level seed
            traj_struct = struct.Struct('<iiiiffffffIfI')  # 52 bytes per trajectory entry
            
            with open(json_levels_traj, 'rb') as f:
                data = f.read()
            
            # Parse trajectory entries into the format expected by json_traj_to_tensor
            entries = []
            offset = seed_struct.size  # Skip the level seed
            
            while offset + traj_struct.size <= len(data):
                rec = traj_struct.unpack_from(data, offset)
                entry = {
                    "observations": {
                        "state_x": rec[4],
                        "state_y": rec[5],
                        "state_z": rec[6],
                    }
                }
                entries.append(entry)
                offset += traj_struct.size
            
            # Wrap entries in a list (expert_demos is expected to be a list of episodes)
            expert_demos = [entries] if entries else [None]
        else:
            print(f"Warning: Unsupported trajectory file extension '{file_extension}', using empty trajectory")
            expert_demos = [None]
    else:
        expert_demos = [None]

    # Every world gets the same expert demo.
    for i in range(n_worlds):
        json_parts_tensor = json_parts_to_tensor(json.loads(json_levels_parts[i]), (n_objs - 2, n_floats))
        json_physics_tensor = json_physics_to_tensor(json.loads(json_levels_physics[i]), (1, n_floats))
        json_traj_tensor = json_traj_to_tensor(expert_demos[0], (1, n_floats), level_seed=level_seed)
        json_levels_tensor[i] = torch.cat((json_physics_tensor, json_traj_tensor, json_parts_tensor), dim=0)
        
    return json_levels_tensor

# Loads a json description file, or every .json file in a folder (recursive:
# also in subfolders). Returns {file path: json string}.
def loadJsonDescriptionsFromDisk(filename: str, recursive=False):
    files_to_json_strings = {}
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            files_to_json_strings[os.path.join(os.getcwd(), filename)] = f.read()
    else:
        for root, dirs, files in os.walk(filename):
            if recursive or root == filename:
                for f_path in [f if f.endswith(".json") else None for f in files]:
                    full_path = os.path.join(root, f_path)
                    with open(full_path, 'r') as f:
                        files_to_json_strings[full_path] = f.read()

    # Files saved with metadata keep the level under "level_json".
    for f, j in files_to_json_strings.items():
        if "level_json" in j:
            files_to_json_strings[f] = json.dumps(json.loads(j)["level_json"])

    return files_to_json_strings

# Generates components_per_level components per level with generate_json_levels
# and merges them in sequence with merge_json_parts.
def generateJsonLevels(
        num_levels, 
        level_seed, 
        level_types, 
        level_selector, 
        components_per_level, 
        level_generation_params,
        rotation_distribution=None
    ):

    # Default rotation: fully random.
    if not rotation_distribution:
        rotation_distribution = Uniform(0, 360, seed=level_seed)
    
    merged_json_parts = []
    for _ in range(components_per_level):
        json_levels_parts = generate_json_levels(
            num_levels,
            level_seed, 
            level_types,
            level_selector,
            level_generation_params,
            rotation_distribution) # Repeatable for a given seed.
        merged_json_parts = merge_json_parts(merged_json_parts, json_levels_parts)

    return merged_json_parts

# Generates levels with generateJsonLevels and returns them as the simulator's
# json descriptions tensor.
def generateJsonLevelsTensor(
        json_levels_shape, 
        level_seed: int, 
        level_type: list[str],
        level_selector,
        level_generation_params: TrainParameters1,
        components_per_level: int, 
        expert_traj: str = None,
    ):
    merged_json_parts = generateJsonLevels(
        json_levels_shape[0], level_seed, level_type, level_selector, components_per_level, level_generation_params
    )
    
    return jsonDescriptionsToTensor(json_levels_shape, merged_json_parts, level_seed, json_levels_traj=expert_traj)


if __name__ == '__main__':
    # Example: load one file, then a directory.
    print(loadJsonDescriptionsFromDisk("EasyObbySections/Section1.json"))

    easyObbySections = loadJsonDescriptionsFromDisk("EasyObbySections", True)
    # The caller sorts the keys; here by section number.
    for key in sorted(easyObbySections.keys(), key = lambda a: int(a[a.rindex("Section") + len("Section"):a.index(".json")])):
        print(key)
