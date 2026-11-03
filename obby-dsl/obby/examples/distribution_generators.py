import argparse
import collections

# import obby.grammar.behaviors as behaviors
from obby.grammar.part import *
from obby.grammar.distribution import *
from obby.grammar.obstacles import *
from obby.grammar.obby import *
# from obby.grammar.utils import *
# from obby.difficulty.score import GeneratorScoringSystem

parser = argparse.ArgumentParser()
parser.add_argument(
    '--example', type=str, 
    help='The example to generate.'
)
parser.add_argument(
    '--unit', '-u', type=float, default=4, 
    help='Unit of grid cell.'
)
parser.add_argument(
    '--seed', '-s', type=int, default=[0], nargs='+',
    help='Random seed.'
)
parser.add_argument(
    '--n_levels', '-n', type=int, default=1, 
    help='Number of levels to generate.'
)
parser.add_argument(
    '--solve', action='store_true', default=False, 
    help='Generate a solution if true.'
)
parser.add_argument(
    '--randomize', action='store_true', default=False, 
    help='Randomize the geometry if true.'
)
parser.add_argument(
    '--json-path', type=str, default="ObbyTest/JSON/",
    help="Folder at which to save the JSON description file."
)


# def generate_random_part(part_type: Distribution, part_width: Distribution, part_length: Distribution) -> Part:
#     part_type_sample = part_type.sample()
#     if part_type_sample == Ball:
#         radius = part_width.sampe()
#         part_size = (radius, radius, radius)
#     elif part_type_sample == Cylinder:
#         radius = part_width.sampe()
#         height = part_length.sample()
#         part_size = (radius, height, radius)
#     else:
#         part_size = (part_width.sample(), PLATFORM_Y, part_length.sample())
    
#     return part_type_sample(
#         f"{part_type_sample.part_type}",
#         (0, 0, 0),
#         part_size
#     )


def random_obby(
    name: str,
    generators: dict[str, ObstacleDistribution],
    n_sections: Distribution,
    obstacle_type: Distribution,
    unit: float = 1,
    return_solution: bool = False
) -> Obby:
    obby = Obby(name, unit=unit)
    n_sections_sample = n_sections.sample()

    # Spawn.
    solution = Path(start=(0, 0, 0))
    obby.spawn((0, 0, 0), 3)
    n_generators = len(generators)
    generator_names = list(generators.keys())
    generator_objects = list(generators.values())

    prev_generator = None
    obstacle_position = (0, 0, 1.5)
    orientation = (0, 0, 0)
    directions = 'f' * n_sections_sample #"fff" + generate_pretty_directions(n_sections_sample - 3, weights=(0.15, 0.7, 0.15))
    for i in range(n_sections_sample):
        # if "Tightrope" in str(prev_generator):
        #     # Make every few obstacle a truss to reduce the probability of collisions.
        #     orientation = (0, obby.parts[-1].orientation[1] + 90, 0)
        #     last_segment = obby.parts[-1]
        #     obstacle_position = get_position_after(last_segment, orientation[1], gap=0)
        #     if return_solution:
        #         obstacle_parts, waypoints = generators["Truss"].sample(
        #             name=f"Section{i}_Truss",
        #             checkpoint_name=f"Checkpoint{i + 1}",
        #             position=obstacle_position,
        #             orientation=orientation
        #         ).generate(return_solution=True)
        #         solution.append_waypoints(waypoints)
        #     else:
        #         obstacle_parts = generators["Truss"].sample(
        #             name=f"Section{i}_Truss",
        #             checkpoint_name=f"Checkpoint{i + 1}",
        #             position=obstacle_position,
        #             orientation=orientation
        #         ).generate()
        #     obstacle_parts[-1].name = f"Checkpoint{i + 1}"
        #     obby.append_parts(obstacle_parts)
        #     prev_generator = "Truss"
        # else:
            # Place a random obstacle.
            rand = i / n_sections_sample #obstacle_type.sample()
            for n in range(n_generators):
                if rand >= n / n_generators and rand < (n + 1) / n_generators:
                    if directions[i] == 'r':
                        orientation = vadd(obby.parts[-1].orientation, (0, -90, 0))
                    elif directions[i] == 'l':
                        orientation = vadd(obby.parts[-1].orientation, (0, 90, 0))

                    last_segment = obby.parts[-1]
                    obstacle_position = get_position_after(last_segment, orientation[1], gap=0)
                    if return_solution:
                        obstacle_parts, waypoints = generator_objects[n].sample(
                            name=f"Section{i}_{generator_names[n]}",
                            checkpoint_name=f"Checkpoint{i + 1}",
                            position=obstacle_position,
                            orientation=orientation
                        ).generate(return_solution=True)
                        solution.append_waypoints(waypoints)
                    else:
                        print(generator_objects[n])
                        obstacle_parts = generator_objects[n].sample(
                            name=f"Section{i}_{generator_names[n]}",
                            checkpoint_name=f"Checkpoint{i + 1}",
                            position=obstacle_position,
                            orientation=orientation
                        ).generate()
                    obstacle_parts[-1].name = f"Checkpoint{i + 1}"
                    obby.append_parts(obstacle_parts)
                    prev_generator = generator_names[n]
    
    if return_solution:
        solution.end = solution.waypoints[-1].position
        return obby, solution
    else:
        return obby


def random_sorted_obby(
    name: str,
    generators: dict[str, tp.Callable],
    n_sections: Distribution,
    obstacle_type: Distribution,
    unit: float,
) -> Obby:
    
    n_generators = len(generators)
    generator_names = list(generators.keys())
    generator_objects = list(generators.values())

    # First generate all the part groups.
    n_sections_sample = n_sections.sample()
    obstacles_by_score = collections.defaultdict(list)
    i = 0
    while i < n_sections_sample:
        # Place a random obstacle.
        rand = obstacle_type.sample()
        for n in range(n_generators):
            if rand >= n / n_generators and rand < (n + 1) / n_generators:
                # generator = generator_methods[n]
                generator_name = generator_names[n]
                obstacle = generator_objects[n].sample(
                    name=generator_name, 
                    position=(0, 0, 0), 
                    orientation=0,
                    checkpoint_name=f"Checkpoint{i + 1}", 
                )
                difficulty_score = obstacle.score(unit=unit)
                obstacles_by_score[difficulty_score].append(obstacle)
                i += 1

                if "Tightrope" in str(type(obstacle)) and i < n_sections_sample - 1:
                    i += 1

    # Repeatedly connect obstacles in order of increasing difficulty.
    obby = Obby(name, unit=unit)

    # Spawn.
    obby.spawn((0, 0, 0), 3)

    obstacle_position = (0, 0, 1.5)
    orientation = 0
    directions = "fff" + generate_pretty_directions(n_sections_sample - 3, weights=(0.15, 0.7, 0.15))
    idx = 0
    for i, score in enumerate(sorted(list(obstacles_by_score.keys()))):
        print("Score:", score)
        obstacles = obstacles_by_score[score]
        for obstacle in obstacles:
            last_segment = obby.parts[-1]
            if directions[i] == 'r':
                orientation = last_segment.orientation[1] - 90
            elif directions[i] == 'l':
                orientation = last_segment.orientation[1] + 90
            
            obstacle.name = f"Section{idx}_{type(obstacle)}"
            obstacle.position = get_position_after(last_segment, orientation, gap=0)
            obstacle.orientation = orientation
            obstacle.checkpoint_name = f"Checkpoint{idx + 1}"
            obstacle_parts = obstacle.generate()
            
            obby.append_parts(obstacle_parts)
            idx += 1
            
            last_segment = obby.parts[-1]
            obstacle_position = get_position_after(last_segment, orientation, gap=0)

            # Add a truss after tightropes to reduce chances of self-intersection.
            if "Tightrope" in str(type(obstacle)) and i < n_sections_sample - 1:
                generator_name = "random_truss"
                last_segment = obby.parts[-1]
                orientation = last_segment.orientation[1] + 90
                obstacle_position = get_position_after(last_segment, orientation, gap=0)
                obstacle = generators["Truss"].sample(
                    name=f"Section{idx}_{generator_name}", 
                    position=obstacle_position, 
                    orientation=orientation,
                    checkpoint_name=f"Checkpoint{idx + 1}", 
                )
                obstacle_parts = obstacle.generate()
                obby.append_parts(obstacle_parts)
                idx += 1

    return obby


def main():
    args = parser.parse_args()

    # Generation.
    for seed in args.seed:
        # # Set up some default distributions.
        # all_part_types = UniformChoice([Block, Cylinder, Wedge, Ball, Lava], seed=seed)
        # safe_part_types = UniformChoice([Block], seed=seed)
        # uniform_sample = Uniform(0, 1, seed=seed)
        # n_sections = UniformInt(10, 10, seed=seed)
        # a_few = UniformInt(1, 3, seed=seed)
        # n_parts_low = UniformInt(2, 4, seed=seed)
        # n_parts_high = UniformInt(4, 8, seed=seed)
        # xy_gap_length = Uniform(0.1, 2.5, seed=seed)
        # z_gap_length = Uniform(0.5, 1.0, seed=seed)
        # part_length_short = Uniform(1.0, 4.0, seed=seed)
        # part_length_long = Uniform(4.0, 12.0, seed=seed)
        # tightrope_width = Uniform(0.1, 1.0, seed=seed)
        # part_width = Uniform(1.0, 5.0, seed=seed)
        # platform_height = Uniform(0.1, 0.5, seed=seed)
        # checkerboard_size = Uniform(0.5, 1.0, seed=seed)
        # part_height = Uniform(0.5, 0.75, seed=seed)
        # block_height = Uniform(0.5, 2.0, seed=seed)
        # amplitude = Uniform(-6, 6, seed=seed)
        # timing = Uniform(1, 10, seed=seed)
        # velocity = Uniform(0.5, 5, seed=seed)
        # rotate = Uniform(0.5, 5, seed=seed)
        # binary = Binary(seed=seed)
        # bounce = Uniform(10, 20, seed=seed)
        # height_increment = Uniform(0.3, 0.75, seed=seed)
        # starting_orientation = Uniform(0.0, 360.0, seed=seed)
        # size_jitter = Gaussian(0.0, 0.05, seed=seed)
        # orientation_jitter = Gaussian(0.0, 5, seed=seed)
        # position_jitter = Gaussian(0.0, 0.1, seed=seed)
        # random_safe_part = RandomPart(
        #     safe_part_types, part_width, platform_height, z_gap_length,
        #     orientation_jitter, orientation_jitter, orientation_jitter
        # )

        # generators = {
        #     "PathJumps": PathJumpsDistribution(
        #         n_parts_low,
        #         xy_gap_length, 
        #         part_length_short,
        #         part_width,
        #         binary
        #     ),
        #     "ThickPathJumps": ThickPathJumpsDistribution(
        #         n_parts_low,
        #         xy_gap_length, 
        #         part_length_short,
        #         part_width,
        #         block_height,
        #         binary
        #     ),
        #     "GeneralPathJumps": GeneralPathJumpsDistribution(
        #         n_parts_low,
        #         xy_gap_length, 
        #         part_length_long,
        #         Constant(False)
        #     ),
        #     "Lilypads": LilypadsDistribution(
        #         safe_part_types, 
        #         n_parts_high, 
        #         xy_gap_length
        #     ),
        #     "ShrinkingLilypads": ShrinkingLilypadsDistribution(
        #         safe_part_types, 
        #         n_parts_high, 
        #         xy_gap_length,
        #         timing
        #     ),
        #     "Tightrope": TightropeDistribution(
        #         n_parts_high,
        #         binary,
        #         tightrope_width,
        #         part_length_long
        #     ),
        #     "WalledTightrope": WalledTightropeDistribution(
        #         n_parts_high,
        #         binary,
        #         tightrope_width,
        #         part_length_long
        #     ),
        #     "SafetyNetTightrope": SafetyNetTightropeDistribution(
        #         n_parts_high,
        #         binary,
        #         tightrope_width,
        #         part_length_long
        #     ),
        #     "LavaTightrope": LavaTightropeDistribution(
        #         n_parts_high,
        #         binary,
        #         tightrope_width,
        #         part_length_long
        #     ),
        #     "IceTightrope": IceTightropeDistribution(
        #         n_parts_high,
        #         binary,
        #         tightrope_width,
        #         part_length_long,
        #         Constant(0),
        #         Uniform(20, 50)
        #     ),
        #     "ConveyorTightrope": ConveyorTightropeDistribution(
        #         n_parts_high,
        #         binary,
        #         tightrope_width,
        #         part_length_long,
        #         velocity
        #     ),
        #     "ABAB": ABABDistribution(
        #         n_parts_high,
        #         all_part_types,
        #         all_part_types,
        #         xy_gap_length
        #     ),
        #     "Stairs": StairsDistribution(
        #         random_safe_part,
        #         n_parts_high,
        #         z_gap_length,
        #     ),
        #     "DisappearingStairs": DisappearingStairsDistribution(
        #         random_safe_part,
        #         n_parts_high,
        #         z_gap_length,
        #         timing,
        #         timing
        #     ),
        #     "LavaJumps": LavaJumpsDistribution(
        #         n_parts_low,
        #         part_length_long,
        #         part_height
        #     ),
        #     "IceJumps": IceJumpsDistribution(
        #         n_parts_low,
        #         xy_gap_length,
        #         part_length_short, 
        #         z_gap_length, 
        #         Constant(0),
        #         Uniform(20, 50),
        #         binary
        #     ),
        #     "LavaSpinner": LavaSpinnerDistribution(
        #         part_length_long,
        #         z_gap_length,
        #         part_length_long,
        #         rotate,
        #         starting_orientation
        #     ),
        #     "Trampoline": TrampolineDistribution(
        #         part_width,
        #         part_length_long,
        #         bounce,
        #     ),
        #     "TrampolineWalls": TrampolineWallsDistribution(
        #         n_parts_low,
        #         part_width,
        #         part_length_long,
        #         bounce,
        #         height_increment,
        #         xy_gap_length
        #     ),
        #     "LavaTrampoline": LavaTrampolineDistribution(
        #         part_width,
        #         part_length_long,
        #         bounce,
        #         a_few
        #     ),
        #     "WallJumps": WallJumpsDistribution(
        #         n_parts_low
        #     ),
        #     "Truss": TrussDistribution(
        #         part_length_long
        #     ),
        #     "Drop": DropDistribution(
        #         part_length_long
        #     ),
        #     "LavaCheckerboard": LavaCheckerboardDistribution(
        #         n_parts_high,
        #         checkerboard_size
        #     ),
        #     "Wedges": WedgesDistribution(
        #         n_parts_low,
        #         part_length_short,
        #         part_height
        #     ),
        #     "LavaWedges": LavaWedgesDistribution(
        #         n_parts_low,
        #         part_length_short,
        #         checkerboard_size
        #     ),
        #     "SideToSideBlocks": SideToSideBlocksDistribution(
        #         n_parts_low,
        #         velocity,
        #         xy_gap_length,
        #         amplitude,
        #     ),
        #     "HiddenStairs": HiddenStairsDistribution(
        #         n_parts_high,
        #         uniform_sample,
        #         part_length_short
        #     ),
        #     "BlockPush": BlockPushDistribution(
        #         part_width,
        #         part_length_long,
        #         block_height,
        #         uniform_sample,
        #     ),
        #     "ButtonHoldSpinner": ButtonHoldSpinnerDistribution(
        #         part_width, 
        #         part_length_long, 
        #         starting_orientation,
        #         rotate
        #     ),
        #     "OneButtonPressWall": OneButtonPressWallDistribution(
        #         part_width,
        #         part_length_short,
        #         uniform_sample
        #     ),
        #     "DextrousButtonPressWall": DextrousButtonPressWallDistribution(
        #         part_width,
        #         part_length_short,
        #         uniform_sample
        #     ),
        #     "PathVsLilypads": PathVsLilypadsDistribution(
        #         n_parts_high,
        #         xy_gap_length
        #     ),
        #     "ConveyorVsLilypads": ConveyorVsLilypadsDistribution(
        #         n_parts_high,
        #         xy_gap_length,
        #         velocity
        #     ),
        #     "Plank": PlankDistribution(
        #         part_width,
        #         part_length_long
        #     )
        # }

        level_seed = 0

        n_blocks = UniformInt(2, 6, seed=level_seed)  # one or two gaps #was 4, 6 for LIDAR ablation.
        gap_lengths = Uniform(0, 3, seed=level_seed)  # actual challenge
        thick_path_jump_gap_lengths = Uniform(0, 3, seed=level_seed)  # actual challenge
        #thick_path_jump_gap_lengths = Uniform(0.488, 3, seed=level_seed)  # actual challenge
        block_lengths = Uniform(0.5, 3.0, seed=level_seed)
        block_thicknesses = Uniform(0.1, 5.0, seed=level_seed)
        truss_thicknesses = Uniform(1.0, 2.0, seed=level_seed)
        n_walls = UniformInt(1, 1, seed=level_seed)

        # Distributions for levels
        all_part_types = UniformChoice([Block, Cylinder, Wedge, Ball, Lava], seed=level_seed)
        safe_part_types = UniformChoice([Block], seed=level_seed)
        obstacle_type = Uniform(0, 1, seed=level_seed)
        n_sections = UniformInt(7, 7, seed=level_seed)
        a_few = UniformInt(1, 3, seed=level_seed)

        n_parts_range = UniformInt(2, 8, seed=level_seed)
        n_parts_low = UniformInt(2, 4, seed=level_seed)
        n_silos = UniformInt(1, 8, seed=level_seed)
        n_parts_low_wall = UniformInt(1, 5, seed=level_seed) # TODO: restore
        wall_jump_wall_depth = Uniform(0.1, 0.5, seed=level_seed) # original was 0.25
        n_parts_high = UniformInt(5, 8, seed=level_seed) # Lava Check # TODO: Restore
        checkerboard_size = Uniform(0.5, 3.0, seed=level_seed) # Lava Check

        n_truss_hurdles_wide = UniformInt(1, 3, seed=level_seed)

        #xy_gap_length = Uniform(0.5, 4, seed=level_seed)
        xy_gap_length = Uniform(0.2, 2.2, seed=level_seed)
        side_to_side_gap_length = Uniform(0, 3, seed=level_seed)
        z_gap_length = Uniform(0.5, 1.0, seed=level_seed)
        part_length_short = Uniform(2.0, 4.0, seed=level_seed)
        spinner_extra_length = Uniform(0.5, 4.0, seed=level_seed)
        lava_spinner_height = Constant(0.5)
        spinner_platform_length = Uniform(6.0, 12.0, seed=level_seed)
        spinner_platform_ratio = Uniform(0.5, 1.5, seed=level_seed)
        part_length_long = Uniform(4.0, 12.0, seed=level_seed)
        silo_size_xy = Uniform(1.0, 3.0, seed=level_seed)
        silo_height = Uniform(5.0, 10.0, seed=level_seed)
        part_width = Uniform(0.01, 5.0, seed=level_seed) # TODO: restore
        door_width = Uniform(0.5, 2.0)
        tightrope_width = Uniform(0.05, 2.0, seed=level_seed)
        tightrope_length = Uniform(3.0, 6.0, seed=level_seed)
        n_ropes = UniformInt(3, 8, seed=level_seed)
        part_height = Uniform(0.01, 2.25, seed=level_seed)
        # Sign of the amplitude determines starting position
        # must be randomized to de-sync blocks.
        amplitude = UniformChoice([-3, 3], seed=level_seed)
        timing = Uniform(10, 20, seed=level_seed)
        velocity = Uniform(5, 5, seed=level_seed) # inverse
        #velocity = Uniform(2, 8, seed=level_seed) # inverse
        rotate = Uniform(0.0 * np.pi, 1.5 * np.pi, seed=level_seed) # From a quarter rotation per second to a full rotation per second.
        #rotate = Uniform(0, 0, seed=level_seed) # From a quarter rotation per second to a full rotation per second.
        starting_orientation = Uniform(-90.0, 90.0, seed=level_seed)
        binary = Binary(seed=level_seed)
        bounce = Uniform(20, 25, seed=level_seed)
        height_increment = Uniform(0.8, 1.2, seed=level_seed)
        platform_height = Uniform(0.1, 0.5, seed=level_seed)
        size_jitter = Gaussian(0.0, 0.05, seed=level_seed)
        orientation_jitter = Gaussian(0.0, 5, seed=level_seed)
        position_jitter = Gaussian(0.0, 0.1, seed=level_seed)

        n_trampolines = UniformInt(1, 2, seed=level_seed)
        length_gap = Uniform(1.0, 1.0, seed=level_seed)
        side_gap = Uniform(1.8, 1.8, seed=level_seed)
        random_safe_part = RandomPart(
            safe_part_types, part_width, platform_height, z_gap_length,
            orientation_jitter, orientation_jitter, orientation_jitter
        )


        # Conveyors
        conveyor_speed = Uniform(0.5, 1.5, seed=level_seed)
        conveyor_width = Uniform(0.25, 0.5, seed=level_seed)

        madrona_part_types = UniformChoice([Block, Lava], seed=level_seed)
        n_turns = UniformInt(1, 4, seed=level_seed) #TODO: restore to 2, 8 for staircase
        n_staircase_parts_high = UniformInt(2, 4, seed=level_seed)
        turnDir = Binary(seed=level_seed)

        uniform_sample = Uniform(0, 1, seed=level_seed)


        orientation = Uniform(0.0, 1.0, seed=level_seed)

        # Wall wrap
        n_wall_turns = UniformInt(1, 12, seed=level_seed)
        wrap_height_increment = Uniform(1.5, 2.0, seed=level_seed)
        wall_width = Uniform(3.0, 6.0, seed=level_seed)
        wall_length = Uniform(3.0, 5.0, seed=level_seed)

        # Physics distributions
        gravity = Uniform(196.2, 196.2, seed=level_seed)
        move_force = Uniform(216.034, 216.034, seed=level_seed)
        jump_force = Uniform(55.0, 55.0, seed=level_seed)
        air_resistance = Uniform(4.0, 4.0, seed=level_seed)

        # Not randomized
        spawn_position = Uniform(0.0, 0.0, seed=level_seed)
        spawn_velocity = Uniform(0.0, 0.0, seed=level_seed)


        #physicsDist = PhysicsDistribution(gravity, move_force, jump_force, air_resistance, spawn_position, spawn_velocity)

        generators = {
        #     "PaperExample": PaperExampleDistribution(
        #         Constant(5.0),
        #         Constant(2.0)
        #     ),
            "Plank": PlankDistribution(tightrope_width, part_length_long),
            # "PathJumps": PathJumpsDistribution(n_blocks, gap_lengths, block_lengths, part_width, binary),
            # "ThickPathJumps": ThickPathJumpsDistribution(n_blocks, gap_lengths, block_lengths, tightrope_width, block_thicknesses, binary),
            "ThickPathJumps": ThickPathJumpsDistribution(n_blocks, gap_lengths, block_lengths, tightrope_width, Constant(1.0), binary),
            # "ThickPathJumps": ThickPathJumpsDistribution(Constant(2), Uniform(0, 4), Constant(2), Constant(0.5), Constant(1.0), binary),
            # "GeneralPathJumps": GeneralPathJumpsDistribution(
            #     n_parts_low,
            #     part_length_short,
            #     part_length_long,
            #     binary
            # ),
            # "Lilypads": LilypadsDistribution(safe_part_types, n_parts_high, xy_gap_length),
            # NOT IMPLEMENTED
            # "ShrinkingLilypads": ShrinkingLilypadsDistribution(
            #     safe_part_types, 
            #     n_parts_high, 
            #     xy_gap_length,
            #     timing
            # ),
            "Tightrope": TightropeDistribution(
                n_ropes,
                binary,
                tightrope_width,
                tightrope_length
            ),
            # "LavaTightrope": LavaTightropeDistribution(
            #     n_parts_high,
            #     binary,
            #     Constant(0.5),
            #     part_length_long
            # ),
            # "IceTightrope": IceTightropeDistribution(
            #     n_parts_high,
            #     binary,
            #     Constant(0.5),
            #     part_length_long,
            #     Constant(0),
            #     Uniform(20, 50)
            # ),
            # "ConveyorTightrope": ConveyorTightropeDistribution(
            #     n_parts_high,
            #     turnDir,
            #     conveyor_width,
            #     part_length_long,
            #     conveyor_speed
            # ),
            # "ABAB": ABABDistribution(n_parts_high, madrona_part_types, madrona_part_types, xy_gap_length),
            # "Stairs": StairsDistribution(
            #     random_safe_part,
            #     n_parts_high,
            #     z_gap_length
            # ),
            "Staircase": StaircaseDistribution(
                Constant(1),#n_turns,
                binary,
                StairsDistribution(
                    random_safe_part,
                    Constant(7),#n_staircase_parts_high,
                    Constant(0.5)#part_height,
                ),
            ),
            # NOT IMPLEMENTED
            # "DisappearingStairs": DisappearingStairsDistribution(
            #     random_safe_part, 
            #     n_parts_high, 
            #     z_gap_length, 
            #     timing, 
            #     timing
            # ),
            "LavaJumps": LavaJumpsDistribution(
                n_parts_low, 
                part_length_long, 
                part_height
            ),
            # "IceJumps": IceJumpsDistribution(
            #     n_parts_low,
            #     xy_gap_length,
            #     part_length_short,
            #     z_gap_length,
            #     Constant(0),
            #     Uniform(20, 50),
            #     binary
            # ),
            "LavaDoors": LavaDoorsDistribution(
                n_parts_low, 
                part_length_long, 
                part_width,
                part_height,
                Uniform(0.75, 1.5),
                Uniform(0.1, 0.9)
            ),
            "LavaSpinner": LavaSpinnerDistribution(
                spinner_platform_ratio,
                lava_spinner_height,
                spinner_platform_length,
                Constant(np.pi), #rotate, 
                starting_orientation
            ),
            "ThinLavaSpinner": ThinLavaSpinnerDistribution(
                spinner_platform_ratio,
                lava_spinner_height,
                spinner_platform_length,
                Constant(np.pi), #rotate, 
                starting_orientation
            ),
            "LavaCheckerboard": LavaCheckerboardDistribution(
                n_parts_high, 
                checkerboard_size
            ),
            # "Trampoline": TrampolineDistribution(
            #     part_width, part_length_long, bounce
            # ),
            # "TrampolineWalls": TrampolineWallsDistribution(
            #     n_parts_low,
            #     part_width,
            #     part_length_long,
            #     bounce,
            #     height_increment,
            #     xy_gap_length
            # ),
            # "LavaTrampoline": LavaTrampolineDistribution(
            #     part_width,
            #     part_length_long,
            #     bounce,
            #     a_few
            # ),
            # "TrampolineCourse": TrampolineCourseDistribution(
            #     part_length_long, part_length_long, bounce, n_trampolines, side_gap, length_gap, turnDir
            # ),
            "WallJumps": WallJumpsDistribution(
                Constant(4),#n_parts_low_wall, 
                wall_jump_wall_depth
            ),
            # "Truss": TrussDistribution(
            #     part_length_long
            # ),
            # "TrussHurdles": TrussHurdlesDistribution(
            #     part_length_long,
            #     part_length_long, # distance between hurdles
            #     xy_gap_length,
            #     n_truss_hurdles_wide, # n wide
            #     n_parts_low,  # n hurdles
            #     truss_thicknesses
            # ),
            # "TrussSilo": TrussSiloDistribution(
            #     silo_height,
            #     silo_size_xy, # distance between hurdles
            #     xy_gap_length,
            #     n_truss_hurdles_wide, # n wide
            #     n_silos,  # n hurdles
            #     truss_thicknesses
            # ),
            # "WallWrapTightrope": WallWrapTightropeDistribution(
            #     wrap_height_increment,
            #     Constant(0.75),
            #     wall_length,
            #     wall_width,
            #     n_wall_turns,
            #     binary
            # ),
            # "Drop": DropDistribution(
            #     part_length_long
            # ),
            # "Wedges": WedgesDistribution(
            #     n_parts_low,
            #     part_length_short,
            #     part_height
            # ),
            # "SideToSideBlocks": SideToSideBlocksDistribution(
            #     n_parts_range, 
            #     velocity,
            #     side_to_side_gap_length, 
            #     amplitude
            # ),
            # "HiddenStairs": HiddenStairsDistribution(
            #     n_parts_high,
            #     uniform_sample,
            #     part_length_short
            # ),
            # "BlockPush": BlockPushDistribution(
            #     part_width,
            #     part_length_long,
            #     part_height,
            #     uniform_sample,
            # ),
            # "ButtonHoldSpinner": ButtonHoldSpinnerDistribution(
            #     part_width, 
            #     part_length_long, 
            #     starting_orientation,
            #     rotate
            # ),
            # "OneButtonPressWall": OneButtonPressWallDistribution(
            #     part_width,
            #     part_length_short,
            #     uniform_sample
            # ),
            # "DextrousButtonPressWall": DextrousButtonPressWallDistribution(
            #     part_width,
            #     part_length_short,
            #     uniform_sample
            # )
        }

        # for n in range(args.n_levels):
        # Add obstacles.
        if args.example == 'RandomObby':
            obby = random_obby(
                f"random_obby_s{seed}_l{args.n_levels}",
                generators=generators,
                n_sections=Constant(args.n_levels),
                obstacle_type=uniform_sample,
                unit=args.unit
            )
        elif args.example == 'RandomSortedObby':
            obby = random_sorted_obby(
                f"random_sorted_obby_s{seed}_l{args.n_levels}",
                generators=generators,
                n_sections=Constant(args.n_levels),
                obstacle_type=uniform_sample,
                unit=args.unit
            )
        elif args.example in generators:
            # obby = Obby(f"{args.example}_s{seed}_l{n}")
            # obby.spawn((0, 0, 0), 3)
            # obby.append_parts(generators[args.example].sample(
            #     name=args.example,
            #     checkpoint_name="Checkpoint 0",
            #     position=(0, 0, 1.5)
            # ).generate())
            example_generators = {args.example: generators[args.example]}
            if "Tightrope" in args.example:
                example_generators["Truss"] = generators["Truss"]
            if args.solve:
                obby, solution = random_obby(
                    f"{args.example}_s{seed}_l{args.n_levels}",
                    generators=example_generators,
                    n_sections=Constant(args.n_levels),
                    obstacle_type=uniform_sample,
                    unit=args.unit,
                    return_solution=True
                )
            else:
                obby = random_obby(
                    f"{args.example}_s{seed}_l{args.n_levels}",
                    generators=example_generators,
                    n_sections=Constant(args.n_levels),
                    obstacle_type=uniform_sample,
                    unit=args.unit
                )
            if args.randomize:
                obby.randomize(position_sampler=position_jitter, size_sampler=size_jitter, orientation_sampler=orientation_jitter)
        else:
            raise ValueError("Invalid example:", args.example)
        obby_path = args.json_path + obby.name + '.json'
        json_str = obby.jsonify()
        # print(json_str)


        with open(obby_path, 'w') as f:
            f.write(json_str)
        print("Saved to:", obby_path)

        if args.solve:
            solution_path = args.json_path + obby.name + '_solution.json'
            solution_json_str = solution.jsonify(unit=args.unit)
            with open(solution_path, 'w') as f:
                f.write(solution_json_str)
            print("Saved solution to:", solution_path)



if __name__ == "__main__":
    main()