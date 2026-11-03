import random

from obby.grammar.part import *
from obby.grammar.distribution import *
from obby.grammar.obstacles import *
from obby.examples.distribution_generators import *
from madrona_scripts.level_physics import PhysicsDistribution, Physics

from .obby_parameters import ObbyParameters

class TrainParameters1(ObbyParameters):
    def __init__(self, level_seed: int):
        rng = random.Random()
        rng.seed(level_seed)

        def get_random_seed(N=1000):
            return int(N * rng.random())

        n_blocks = UniformInt(2, 6, seed=get_random_seed())
        gap_lengths = Uniform(0, 3, seed=get_random_seed())  # the gap length sets the difficulty
        thick_path_jump_gap_lengths = Uniform(0, 3, seed=get_random_seed())
        block_lengths = Uniform(0.5, 3.0, seed=get_random_seed())
        block_thicknesses = Uniform(0.1, 5.0, seed=get_random_seed())
        truss_thicknesses = Uniform(1.0, 2.0, seed=get_random_seed())
        n_walls = UniformInt(1, 1, seed=get_random_seed())

        # Distributions for levels
        all_part_types = UniformChoice([Block, Cylinder, Wedge, Ball, Lava], seed=get_random_seed())
        safe_part_types = UniformChoice([Block], seed=get_random_seed())
        obstacle_type = Uniform(0, 1, seed=get_random_seed())
        n_sections = UniformInt(7, 7, seed=get_random_seed())
        a_few = UniformInt(1, 3, seed=get_random_seed())

        n_parts_range = UniformInt(2, 6, seed=get_random_seed())
        n_parts_low = UniformInt(2, 4, seed=get_random_seed())
        n_silos = UniformInt(1, 8, seed=get_random_seed())
        n_parts_low_wall = UniformInt(1, 5, seed=get_random_seed())
        wall_jump_wall_depth = Uniform(0.1, 2.0, seed=get_random_seed())
        n_parts_high = UniformInt(5, 8, seed=get_random_seed())  # lava checkerboard
        checkerboard_size = Uniform(0.5, 3.0, seed=get_random_seed())  # lava checkerboard

        n_truss_hurdles_wide = UniformInt(1, 3, seed=get_random_seed())

        xy_gap_length = Uniform(0.2, 2.2, seed=get_random_seed())
        side_to_side_gap_length = Uniform(0, 3, seed=get_random_seed())
        z_gap_length = Uniform(0.5, 1.0, seed=get_random_seed())
        part_length_short = Uniform(2.0, 4.0, seed=get_random_seed())
        spinner_extra_length = Uniform(0.5, 4.0, seed=get_random_seed())
        lava_spinner_height = Uniform(0.1, 3.0, seed=get_random_seed())
        
        # A thickness of 0.5 means the agent can always jump over or slide under the spinner.
        lava_spinner_thickness_jump_or_under = Uniform(0.05, 0.5, seed=get_random_seed())
        lava_spinner_thickness = Uniform(0.05, 1.5, seed=get_random_seed())
        
        spinner_platform_length = Uniform(6.0, 12.0, seed=get_random_seed())
        spinner_platform_ratio = Uniform(0.5, 1.5, seed=get_random_seed())
        spinner_platform_ratio_short = Uniform(0.2, 0.4, seed=get_random_seed())
        part_length_long = Uniform(4.0, 12.0, seed=get_random_seed())
        silo_size_xy = Uniform(1.0, 3.0, seed=get_random_seed())
        silo_height = Uniform(5.0, 10.0, seed=get_random_seed())
        part_width = Uniform(0.01, 5.0, seed=get_random_seed())
        tightrope_width = Uniform(0.05, 2.0, seed=get_random_seed())
        triplank_width = Uniform(0.5, 0.5, seed=get_random_seed())
        tightrope_length = Uniform(3.0, 6.0, seed=get_random_seed())
        n_ropes = UniformInt(3, 8, seed=get_random_seed())
        part_height = Uniform(0.01, 2.25, seed=get_random_seed())
        door_widths = Uniform(0.75, 1.5, seed=get_random_seed())
        door_centers = Uniform(0.1, 0.9, seed=get_random_seed())
        side_to_side_amplitude = Uniform(0.25, 4, seed=get_random_seed())
        side_to_side_velocity = Uniform(0.2, 2.5, seed=get_random_seed())

        rotate = Uniform(0.0 * np.pi, 1.5 * np.pi, seed=get_random_seed())
        starting_orientation = Uniform(-90.0, 90.0, seed=get_random_seed())
        binary = Binary(seed=get_random_seed())
        bounce = Uniform(20, 25, seed=get_random_seed())
        height_increment = Uniform(0.8, 1.2, seed=get_random_seed())
        platform_height = Uniform(0.1, 0.5, seed=get_random_seed())
        size_jitter = Gaussian(0.0, 0.05, seed=get_random_seed())
        orientation_jitter = Gaussian(0.0, 5, seed=get_random_seed())
        position_jitter = Gaussian(0.0, 0.1, seed=get_random_seed())

        n_trampolines = UniformInt(1, 2, seed=get_random_seed())
        length_gap = Uniform(1.0, 1.0, seed=get_random_seed())
        side_gap = Uniform(1.8, 1.8, seed=get_random_seed())
        random_safe_part = RandomPart(
            safe_part_types, part_width, platform_height, z_gap_length,
            orientation_jitter, orientation_jitter, orientation_jitter
        )

        # Lava Ring Drop
        offsets = Uniform(-2.0, 2.0, seed=get_random_seed())
        # Conveyors
        conveyor_speed = Uniform(0.5, 1.5, seed=get_random_seed())
        conveyor_width = Uniform(0.25, 0.5, seed=get_random_seed())

        madrona_part_types = UniformChoice([Block, Lava], seed=get_random_seed())
        n_turns = UniformInt(1, 4, seed=get_random_seed())
        n_staircase_parts_high = UniformInt(2, 4, seed=get_random_seed())
        turnDir = Binary(seed=get_random_seed())

        uniform_sample = Uniform(0, 1, seed=get_random_seed())


        orientation = Uniform(0.0, 1.0, seed=get_random_seed())

        # Wall wrap
        n_wall_turns = UniformInt(1, 12, seed=get_random_seed())
        wrap_height_increment = Uniform(1.5, 2.0, seed=get_random_seed())
        wall_width = Uniform(3.0, 6.0, seed=get_random_seed())
        wall_length = Uniform(3.0, 5.0, seed=get_random_seed())

        # Physics distributions
        gravity = Uniform(196.2, 196.2, seed=get_random_seed())
        move_force = Uniform(216.034, 216.034, seed=get_random_seed())
        jump_force = Uniform(55.0, 55.0, seed=get_random_seed())
        air_resistance = Uniform(4.0, 4.0, seed=get_random_seed())

        # Not randomized
        spawn_position = Uniform(0.0, 0.0, seed=get_random_seed())
        spawn_velocity = Uniform(0.0, 0.0, seed=get_random_seed())


        example_tightrope_width = Uniform(0.1, 0.5, seed=get_random_seed())
        example_tightrope_length = Uniform(8.0, 12.0, seed=get_random_seed())


        example_tightrope_width = Uniform(0.1, 0.5, seed=get_random_seed())
        example_tightrope_length = Uniform(8.0, 12.0, seed=get_random_seed())

        self.generators = {
            "Plank": PlankDistribution(tightrope_width, part_length_long),
            "TriplePlank": TriplePlankDistribution(triplank_width, part_length_long),
            "ThickPathJumps": ThickPathJumpsDistribution(n_blocks, gap_lengths, block_lengths, tightrope_width, Constant(1.0), Constant(0)),
            "Tightrope": TightropeDistribution(
                n_ropes,
                binary,
                tightrope_width,
                tightrope_length
            ),
            "ConveyorTightrope": ConveyorTightropeDistribution(
                n_parts_high,
                turnDir,
                conveyor_width,
                part_length_long,
                conveyor_speed
            ),
            "LavaStairs": LavaStairsDistribution(
                n_parts_high,
                Uniform(0.5, 1.5),
                Uniform(0.1, 2.0),
                part_width,
                Uniform(0.2, 3.0),
            ),
            "LavaJumps": LavaJumpsDistribution(
                n_parts_low, 
                part_length_long, 
                part_height
            ),
            "LavaDoors": LavaDoorsDistribution(
                n_parts_low, 
                part_length_long, 
                part_width,
                part_height,
                door_widths,
                door_centers
            ),
            "LavaSpinner": LavaSpinnerDistribution(
                spinner_platform_ratio,
                lava_spinner_thickness,
                lava_spinner_height,
                Constant(0.5),
                Constant(0.5),
                spinner_platform_length,
                Constant(0.75 * np.pi),
                starting_orientation
            ),
            "MoveAnywhereLavaSpinner": LavaSpinnerDistribution(
                spinner_platform_ratio,
                lava_spinner_thickness,
                lava_spinner_height,
                uniform_sample,
                uniform_sample,
                spinner_platform_length,
                Constant(0.75 * np.pi),
                starting_orientation
            ),
            "DoubleLavaSpinner": DoubleLavaSpinnerDistribution(
                spinner_platform_ratio,
                spinner_platform_ratio,
                lava_spinner_thickness,
                lava_spinner_thickness,
                lava_spinner_height,
                lava_spinner_height,
                uniform_sample,
                uniform_sample,
                uniform_sample,
                uniform_sample,
                spinner_platform_length,
                Constant(0.75 * np.pi),
                Constant(0.75 * np.pi),
                starting_orientation,
                starting_orientation
            ),
            "LavaCheckerboard": LavaCheckerboardDistribution(
                n_parts_high, 
                checkerboard_size
            ),
            "RotatingPlatforms": TripleWaterWheelsDistribution(
                part_width,
                part_width,
                part_width,
                part_length_short,
                part_length_short,
                part_length_short,
                Uniform(0.3 * np.pi, 0.5 * np.pi),
                Uniform(0.3 * np.pi, 0.5 * np.pi),
                Uniform(0.3 * np.pi, 0.5 * np.pi),
                starting_orientation,
                starting_orientation,
                starting_orientation
            ),
            "Trampoline": TrampolineDistribution(
                part_width, part_length_long, bounce
            ),
            "WallJumps": WallJumpsDistribution(
                n_parts_low_wall, 
                wall_jump_wall_depth
            ),
            "Truss": TrussDistribution(
                part_length_long
            ),
            "LavaRingDrop": LavaRingDropDistribution(
                part_length_long,
                Uniform(-5.0, 5.0),  # absolute offset from the platform edge
                Uniform(-5.0, 5.0),  # absolute offset from the platform edge
                Uniform(3, 6),  # size of the area around the checkpoint
                Uniform(-2.0, 2.0),  # absolute offset from the platform edge
                Uniform(-2.0, 2.0),  # absolute offset from the platform edge
                uniform_sample,  # fraction of drop_height, in [0, 1]
                xy_gap_length,  # side length of the ring hole
                lava_spinner_thickness,  # offset from the platform edge
            ),
            "SideToSideBlocks": SideToSideBlocksDistribution(
                n_parts_range, 
                side_to_side_velocity,
                side_to_side_gap_length, 
                side_to_side_amplitude
            ),
            "BlockPush": BlockPushDistribution(
                part_width,
                part_length_long,
                part_height,
                uniform_sample,
            ),
            "OneButtonPressWall": OneButtonPressWallDistribution(
                part_width,
                part_length_short,
                uniform_sample
            ),
            "DextrousButtonPressWall": DextrousButtonPressWallDistribution(
                part_width,
                part_length_short,
                uniform_sample
            )
        }

    def get_generators(self):
        return self.generators
    

class TrainParameters2(ObbyParameters):
    def __init__(self, seed: int):

        # Default distributions.
        all_part_types = UniformChoice([Block, Cylinder, Wedge, Ball, Lava], seed=seed)
        safe_part_types = UniformChoice([Block], seed=seed)
        uniform_sample = Uniform(0, 1, seed=seed)
        n_sections = UniformInt(10, 10, seed=seed)
        a_few = UniformInt(1, 3, seed=seed)
        n_parts_low = UniformInt(2, 4, seed=seed)
        n_parts_high = UniformInt(4, 8, seed=seed)
        xy_gap_length = Uniform(0.1, 2.5, seed=seed)
        z_gap_length = Uniform(0.5, 1.0, seed=seed)
        part_length_short = Uniform(1.0, 4.0, seed=seed)
        part_length_long = Uniform(4.0, 12.0, seed=seed)
        tightrope_width = Uniform(0.1, 1.0, seed=seed)
        part_width = Uniform(1.0, 5.0, seed=seed)
        platform_height = Uniform(0.1, 0.5, seed=seed)
        checkerboard_size = Uniform(0.5, 1.0, seed=seed)
        part_height = Uniform(0.5, 1.5, seed=seed)
        block_height = Uniform(0.5, 2.0, seed=seed)
        amplitude = Uniform(-6, 6, seed=seed)
        timing = Uniform(1, 10, seed=seed)
        velocity = Uniform(0.5, 5, seed=seed)
        rotate = Uniform(0.5, 5, seed=seed)
        binary = Binary(seed=seed)
        bounce = Uniform(10, 20, seed=seed)
        height_increment = Uniform(0.3, 0.75, seed=seed)
        starting_orientation = Uniform(0.0, 360.0, seed=seed)
        n_truss_hurdles_wide = UniformInt(1, 3, seed=seed)
        size_jitter = Gaussian(0.0, 0.05, seed=seed)
        orientation_jitter = Gaussian(0.0, 5, seed=seed)
        position_jitter = Gaussian(0.0, 0.1, seed=seed)
        random_safe_part = RandomPart(
            safe_part_types, part_width, platform_height, z_gap_length,
            orientation_jitter, orientation_jitter, orientation_jitter
        )

        self.generators = {
            "PathJumps": PathJumpsDistribution(
                n_parts_low,
                xy_gap_length, 
                part_length_short,
                part_width,
                binary
            ),
            "ThickPathJumps": ThickPathJumpsDistribution(
                n_parts_low,
                xy_gap_length, 
                part_length_short,
                part_width,
                block_height,
                binary
            ),
            "GeneralPathJumps": GeneralPathJumpsDistribution(
                n_parts_low,
                xy_gap_length, 
                part_length_long,
                Constant(False)
            ),
            "Lilypads": LilypadsDistribution(
                safe_part_types, 
                n_parts_high, 
                xy_gap_length
            ),
            "ShrinkingLilypads": ShrinkingLilypadsDistribution(
                safe_part_types, 
                n_parts_high, 
                xy_gap_length,
                timing
            ),
            "Tightrope": TightropeDistribution(
                n_parts_high,
                binary,
                tightrope_width,
                part_length_long
            ),
            "WalledTightrope": WalledTightropeDistribution(
                n_parts_high,
                binary,
                tightrope_width,
                part_length_long
            ),
            "SafetyNetTightrope": SafetyNetTightropeDistribution(
                n_parts_high,
                binary,
                tightrope_width,
                part_length_long
            ),
            "LavaTightrope": LavaTightropeDistribution(
                n_parts_high,
                binary,
                tightrope_width,
                part_length_long
            ),
            "IceTightrope": IceTightropeDistribution(
                n_parts_high,
                binary,
                tightrope_width,
                part_length_long,
                Constant(0),
                Uniform(20, 50)
            ),
            "ConveyorTightrope": ConveyorTightropeDistribution(
                n_parts_high,
                binary,
                tightrope_width,
                part_length_long,
                velocity
            ),
            "ABAB": ABABDistribution(
                n_parts_high,
                all_part_types,
                all_part_types,
                xy_gap_length
            ),
            "Stairs": StairsDistribution(
                random_safe_part,
                n_parts_high,
                z_gap_length,
            ),
            "DisappearingStairs": DisappearingStairsDistribution(
                random_safe_part,
                n_parts_high,
                z_gap_length,
                timing,
                timing
            ),
            "LavaJumps": LavaJumpsDistribution(
                n_parts_low,
                part_length_long,
                part_height
            ),
            "IceJumps": IceJumpsDistribution(
                n_parts_low,
                xy_gap_length,
                part_length_short, 
                z_gap_length, 
                Constant(0),
                Uniform(20, 50),
                binary
            ),
            "LavaSpinner": LavaSpinnerDistribution(
                part_length_long,
                z_gap_length,
                part_length_long,
                rotate,
                starting_orientation
            ),
            "Trampoline": TrampolineDistribution(
                part_width,
                part_length_long,
                bounce,
            ),
            "TrampolineWalls": TrampolineWallsDistribution(
                n_parts_low,
                part_width,
                part_length_long,
                bounce,
                height_increment,
                xy_gap_length
            ),
            "LavaTrampoline": LavaTrampolineDistribution(
                part_width,
                part_length_long,
                bounce,
                a_few
            ),
            "WallJumps": WallJumpsDistribution(
                n_parts_low
            ),
            "Truss": TrussDistribution(
                part_length_long
            ),
            "TrussHurdles": TrussHurdlesDistribution(
                part_length_long,
                part_length_long, # distance between hurdles
                xy_gap_length,
                n_truss_hurdles_wide, # n wide
                n_parts_low  # n hurdles
            ),
            "Drop": DropDistribution(
                part_length_long
            ),
            "LavaCheckerboard": LavaCheckerboardDistribution(
                n_parts_high,
                checkerboard_size
            ),
            "Wedges": WedgesDistribution(
                n_parts_low,
                part_length_short,
                part_height
            ),
            "LavaWedges": LavaWedgesDistribution(
                n_parts_low,
                part_length_short,
                checkerboard_size
            ),
            "SideToSideBlocks": SideToSideBlocksDistribution(
                n_parts_low,
                velocity,
                xy_gap_length,
                amplitude,
            ),
            "HiddenStairs": HiddenStairsDistribution(
                n_parts_high,
                uniform_sample,
                part_length_short
            ),
            "BlockPush": BlockPushDistribution(
                part_width,
                part_length_long,
                block_height,
                uniform_sample,
            ),
            "ButtonHoldSpinner": ButtonHoldSpinnerDistribution(
                part_width, 
                part_length_long, 
                starting_orientation,
                rotate
            ),
            "OneButtonPressWall": OneButtonPressWallDistribution(
                part_width,
                part_length_short,
                uniform_sample
            ),
            "DextrousButtonPressWall": DextrousButtonPressWallDistribution(
                part_width,
                part_length_short,
                uniform_sample
            ),
            "PathVsLilypads": PathVsLilypadsDistribution(
                n_parts_high,
                xy_gap_length
            ),
            "ConveyorVsLilypads": ConveyorVsLilypadsDistribution(
                n_parts_high,
                xy_gap_length,
                velocity
            ),
            "Plank": PlankDistribution(
                part_width,
                part_length_long
            )
        }

    def get_generators(self):
        return self.generators