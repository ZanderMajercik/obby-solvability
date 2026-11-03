import copy

from obby.grammar.behaviors import *
from obby.grammar.constants import *
from obby.grammar.part import *
from obby.grammar.type_aliases import Vector as VectorType
from obby.grammar.utils import *
from obby.grammar.waypoint import *


################################################################################
# HELPER METHODS
################################################################################

def get_position_after(part: Part, turn: float = 0, gap: float = 0) -> VectorType:
    if part.orientation[1] % 180 == 0:
        x_len = part.size[0]
        z_len = part.size[2]
    else:
        x_len = part.size[2]
        z_len = part.size[0]

    if turn % 360 == 0:
        x = 0
        z = z_len / 2 + gap
    elif turn % 360 == 90:
        x = x_len / 2 + gap
        z = 0
    elif turn % 360 == 180:
        x = 0
        z = -(z_len / 2 + gap)
    else:
        x = -(x_len / 2 + gap)
        z = 0
    return (
        part.position[0] + x,
        part.position[1],
        part.position[2] + z
    )


def place_checkpoint_after(
        part: Part, checkpoint_name: str, checkpoint_size: float = 1.5, gap: float = 0) -> Checkpoint:
    checkpoint_position = get_position_after(part, part.orientation[1], gap=checkpoint_size)
    checkpoint = Checkpoint(
        checkpoint_name,
        position=checkpoint_position,
        size=(3, PLATFORM_Y, 3)
    )
    return checkpoint


def set_orientation(parts: list[Part] | list[Waypoint], about: VectorType, orientation: VectorType):
    for part in parts:
        part.position = orient(np.array(vsub(part.position, about)), orientation)

        if isinstance(part, Part):
            part.orientation = vadd(part.orientation, orientation)
            for behavior in part.behaviors:
                if isinstance(behavior, (behaviors.TouchTranslateBehavior, behaviors.PersistentTranslateBehavior)):
                    behavior.start_position = orient(np.array(vsub(behavior.start_position, about)), orientation)
                    behavior.end_position = orient(np.array(vsub(behavior.end_position, about)), orientation)
                elif isinstance(behavior, behaviors.PersistentRotateBehavior):
                    behavior.about = orient(np.array(vsub(behavior.about, about)), orientation)


def set_position(parts: list[Part], position: VectorType):
    for part in parts:
        part.position = (
            part.position[0] + position[0],
            part.position[1] + position[1],
            part.position[2] + position[2]
        )

        if isinstance(part, Part):
            for behavior in part.behaviors:
                if isinstance(behavior, behaviors.PersistentRotateBehavior):
                    behavior.about = (
                        behavior.about[0] + position[0],
                        behavior.about[1] + position[1],
                        behavior.about[2] + position[2],
                    )


def generate_pretty_rl_directions(n_turns: int, turns: Distribution) -> str:
    directions = "r" if n_turns % 2 == 0 else "l"
    in_a_row = False
    for _ in range(n_turns - 1):
        if directions[-1] == 'l':
            turn = 'r'
        else:
            turn = 'l'
        directions += turn
    return directions

#def generate_pretty_rl_directions(n_turns: int, turns: Distribution) -> str:
#    directions = ""
#    in_a_row = False
#    for _ in range(n_turns):
#        if len(directions) > 1:
#            in_a_row = (directions[-2] == directions[-1])
#        if in_a_row:
#            turn = 'l' if directions[-1] == 'r' else 'r'
#            in_a_row = False
#        else:
#            turn = 'l' if turns.sample() < 0.5 else 'r'
#        directions += turn
#    return directions

def generate_random_rl_directions(n_turns: int, turns: Distribution) -> str:
    directions = ""
    for _ in range(n_turns):
        turn = 'l' if turns.sample() < 0.5 else 'r'
        directions += turn
    return directions


def generate_pretty_url_directions(n_turns: int, turns: Distribution) -> str:
    directions = "u"
    for _ in range(n_turns - 1):
        turn_sample = turns.sample()
        if turn_sample < 0.5:
            directions += 'u'
        else:
            if directions[-1] == 'u':
                if turns.sample() < 0.5:
                    directions += 'r'
                else:
                    directions += 'l'
            elif directions[-1] == 'r':
                if turns.sample() < 0.75:
                    directions += 'u'
                else:
                    directions += 'r'
            else:
                if turns.sample() < 0.75:
                    directions += 'u'
                else:
                    directions += 'l'
    return directions


def generate_pretty_directions(n_turns: int, weights=None) -> str:
    directions = ""
    direction = UniformInt(1, 3) if not weights else UniformInt(1, 3, weights=weights)
    for _ in range(n_turns):
        if direction.sample() == 1:
            turn = 'l'
        elif direction.sample() == 2:
            turn = 'f'
        else:
            turn = 'r'
        directions += turn
    return directions


################################################################################
# OBSTACLES
################################################################################
class Obstacle:
    def __init__(
            self,
            name: str,
            position: VectorType = (0, 0, 0),
            orientation: VectorType = (0, 0, 0),
            checkpoint_name: str = "Checkpoint",
            base_score: float = 1
    ):
        self.name = name
        self.position = position
        self.checkpoint_name = checkpoint_name
        self.orientation = orientation
        self.base_score = base_score
        self.parts = []

    def __str__(self) -> str:
        rep_str = ""
        for part in self.parts:
            rep_str += str(part)
        return rep_str

    def jsonify(self) -> tuple[str, dict[str, tp.Any]]:
        for part in self.parts:
            part.jsonify()

    def geometry(self, unit: float = 1, chkpt=True):
        all_v = []  # np.empty((0,3))
        all_f = []  # np.empty((0,3))
        all_c = []
        v_offset = 0
        parts = self.generate()
        if not chkpt:
            parts = parts[:-1]
        for part in parts:
            v, f, c = part.geometry(unit=unit, v_offset=v_offset)
            v_offset += v.shape[0]
            all_v.append(v)
            all_f.append(f)
            all_c.append(c)

        # print(all_v)
        return np.concatenate(all_v), np.concatenate(all_f), np.concatenate(all_c)
        # return torch.tensor(all_v), torch.tensor(all_f), torch.tensor(all_c)

    def generate(self) -> list[Part]:
        return []

    def get_last_block_position(self) -> VectorType:
        return self.parts[-1].position

    def waypoints(self) -> list[VectorType]:
        return []

    def score(self, unit: float = 1) -> float:
        """Assign a difficulty score computed based on the parameters, where higher = more difficult."""
        return 0


class ObstacleDistribution():
    def sample(
            self,
            name: str,
            checkpoint_name: str,
            position: VectorType = (0, 0, 0),
            orientation: VectorType = (0, 0, 0)
    ) -> Obstacle:
        return Obstacle(name, position, orientation, checkpoint_name)


class Plank(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            width: float,
            length: float,
            base_score: float = 1
    ):
        super(Plank, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.width = width
        self.length = length

    def score(self, unit=1):
        return 1

    def generate(self, return_solution: bool = False):
        parts = []
        plank = Block(
            f"{self.name}_Plank",
            (0, 0, self.length / 2),
            (self.width, PLATFORM_Y, self.length),
        )
        # Add checkpoint.
        checkpoint = place_checkpoint_after(plank, self.checkpoint_name)
        parts.append(plank)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts
    
class TriplePlank(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            width: float,
            length: float,
            base_score: float = 1
    ):
        super(TriplePlank, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.width = width
        self.length = length

    def score(self, unit=1):
        return 1

    def generate(self, return_solution: bool = False):
        parts = []
        CHECKPOINT_WIDTH = 3
        middle_plank = None
        for i in [-1, 0, 1]:
            plank = Block(
                f"{self.name}_TriplePlank{i}",
                (i * ((CHECKPOINT_WIDTH - self.width) * 0.5), 0, self.length / 2),
                (self.width, PLATFORM_Y, self.length),
            )
            parts.append(plank)
            if i == 0:
                middle_plank = plank

        # Add checkpoint, must be last.
        checkpoint = place_checkpoint_after(middle_plank, self.checkpoint_name)
        parts.append(checkpoint)
        
        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts

class PaperExample(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            tightrope_width: float,
            length: float,
            base_score: float = 1
    ):
        super(PaperExample, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.tightrope_width = tightrope_width
        self.length = length

    def score(self, unit=1):
        return 1

    def generate(self, return_solution: bool = False):
        parts = []
        # Narrow plank/tightrope from start to end.
        plank = Block(
            f"{self.name}_Plank",
            (0, 0, self.length / 2),
            (self.tightrope_width, PLATFORM_Y, self.length),
        )

        # Put LAVA on the plank
        lava_spacing = 3
        num_lava = int(self.length / lava_spacing)
        lava_height = 0.5
        for n in range(num_lava):
            lava_block = Lava(
                f"{self.name}_Lava{n}",
                (0, lava_height / 2, n * lava_spacing + lava_spacing / 2),
                (self.tightrope_width, lava_height, 0.5)
            )
            parts.append(lava_block)


        # Wide path of 3 blocks making a U-shape from start to end
        wide_path_width = 2
        wide_path_out = 1
        wide_start = Block(
            f"{self.name}_WideStart",
            (-(1.5 + wide_path_out / 2), 0, -1.5),
            (wide_path_out, PLATFORM_Y, wide_path_width),
        )
        wide_mid = Block(
            f"{self.name}_WideMid",
            (-(1.5 + wide_path_out + wide_path_width / 2), 0, self.length / 2),
            (wide_path_width, PLATFORM_Y, self.length + wide_path_width + 3),
        )
        wide_end = Block(
            f"{self.name}_WideEnd",
            (-(1.5 + wide_path_out / 2), 0, self.length + 1.5),
            (wide_path_out, PLATFORM_Y, wide_path_width),
        )


        # Add checkpoint.
        checkpoint = place_checkpoint_after(plank, self.checkpoint_name)
        parts.append(plank)
        parts.append(checkpoint)
        parts.append(wide_start)
        parts.append(wide_mid)
        parts.append(wide_end)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class PaperExampleDistribution(ObstacleDistribution):
    def __init__(self, width: Distribution, length: Distribution):
        self.tightrope_width = width
        self.length = length

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return PaperExample(
            name, position, orientation, checkpoint_name,
            self.tightrope_width.sample(), self.length.sample()
        )

class PlankDistribution(ObstacleDistribution):
    def __init__(self, width: Distribution, length: Distribution):
        self.width = width
        self.length = length

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return Plank(
            name, position, orientation, checkpoint_name,
            self.width.sample(), self.length.sample()
        )

class TriplePlankDistribution(ObstacleDistribution):
    def __init__(self, width: Distribution, length: Distribution):
        self.width = width
        self.length = length

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TriplePlank(
            name, position, orientation, checkpoint_name,
            self.width.sample(), self.length.sample()
        )


class PathJumps(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            gap_lengths: list[float],
            block_lengths: list[float],
            block_widths: list[float],
            base_score: float = 1
    ):
        super(PathJumps, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.block_widths = block_widths

    def score(self, unit=1):
        total_gap_lengths = 0
        for gap_length in self.gap_lengths:
            if gap_length * unit > MAX_JUMP_LENGTH:
                return math.inf
            total_gap_lengths += gap_length

        total_block_length = 0
        for block_length in self.block_lengths:
            total_block_length += block_length
        return self.base_score * self.n_blocks * total_gap_lengths / total_block_length

    def generate(self, return_solution: bool = False):
        parts = []
        waypoints = []
        this_block_position = (0, 0, self.block_lengths[0] / 2)
        for n in range(self.n_blocks):
            this_block = Block(
                f"{self.name}_Block{n}",
                this_block_position,
                (self.block_widths[n], PLATFORM_Y, self.block_lengths[n]),
            )
            end_of_block = vadd(this_block_position, (0, 0, self.block_lengths[n] / 2 - 0.2))
            waypoints.append(Waypoint(end_of_block, Action.WALK))
            if n < self.n_blocks - 1:
                this_block_position = vadd(
                    this_block_position,
                    (0, 0, (self.block_lengths[n] + self.block_lengths[n + 1]) / 2 + self.gap_lengths[n])
                )
                start_of_block = vsub(this_block_position, (0, 0, self.block_lengths[n + 1] / 2 - 0.2))
                waypoints.append(Waypoint(start_of_block, Action.JUMP))
            parts.append(this_block)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)
        waypoints.append(Waypoint(checkpoint.position, Action.WALK))

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_orientation(waypoints, (0, 0, 0), self.orientation)
        set_position(parts, self.position)
        set_position(waypoints, self.position)

        if return_solution:
            return parts, waypoints
        else:
            return parts


class PathJumpsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            gap_lengths: Distribution,
            block_lengths: Distribution,
            block_widths: Distribution,
            uniform: bool
    ):
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.block_widths = block_widths
        self.uniform = uniform

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_blocks_sample = int(self.n_blocks.sample())
        if self.uniform:
            block_length_samples = n_blocks_sample * [self.block_lengths.sample()]
            block_width_samples = n_blocks_sample * [self.block_widths.sample()]
            gap_length_samples = (n_blocks_sample - 1) * [self.gap_lengths.sample()]
        else:
            block_length_samples = [self.block_lengths.sample() for _ in range(n_blocks_sample)]
            block_width_samples = [self.block_widths.sample() for _ in range(n_blocks_sample)]
            gap_length_samples = [self.gap_lengths.sample() for _ in range(n_blocks_sample - 1)]

        # Sample params and create instance.
        return PathJumps(
            name, position, orientation, checkpoint_name,
            n_blocks_sample,
            gap_length_samples,
            block_length_samples,
            block_width_samples
        )


class ThickPathJumps(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            gap_lengths: list[float],
            block_lengths: list[float],
            block_widths: list[float],
            block_thicknesses: list[float],
            base_score: float = 1
    ):
        super(ThickPathJumps, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.block_widths = block_widths
        self.block_thicknesses = block_thicknesses

    def score(self, unit=1):
        total_gap_lengths = 0
        for gap_length in self.gap_lengths:
            if gap_length * unit > MAX_JUMP_LENGTH:
                return math.inf
            total_gap_lengths += gap_length

        total_block_length = 0
        for block_length in self.block_lengths:
            total_block_length += block_length
        return self.base_score * self.n_blocks * total_gap_lengths / total_block_length

    def generate(self, return_solution: bool = False):
        parts = []
        waypoints = []
        this_block_position = (0, -self.block_thicknesses[0] / 2, self.block_lengths[0] / 2)
        for n in range(self.n_blocks):
            this_block = Block(
                f"{self.name}_Block{n}",
                this_block_position,
                (self.block_widths[n], self.block_thicknesses[n], self.block_lengths[n]),
            )
            end_of_block = vadd(this_block_position, (0, 0, self.block_lengths[n] / 2 - 0.2))
            waypoints.append(Waypoint(end_of_block, Action.WALK))
            if n < self.n_blocks - 1:
                this_block_position = vadd(
                    this_block_position,
                    (0,
                     (self.block_thicknesses[n] / 2) - (self.block_thicknesses[n + 1] / 2),
                     (self.block_lengths[n] + self.block_lengths[n + 1]) / 2 + self.gap_lengths[n])
                )
                start_of_block = vsub(this_block_position, (0, 0, self.block_lengths[n + 1] / 2 - 0.2))
                waypoints.append(Waypoint(start_of_block, Action.JUMP))
            parts.append(this_block)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.position = vadd(checkpoint.position, (0, self.block_thicknesses[-1] / 2, 0))
        parts.append(checkpoint)
        waypoints.append(Waypoint(checkpoint.position, Action.WALK))

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_orientation(waypoints, (0, 0, 0), self.orientation)
        set_position(parts, self.position)
        set_position(waypoints, self.position)

        if return_solution:
            return parts, waypoints
        else:
            return parts


class ThickPathJumpsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            gap_lengths: Distribution,
            block_lengths: Distribution,
            block_widths: Distribution,
            block_thicknesses: Distribution,
            uniform: bool
    ):
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.block_widths = block_widths
        self.block_thicknesses = block_thicknesses
        self.uniform = uniform

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_blocks_sample = int(self.n_blocks.sample())
        if self.uniform:
            block_length_samples = n_blocks_sample * [self.block_lengths.sample()]
            block_width_samples = n_blocks_sample * [self.block_widths.sample()]
            block_thickness_samples = n_blocks_sample * [self.block_thicknesses.sample()]
            gap_length_samples = (n_blocks_sample - 1) * [self.gap_lengths.sample()]
        else:
            block_length_samples = [self.block_lengths.sample() for _ in range(n_blocks_sample)]
            block_width_samples = [self.block_widths.sample() for _ in range(n_blocks_sample)]
            block_thickness_samples = [self.block_thicknesses.sample() for _ in range(n_blocks_sample)]
            gap_length_samples = [self.gap_lengths.sample() for _ in range(n_blocks_sample - 1)]

        # Sample params and create instance.
        return ThickPathJumps(
            name, position, orientation, checkpoint_name,
            n_blocks_sample,
            gap_length_samples,
            block_length_samples,
            block_width_samples,
            block_thickness_samples
        )


class GeneralPathJumps(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            gap_lengths: list[VectorType],
            block_lengths: list[float],
            base_score: float = 1
    ):
        super(GeneralPathJumps, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths

    def score(self, unit=1):
        total_gap_lengths = 0
        for gap_length in self.gap_lengths:
            if np.linalg.norm(gap_length) * unit > MAX_JUMP_LENGTH:
                return math.inf
            total_gap_lengths += np.linalg.norm(gap_length)

        total_block_length = 0
        for block_length in self.block_lengths:
            total_block_length += block_length
        return self.base_score * self.n_blocks * total_gap_lengths / total_block_length

    def generate(self):
        parts = []
        this_block_position = (0, 0, 0 + self.block_lengths[0] / 2)
        for n in range(self.n_blocks):
            this_block = Block(
                f"{self.name}_Block{n}",
                this_block_position,
                (1, PLATFORM_Y, self.block_lengths[n]),
            )
            if n < self.n_blocks - 1:
                this_block_position = vadd(
                    this_block_position,
                    (self.gap_lengths[n][0],
                     self.gap_lengths[n][1],
                     (self.block_lengths[n] + self.block_lengths[n + 1]) / 2 + self.gap_lengths[n][2])
                )
            parts.append(this_block)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class GeneralPathJumpsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            gap_lengths: Distribution,
            block_lengths: Distribution,
            uniform: bool
    ):
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.uniform = uniform

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_blocks_sample = int(self.n_blocks.sample())
        neg = UniformChoice([-1, 1])
        if self.uniform.sample():
            block_length_samples = n_blocks_sample * [self.block_lengths.sample()]
            gap_length_samples = (n_blocks_sample - 1) * [
                (neg.sample() * self.gap_lengths.sample(), neg.sample() * self.gap_lengths.sample(),
                 self.gap_lengths.sample())]
        else:
            block_length_samples = [self.block_lengths.sample() for _ in range(n_blocks_sample)]
            gap_length_samples = [(neg.sample() * self.gap_lengths.sample(), neg.sample() * self.gap_lengths.sample(),
                                   self.gap_lengths.sample()) for
                                  _ in range(n_blocks_sample - 1)]

        # Sample params and create instance.
        return GeneralPathJumps(
            name, position, orientation, checkpoint_name,
            n_blocks_sample,
            gap_length_samples,
            block_length_samples
        )


class IceJumps(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            gap_lengths: list[float],
            block_lengths: list[float],
            block_heights: list[float],
            friction: float,
            friction_weight: float,
            base_score: float = 1
    ):
        super(IceJumps, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.block_heights = block_heights
        self.friction = friction
        self.friction_weight = friction_weight

    def score(self, unit=1):
        total_gap_lengths = 0
        for gap_length in self.gap_lengths:
            if np.linalg.norm(gap_length) * unit > MAX_JUMP_LENGTH:
                return math.inf
            total_gap_lengths += np.linalg.norm(gap_length)

        total_block_length = 0
        for block_length in self.block_lengths:
            total_block_length += block_length
        total_block_height = 0
        for block_height in self.block_heights:
            total_block_height += block_height
        return self.base_score * self.n_blocks * total_block_height * total_gap_lengths * (
                1 + self.friction + self.friction_weight) / total_block_length

    def generate(self):
        parts = []
        this_block_position = (0, self.block_heights[0] / 2, 0 + self.block_lengths[0] / 2)
        for n in range(self.n_blocks):
            this_block = Wedge(
                f"{self.name}_Wedge{n}",
                this_block_position,
                (1, self.block_heights[n], self.block_lengths[n]),
            )
            this_block.add_behavior(FrictionBehavior(friction=self.friction, weight=self.friction_weight))
            if n < self.n_blocks - 1:
                this_block_position = vadd(
                    this_block_position,
                    (0, 0, (self.block_lengths[n] + self.block_lengths[n + 1]) / 2 + self.gap_lengths[n])
                )
            parts.append(this_block)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class IceJumpsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            gap_lengths: Distribution,
            block_lengths: Distribution,
            block_heights: Distribution,
            friction: Distribution,
            friction_weight: Distribution,
            uniform: bool
    ):
        self.n_blocks = n_blocks
        self.gap_lengths = gap_lengths
        self.block_lengths = block_lengths
        self.block_heights = block_heights
        self.friction = friction
        self.friction_weight = friction_weight
        self.uniform = uniform

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_blocks_sample = int(self.n_blocks.sample())
        if self.uniform:
            block_length_samples = n_blocks_sample * [self.block_lengths.sample()]
            block_height_samples = n_blocks_sample * [self.block_heights.sample()]
            gap_length_samples = (n_blocks_sample - 1) * [self.gap_lengths.sample()]
        else:
            block_length_samples = [self.block_lengths.sample() for _ in range(n_blocks_sample)]
            block_height_samples = [self.block_heights.sample() for _ in range(n_blocks_sample)]
            gap_length_samples = [self.gap_lengths.sample() for _ in range(n_blocks_sample - 1)]

        # Sample params and create instance.
        return IceJumps(
            name, position, orientation, checkpoint_name,
            n_blocks_sample,
            gap_length_samples,
            block_length_samples,
            block_height_samples,
            self.friction.sample(),
            self.friction_weight.sample()
        )


class Lilypads(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            pad_type: tp.Callable,
            n_pads: int,
            gap_length: float,
            base_score: float = 2
    ):
        super(Lilypads, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.pad_type = pad_type
        self.n_pads = n_pads
        self.gap_length = gap_length

    def score(self, unit=1):
        if self.gap_length * unit > MAX_JUMP_LENGTH:
            return math.inf
        return self.base_score * self.n_pads * self.gap_length

    def generate(self):
        parts = []
        this_block_position = (0, 0, 0.5 + self.gap_length)
        pad_size = (1, PLATFORM_Y, 1)
        if self.pad_type in [Cylinder, Ball]:
            pad_size = (1, 1, 1)
        for n in range(self.n_pads):
            this_block = self.pad_type(
                f"{self.name}_{self.pad_type.part_type}{n}",
                this_block_position,
                size=pad_size,
                orientation=(0, 0, 0)
            )
            if n < self.n_pads - 1:
                this_block_position = vadd(this_block_position, (0, 0, self.gap_length + 1))
            parts.append(this_block)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.position = vadd(checkpoint.position, (0, 0, self.gap_length))
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class LilypadsDistribution(ObstacleDistribution):
    def __init__(
            self,
            pad_type: Distribution,
            n_pads: Distribution,
            gap_length: Distribution,
    ):
        self.pad_type = pad_type
        self.n_pads = n_pads
        self.gap_length = gap_length

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        pad_type_sample, n_pads_sample, gap_length_sample = (
            self.pad_type.sample(),
            self.n_pads.sample(),
            self.gap_length.sample()
        )
        return Lilypads(
            name, position, orientation, checkpoint_name,
            pad_type_sample,
            n_pads_sample,
            gap_length_sample
        )


class ShrinkingLilypads(Lilypads):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            pad_type: tp.Callable,
            n_pads: int,
            gap_length: float,
            time: float,
            base_score: float = 2
    ):
        super(ShrinkingLilypads, self).__init__(
            name, position, orientation, checkpoint_name,
            pad_type=pad_type, n_pads=n_pads, gap_length=gap_length,
            base_score=base_score
        )
        self.time = time

    def score(self, unit=1):
        if self.gap_length * unit > MAX_JUMP_LENGTH:
            return math.inf
        return self.base_score * self.n_pads * self.gap_length * max(1, self.time)

    def generate(self):
        parts = super(ShrinkingLilypads, self).generate()
        for part in parts:
            if isinstance(part, Block):
                part.add_behavior(TouchScaleBehavior(
                    start_scale=(1.0, 1.0, 1.0), end_scale=(0.0, 1.0, 0.0), time=self.time
                ))
        return parts


class ShrinkingLilypadsDistribution(ShrinkingLilypads):
    def __init__(
            self,
            pad_type: Distribution,
            n_pads: Distribution,
            gap_length: Distribution,
            time: Distribution,
    ):
        self.pad_type = pad_type
        self.n_pads = n_pads
        self.gap_length = gap_length
        self.time = time

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        pad_type_sample, n_pads_sample, gap_length_sample, time_sample = (
            self.pad_type.sample(),
            self.n_pads.sample(),
            self.gap_length.sample(),
            self.time.sample()
        )
        return ShrinkingLilypads(
            name, position, orientation, checkpoint_name,
            pad_type_sample,
            n_pads_sample,
            gap_length_sample,
            time_sample
        )


class Tightrope(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_turns: int,
            turns: list[tp.Literal['l', 'r']],
            width: float,
            lengths: list[float],
            base_score: float = 2
    ):
        super(Tightrope, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def score(self, unit=1):
        total_length = 0
        for length in self.lengths:
            total_length += length
        return self.base_score * self.n_turns * total_length / self.width

    def generate(self):
        parts = []

        orientation = 0
        curr_length = self.lengths[0]
        position = (0, 0, curr_length / 2)
        part = Block(
            f"{self.name}_Rope0",
            position,
            (self.width, PLATFORM_Y, self.lengths[0]),
            orientation=(0, orientation, 0)
        )
        parts.append(part)
        for i, (curr_length, turn) in enumerate(zip(self.lengths[1:], self.turns)):
            position = get_position_after(parts[-1], turn=orientation, gap=0)
            if turn == 'r':
                orientation += 90
            elif turn == 'l':
                orientation -= 90
            else:
                raise TypeError("Invalid turn:", turn)
            if orientation % 360 == 0:
                x = 0
                z = curr_length / 2 - self.width / 2
            elif orientation % 360 == 90:
                x = curr_length / 2 - self.width / 2
                z = 0
            elif orientation % 360 == 180:
                x = 0
                z = -curr_length / 2 + self.width / 2
            else:
                x = -curr_length / 2 + self.width / 2
                z = 0
            position = vadd(position, (x, 0, z))
            length = curr_length
            size = (self.width, PLATFORM_Y, length)
            part = Block(f"{self.name}_Rope{i + 1}", position, size, orientation=(0, orientation, 0))
            parts.append(part)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.orientation = parts[-1].orientation
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class TightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            width: Distribution,
            lengths: Distribution
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_pretty_rl_directions(n_turns_sample, self.turns)
        width_sample = self.width.sample()
        lengths_sample = [self.lengths.sample() for _ in range(n_turns_sample + 1)]
        return Tightrope(
            name, position, orientation, checkpoint_name,
            n_turns_sample,
            turns_sample,
            width_sample,
            lengths_sample
        )


class WalledTightrope(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_turns: int,
            turns: list[tp.Literal['l', 'r']],
            width: float,
            lengths: list[float],
            base_score: float = 2
    ):
        super(WalledTightrope, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def score(self, unit=1):
        total_length = 0
        for length in self.lengths:
            total_length += length
        return self.base_score * self.n_turns * total_length / self.width

    def generate(self):
        parts = []

        orientation = 0
        curr_length = self.lengths[0]
        position = (0, 0, curr_length / 2)
        part = Block(
            f"{self.name}_Rope0",
            position,
            (self.width, PLATFORM_Y, self.lengths[0]),
            orientation=(0, orientation, 0)
        )
        corner_position = vadd(position, (0, 0, curr_length / 2 + self.width / 2))
        wall_size = (PLATFORM_Y, 1, curr_length)
        corner = Block(f"{self.name}_Corner0", corner_position, size=(self.width, PLATFORM_Y, self.width),
                       orientation=(0, orientation, 0))
        left_wall = Block(f"{self.name}_LeftWall0", vadd(vadd(position, (self.width / 2, 0, 0)), (0, 0.5, 0)),
                          wall_size, orientation=(0, orientation, 0))
        right_wall = Block(f"{self.name}_RightWall0", vadd(vsub(position, (self.width / 2, 0, 0)), (0, 0.5, 0)),
                           wall_size, orientation=(0, orientation, 0))
        corner_wall = Block(f"{self.name}_CornerWall0",
                            vadd(get_position_after(corner, turn=orientation, gap=0), (0, 0.5, 0)),
                            (PLATFORM_Y, 1, self.width), orientation=(0, orientation + 90, 0))
        parts.append(corner_wall)
        parts.append(left_wall)
        parts.append(right_wall)
        parts.append(corner)
        parts.append(part)
        for i, (curr_length, turn) in enumerate(zip(self.lengths[1:], self.turns)):
            position = get_position_after(parts[-1], turn=orientation, gap=0)
            if turn == 'r':
                orientation += 90
            elif turn == 'l':
                orientation -= 90
            else:
                raise TypeError("Invalid turn:", turn)
            if orientation % 360 == 0:
                x = -self.width / 2 if turn == 'r' else self.width / 2
                z = curr_length / 2 + self.width / 2
                offset = (self.width / 2, 0, 0)
                corner_offset = (0, 0, curr_length / 2 + self.width / 2)
            elif orientation % 360 == 90:
                x = curr_length / 2 + self.width / 2
                z = self.width / 2 if turn == 'r' else -self.width / 2
                offset = (0, 0, self.width / 2)
                corner_offset = (curr_length / 2 + self.width / 2, 0, 0)
            elif orientation % 360 == 180:
                x = self.width / 2 if turn == 'r' else -self.width / 2
                z = -curr_length / 2 - self.width / 2
                offset = (self.width / 2, 0, 0)
                corner_offset = (curr_length / 2, 0, 0)
                corner_offset = (0, 0, -curr_length / 2 - self.width / 2)
            else:
                x = -curr_length / 2 - self.width / 2
                z = -self.width / 2 if turn == 'r' else self.width / 2
                offset = (0, 0, self.width / 2)
                corner_offset = (-curr_length / 2 - self.width / 2, 0, 0)
            position = vadd(position, (x, 0, z))
            length = curr_length
            size = (self.width, PLATFORM_Y, length)
            wall_size = (PLATFORM_Y, 1, length)
            part = Block(f"{self.name}_Rope{i + 1}", position, size, orientation=(0, orientation, 0))
            left_wall = Block(f"{self.name}_LeftWall{i + 1}", vadd(vadd(position, offset), (0, 0.5, 0)), wall_size,
                              orientation=(0, orientation, 0))
            right_wall = Block(f"{self.name}_RightWall{i + 1}", vadd(vsub(position, offset), (0, 0.5, 0)), wall_size,
                               orientation=(0, orientation, 0))
            corner_position = vadd(position, corner_offset)
            corner = Block(f"{self.name}_Corner{i + 1}", corner_position, size=(self.width, PLATFORM_Y, self.width),
                           orientation=(0, 0, 0))
            parts.append(corner)
            corner_wall = Block(f"{self.name}_CornerWall{i + 1}",
                                vadd(get_position_after(corner, turn=orientation, gap=0), (0, 0.5, 0)),
                                (PLATFORM_Y, 1, self.width), orientation=(0, orientation + 90, 0))
            parts.append(corner_wall)
            parts.append(left_wall)
            parts.append(right_wall)
            parts.append(part)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.orientation = parts[-1].orientation
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class WalledTightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            width: Distribution,
            lengths: Distribution
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_pretty_rl_directions(n_turns_sample, self.turns)
        width_sample = self.width.sample()
        lengths_sample = [self.lengths.sample() for _ in range(n_turns_sample + 1)]
        return WalledTightrope(
            name, position, orientation, checkpoint_name,
            n_turns_sample,
            turns_sample,
            width_sample,
            lengths_sample
        )


class SafetyNetTightrope(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_turns: int,
            turns: list[tp.Literal['l', 'r']],
            width: float,
            lengths: list[float],
            base_score: float = 2
    ):
        super(SafetyNetTightrope, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def score(self, unit=1):
        total_length = 0
        for length in self.lengths:
            total_length += length
        return self.base_score * self.n_turns * total_length / self.width

    def generate(self):
        parts = []

        orientation = 0
        curr_length = self.lengths[0]
        position = (0, 0, curr_length / 2)
        part = Block(
            f"{self.name}_Rope0",
            position,
            (self.width, PLATFORM_Y, self.lengths[0]),
            orientation=(0, orientation, 0)
        )
        safety_size = (self.width + 2, PLATFORM_Y, curr_length)
        net = Block(f"{self.name}_Net0", vsub(position, (0, 0.5, 0)), safety_size, orientation=(0, orientation, 0))
        parts.append(net)
        parts.append(part)
        for i, (curr_length, turn) in enumerate(zip(self.lengths[1:], self.turns)):
            position = get_position_after(parts[-1], turn=orientation, gap=0)
            if turn == 'r':
                orientation += 90
            elif turn == 'l':
                orientation -= 90
            else:
                raise TypeError("Invalid turn:", turn)
            if orientation % 360 == 0:
                x = 0
                z = curr_length / 2 - self.width / 2
            elif orientation % 360 == 90:
                x = curr_length / 2 - self.width / 2
                z = 0
            elif orientation % 360 == 180:
                x = 0
                z = -curr_length / 2 + self.width / 2
            else:
                x = -curr_length / 2 + self.width / 2
                z = 0
            position = vadd(position, (x, 0, z))
            length = curr_length
            size = (self.width, PLATFORM_Y, length)
            safety_size = (self.width + 2, PLATFORM_Y, length)
            part = Block(f"{self.name}_Rope{i + 1}", position, size, orientation=(0, orientation, 0))
            net = Block(f"{self.name}_Net{i + 1}", vsub(position, (0, 0.5, 0)), safety_size,
                        orientation=(0, orientation, 0))
            parts.append(net)
            parts.append(part)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.orientation = parts[-1].orientation
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class SafetyNetTightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            width: Distribution,
            lengths: Distribution
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_pretty_rl_directions(n_turns_sample, self.turns)
        width_sample = self.width.sample()
        lengths_sample = [self.lengths.sample() for _ in range(n_turns_sample + 1)]
        return SafetyNetTightrope(
            name, position, orientation, checkpoint_name,
            n_turns_sample,
            turns_sample,
            width_sample,
            lengths_sample
        )


class LavaTightrope(Tightrope):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_turns: int,
            turns: list[tp.Literal['l', 'r']],
            width: float,
            lengths: list[float],
            base_score: float = 3
    ):
        super(LavaTightrope, self).__init__(
            name, position, orientation, checkpoint_name,
            n_turns, turns, width, lengths, base_score=base_score)

    def score(self, unit=1):
        total_length = 0
        for length in self.lengths:
            total_length += (length + 1)
        return self.base_score * self.n_turns * total_length / self.width

    def generate(self):
        # Create parts from Tightrope, already includes checkpoint and position, orientation are set.
        parts = super(LavaTightrope, self).generate()

        lava_idx = 0
        n_ropes = len(parts) - 1
        for i in range(n_ropes):
            part = parts[i]
            # Randomly place some lava.
            x_sign = math.cos(math.radians(part.orientation[1]))
            z_sign = math.sin(math.radians(part.orientation[1]))
            lava_position = (
                x_sign * (random.random() - 0.5) * part.size[0] + part.position[0],
                part.position[1] + 0.1,
                z_sign * (random.random() - 0.5) * part.size[0] + part.position[2]
            )
            size = (0.75, PLATFORM_Y, 0.75)
            lava_part = Lava(f"{self.name}_Lava{lava_idx}", position=lava_position, size=size,
                             orientation=part.orientation)
            parts.insert(len(parts) - 1, lava_part)
            lava_idx += 1

        return parts


class LavaTightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            width: Distribution,
            lengths: Distribution,
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        # Generate a random turn sequence, but never have more than two of the same 
        # direction turn in a row.
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_pretty_rl_directions(n_turns_sample, self.turns)

        # Generate lengths
        width_sample = self.width.sample()
        lengths_sample = [self.lengths.sample() for _ in range(n_turns_sample + 1)]
        return LavaTightrope(
            name, position, orientation, checkpoint_name,
            n_turns_sample,
            turns_sample,
            width_sample,
            lengths_sample
        )


class IceTightrope(Tightrope):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_turns: int,
            turns: list[tp.Literal['l', 'r']],
            width: float,
            lengths: list[float],
            friction_coeffs: list[float],
            friction_weights: list[float],
            base_score: float = 3
    ):
        super(IceTightrope, self).__init__(
            name, position, orientation, checkpoint_name,
            n_turns, turns, width, lengths, base_score=base_score)
        self.friction_coeffs = friction_coeffs
        self.friction_weights = friction_weights

    def score(self, unit=1):
        total_length = 0
        for length in self.lengths:
            total_length += (length + 1)
        total_friction = 0
        for friction, weight in zip(self.friction_coeffs, self.friction_weights):
            total_friction += (friction * weight + 1)
        return self.base_score * self.n_turns * total_length * total_friction / self.width

    def generate(self):
        # Create parts from Tightrope, already includes checkpoint and position, orientation are set.
        parts = super(IceTightrope, self).generate()

        n_ropes = len(parts) - 1
        for i in range(n_ropes):
            part = parts[i]
            # Add friction behavior to make rope slippery.
            part.add_behavior(FrictionBehavior(
                friction=self.friction_coeffs[i], weight=self.friction_weights[i]
            ))

        return parts


class IceTightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            width: Distribution,
            lengths: Distribution,
            friction_coeffs: Distribution,
            friction_weights: Distribution
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths
        self.friction_coeffs = friction_coeffs
        self.friction_weights = friction_weights

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        # Generate a random turn sequence, but never have more than two of the same 
        # direction turn in a row.
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_pretty_rl_directions(n_turns_sample, self.turns)

        # Generate lengths
        width_sample = self.width.sample()
        lengths_sample = [self.lengths.sample() for _ in range(n_turns_sample + 1)]
        friction_coeffs_sample = [self.friction_coeffs.sample() for _ in range(n_turns_sample + 1)]
        friction_weights_sample = [self.friction_weights.sample() for _ in range(n_turns_sample + 1)]
        return IceTightrope(
            name, position, orientation, checkpoint_name,
            n_turns_sample,
            turns_sample,
            width_sample,
            lengths_sample,
            friction_coeffs_sample,
            friction_weights_sample
        )


class ConveyorTightrope(Tightrope):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_turns: int,
            turns: list[tp.Literal['l', 'r']],
            width: float,
            lengths: list[float],
            velocity: float,
            base_score: float = 3
    ):
        super(Tightrope, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths
        self.velocity = velocity

    def score(self, unit=1):
        total_length = 0
        for length in self.lengths:
            total_length += length
        return self.base_score * self.n_turns * total_length * self.velocity / self.width

    def generate(self):
        # Create parts from Tightrope, already includes checkpoint and position, orientation are set.
        parts = super(ConveyorTightrope, self).generate()

        n_ropes = len(parts) - 1
        for i in range(n_ropes):
            if i % 2 == 1:
                part = parts[i]
                if part.orientation[1] == 0:
                    part.add_behavior(behaviors.ConveyorBehavior(0, self.velocity))
                elif part.orientation[1] == 90:
                    part.add_behavior(behaviors.ConveyorBehavior(-self.velocity, 0))
                elif part.orientation[1] == 180:
                    part.add_behavior(behaviors.ConveyorBehavior(0, -self.velocity))
                else:
                    part.add_behavior(behaviors.ConveyorBehavior(self.velocity, 0))
        return parts


class ConveyorTightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            width: Distribution,
            lengths: Distribution,
            velocity: Distribution
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.width = width
        self.lengths = lengths
        self.velocity = velocity

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        # Generate a random turn sequence, but never have more than two of the same 
        # direction turn in a row.
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_pretty_rl_directions(n_turns_sample, self.turns)

        # Generate lengths
        width_sample = self.width.sample()
        lengths_sample = [self.lengths.sample() for _ in range(n_turns_sample + 1)]
        velocity_sample = self.velocity.sample()
        return ConveyorTightrope(
            name, position, orientation, checkpoint_name,
            n_turns_sample,
            turns_sample,
            width_sample,
            lengths_sample,
            velocity_sample
        )


class ABAB(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_parts: int,
            a: Part,
            b: Part,
            gap_length: float,
            base_score: float = 2
    ):
        super(ABAB, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_parts = n_parts
        self.a = a
        self.b = b
        self.gap_length = gap_length

    def score(self, unit=1):
        if self.gap_length * unit > MAX_JUMP_LENGTH:
            return math.inf
        part_a_score = 1
        if isinstance(self.a, (Lava, Conveyor, Trampoline)):
            part_a_score = 2
        part_b_score = 1
        if isinstance(self.b, (Lava, Conveyor, Trampoline)):
            part_b_score = 2
        return self.base_score * self.gap_length * self.n_parts * part_a_score * part_b_score

    def generate(self):
        parts = []
        position = (0, 0, self.a.size[2] / 2)
        for n in range(self.n_parts):
            part = copy.deepcopy(self.a) if n % 2 == 0 else copy.deepcopy(self.b)
            part.name = f"{self.name}_Part{n}"
            part.position = position
            parts.append(part)
            position = vadd(position, vadd((0, 0, (self.a.size[2] + self.b.size[2]) / 2), (0, 0, self.gap_length)))

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.orientation = parts[-1].orientation
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class ABABDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_parts: Distribution,
            a: Distribution,
            b: Distribution,
            gap: Distribution
    ):
        self.n_parts = n_parts
        self.a = a
        self.b = b
        self.gap = gap

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        # Sample parameters.
        n_parts_sample = self.n_parts.sample()

        part_a_type = self.a.sample()
        part_size = (1, PLATFORM_Y, 1)
        if part_a_type in [Cylinder, Ball]:
            part_size = (1, 1, 1)
        a_sample = part_a_type(
            f"Part a",
            (0, 0, 0),
            size=part_size,
            orientation=(0, 0, 0)
        )

        part_b_type = self.b.sample()
        part_size = (1, PLATFORM_Y, 1)
        if part_b_type in [Cylinder, Ball]:
            part_size = (1, 1, 1)
        b_sample = part_b_type(
            f"Part b",
            (0, 0, 0),
            size=part_size,
            orientation=(0, 0, 0)
        )

        gap_sample = self.gap.sample()
        return ABAB(
            name, position, orientation, checkpoint_name,
            n_parts_sample,
            a_sample,
            b_sample,
            gap_sample
        )


class Stairs(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            stair: Part,
            n_stairs: int,
            gap_height: float,
            base_score: float = 1
    ):
        super(Stairs, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.stair = stair
        self.n_stairs = n_stairs
        self.gap_height = gap_height

    def score(self, unit=1):
        if self.gap_height * unit > MAX_JUMP_LENGTH:
            return math.inf
        return self.base_score * self.n_stairs * self.gap_height

    def generate(self):
        parts = []
        for n in range(self.n_stairs):
            position = vadd((0, 0, self.stair.size[2] / 2),
                            vscale(n, (0, self.gap_height, self.stair.size[2])))
            part = copy.deepcopy(self.stair)
            part.name = f"{self.name}_Stair{n}"
            part.position = position
            parts.append(part)

        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class StairsDistribution(ObstacleDistribution):
    def __init__(
            self,
            stair: Distribution,
            n_stairs: Distribution,
            gap_height: Distribution,
    ):
        self.stair = stair
        self.n_stairs = n_stairs
        self.gap_height = gap_height

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return Stairs(
            name, position, orientation, checkpoint_name,
            self.stair.sample(),
            self.n_stairs.sample(),
            self.gap_height.sample(),
        )


class LavaStairs(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_stairs: int,
            gap_height: float,
            gap_fwd: float,
            stair_width: float,
            stair_fwd: float,
            base_score: float = 1
    ):
        super(LavaStairs, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_stairs = n_stairs
        self.gap_height = gap_height
        self.gap_fwd = gap_fwd
        self.stair_width = stair_width
        self.stair_fwd = stair_fwd

    def score(self, unit=1):
        if self.gap_height * unit > MAX_JUMP_LENGTH:
            return math.inf
        return self.base_score * self.n_stairs * self.gap_height

    def generate(self):
        parts = []
        for n in range(self.n_stairs):
            position = vadd((0, 0, self.stair_fwd / 2),
                            vscale(n, (0, self.gap_height, self.stair_fwd + self.gap_fwd)))
            stair = Block(
                f"{self.name}_Stair{n}",
                position=position,
                size=(self.stair_width, PLATFORM_Y, self.stair_fwd),
                orientation=(0, 0, 0)
            )
            lava = Lava(
                f"{self.name}_Lava{n}",
                position=(vadd(stair.position, (0, self.gap_height / 2, (self.stair_fwd + self.gap_fwd) / 2))),
                size=(self.stair_width, self.gap_height, self.gap_fwd),
                orientation=(0, 0, 0)
            )
            parts.append(stair)
            parts.append(lava)

        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        checkpoint.position = vadd(checkpoint.position, (0, self.gap_height / 2, 0))
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class LavaStairsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_stairs: Distribution,
            gap_height: Distribution,
            gap_fwd: Distribution,
            stair_width: Distribution,
            stair_fwd: Distribution,
    ):
        self.n_stairs = n_stairs
        self.gap_height = gap_height
        self.gap_fwd = gap_fwd
        self.stair_width = stair_width
        self.stair_fwd = stair_fwd

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaStairs(
            name, position, orientation, checkpoint_name,
            self.n_stairs.sample(),
            self.gap_height.sample(),
            self.gap_fwd.sample(),
            self.stair_width.sample(),
            self.stair_fwd.sample()
        )


class Staircase(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            staircase: Stairs,
            turns: list[tp.Literal['l', 'r']],
            base_score: float = 1
    ):
        super(Staircase, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.staircase = staircase
        self.turns = turns

    def score(self, unit=1):
        return self.base_score * self.n_flights * self.turns * self.staircase.score(unit)

    def generate(self):
        parts = self.staircase.generate()
        self.staircase.name += str("0")
        orientation = self.orientation
        debug_parts_len = len(parts)
        for i, turn in enumerate(self.turns):
            # Remove the ending checkpoint
            ckpt = parts[-1]
            parts[-1] = Block(
                "Platform" + str(i),
                ckpt.position,
                ckpt.size,
                (0, 0, 0)
            )
            staircase_parts = self.staircase.generate()
            self.staircase.name += str(i + 1)
            if turn == 'r':
                orientation = (orientation[0], orientation[1] + 90, orientation[2])
            elif turn == 'l':
                orientation = (orientation[0], orientation[1] - 90, orientation[2])
            else:
                raise TypeError("Invalid turn:", turn)
            set_orientation(staircase_parts, (0, 0, 0), orientation)
            offset = orient(np.array([0, 0, ckpt.size[2] * 0.5]), orientation)
            set_position(staircase_parts, ckpt.position + offset)
            parts += staircase_parts

        set_position(parts, self.position)
        return parts


class StaircaseDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            stairs: StairsDistribution
    ):
        self.n_turns = n_turns
        self.stairs = stairs
        self.turns = turns

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_turns_sample = self.n_turns.sample()
        turns_sample = generate_random_rl_directions(n_turns_sample, self.turns)
        return Staircase(
            name, position, orientation, checkpoint_name,
            self.stairs.sample(name + "stairs", checkpoint_name),
            turns_sample
        )


class DisappearingStairs(Stairs):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            stair: Part,
            n_stairs: int,
            gap_height: float,
            disappear_delay: float,
            reappear_delay: float,
            base_score: float = 2
    ):
        self.stair = stair
        # self.stair.add_behavior(DisappearBehavior(disappear_delay, reappear_delay))
        super(DisappearingStairs, self).__init__(
            name, position, orientation, checkpoint_name, self.stair, n_stairs, gap_height, base_score=base_score)
        self.disappear_delay = disappear_delay
        self.reappear_delay = reappear_delay
        self.disappear_behavior = DisappearBehavior(disappear_delay, reappear_delay)

    def score(self, unit=1):
        if self.gap_height * unit > MAX_JUMP_LENGTH:
            return math.inf
        return self.base_score * self.n_stairs * self.gap_height * self.reappear_delay / self.disappear_delay

    def generate(self):
        parts = super(DisappearingStairs, self).generate()
        # added for time profiling
        for part in parts:
            if isinstance(part, Block):
                part.add_behavior(self.disappear_behavior)
        return parts


class DisappearingStairsDistribution(ObstacleDistribution):
    def __init__(
            self,
            stair: Distribution,
            n_stairs: Distribution,
            gap_height: Distribution,
            disappear_delay: Distribution,
            reappear_delay: Distribution
    ):
        self.stair = stair
        self.n_stairs = n_stairs
        self.gap = gap_height
        self.disappear_delay = disappear_delay
        self.reappear_delay = reappear_delay

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return DisappearingStairs(
            name, position, orientation, checkpoint_name,
            self.stair.sample(),
            self.n_stairs.sample(),
            self.gap.sample(),
            self.disappear_delay.sample(),
            self.reappear_delay.sample()
        )


class LavaJumps(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_parts: int,
            platform_length: float,
            lava_height: float,
            base_score: float = 2
    ):
        super(LavaJumps, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_parts = n_parts
        self.platform_length = platform_length
        self.lava_height = lava_height
        self.n_parts = n_parts

    def score(self, unit=1):
        return self.base_score * self.n_parts * self.lava_height

    def generate(self):
        total_platform_length = self.platform_length * (self.n_parts + 1) + self.n_parts
        platform = Block(
            f"{self.name}_Platform",
            (0, 0, total_platform_length / 2),
            (3, PLATFORM_Y, total_platform_length)
        )
        parts = [platform]
        for n in range(self.n_parts):
            lava_block = Lava(
                f"{self.name}_Lava{n}",
                (0, self.lava_height / 2, self.platform_length * (n + 1) + n),
                (7, self.lava_height, 0.5)
            )
            parts.append(lava_block)

        checkpoint = place_checkpoint_after(platform, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class LavaJumpsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_parts: Distribution,
            platform_length: Distribution,
            lava_height: Distribution,
    ):
        self.n_parts = n_parts
        self.platform_length = platform_length
        self.lava_height = lava_height

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaJumps(
            name, position, orientation, checkpoint_name,
            self.n_parts.sample(),
            self.platform_length.sample(),
            self.lava_height.sample()
        )


class LavaSpinner(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            spinner_length: float,
            spinner_thickness: float,
            spinner_height: float,
            spinner_x: float,
            spinner_y: float,
            platform_size: float,
            angular: float,
            starting_orientation: float,
            base_score: float = 2
    ):
        super(LavaSpinner, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.spinner_length = spinner_length * platform_size # + platform_size # ensure spinner longer than platform
        self.spinner_thickness = spinner_thickness
        self.spinner_height = spinner_height
        self.spinner_x = spinner_x
        self.spinner_y = spinner_y
        self.platform_size = platform_size
        self.angular = angular
        self.starting_orientation = starting_orientation

    def score(self, unit=1):
        if self.spinner_height > CHARACTER_HEIGHT:
            return 1 / self.platform_size
        if self.spinner_height * unit < CHARACTER_HEIGHT and self.spinner_height * unit > MAX_JUMP_HEIGHT:
            return math.inf
        return self.base_score * self.spinner_length * self.angular / (self.spinner_height * self.platform_size)

    def generate(self):
        before = Block(
            f"{self.name}_Before",
            (0.0, 0.0, 1.0),
            (1, PLATFORM_Y, 2)
        )
        platform = Block(
            f"{self.name}_Platform",
            (0.0, 0.0, self.platform_size / 2 + 2),
            (self.platform_size, PLATFORM_Y, self.platform_size)
        )
        pole = Cylinder(
            f"{self.name}_Pole",
            ((0.5 - self.spinner_x) * self.platform_size, 0.25, self.platform_size * self.spinner_y + 2),
            (self.spinner_height, 0.5, 0.5),
            orientation=(0, 0, 90)
        )
        spinner = Lava(
            f"{self.name}_LavaSpinner",
            ((0.5 - self.spinner_x) * self.platform_size, 0.5 + self.spinner_height / 2, self.platform_size * self.spinner_y + 2),
            (self.spinner_length, self.spinner_thickness, 0.3),
            orientation=(0, self.starting_orientation, 0)
        )
        rotate_behavior = behaviors.PersistentRotateBehavior(
            (0.0, 0.5 + self.spinner_height, self.platform_size / 2 + 2),
            self.angular
        )
        after = Block(
            f"{self.name}_After",
            (0.0, 0.0, self.platform_size + 3),
            (1, PLATFORM_Y, 2)
        )
        spinner.add_behavior(rotate_behavior)
        checkpoint = place_checkpoint_after(after, self.checkpoint_name)
        parts = [before, platform, pole, spinner, after, checkpoint]

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class LavaSpinnerDistribution(ObstacleDistribution):
    def __init__(
            self,
            spinner_length: Distribution,
            spinner_thickness: Distribution,
            spinner_height: Distribution,
            spinner_x: Distribution,
            spinner_y: Distribution,
            platform_size: Distribution,
            angular: Distribution,
            starting_orientation: Distribution
    ):
        self.spinner_length = spinner_length
        self.spinner_thickness = spinner_thickness
        self.spinner_height = spinner_height
        self.spinner_x = spinner_x
        self.spinner_y = spinner_y
        self.platform_size = platform_size
        self.angular = angular
        self.starting_orientation = starting_orientation

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaSpinner(
            name, position, orientation, checkpoint_name,
            self.spinner_length.sample(),
            self.spinner_thickness.sample(),
            self.spinner_height.sample(),
            self.spinner_x.sample(),
            self.spinner_y.sample(),
            self.platform_size.sample(),
            self.angular.sample(),
            self.starting_orientation.sample()
        )
    

class ThinLavaSpinner(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            spinner_length: float,
            spinner_height: float,
            platform_size: float,
            angular: float,
            starting_orientation: float,
            base_score: float = 2
    ):
        super(ThinLavaSpinner, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.spinner_length = spinner_length * platform_size # + platform_size # ensure spinner longer than platform
        self.spinner_height = spinner_height
        self.platform_size = platform_size
        self.angular = angular
        self.starting_orientation = starting_orientation

    def score(self, unit=1):
        if self.spinner_height > CHARACTER_HEIGHT:
            return 1 / self.platform_size
        if self.spinner_height * unit < CHARACTER_HEIGHT and self.spinner_height * unit > MAX_JUMP_HEIGHT:
            return math.inf
        return self.base_score * self.spinner_length * self.angular / (self.spinner_height * self.platform_size)

    def generate(self):
        before = Block(
            f"{self.name}_Before",
            (0.0, 0.0, 1.0),
            (1, PLATFORM_Y, 2)
        )
        platform = Block(
            f"{self.name}_Platform",
            (0.0, 0.0, self.platform_size / 2 + 2),
            (self.platform_size, PLATFORM_Y, self.platform_size)
        )
        pole = Cylinder(
            f"{self.name}_Pole",
            (0.0, 0.25, self.platform_size / 2 + 2),
            (self.spinner_height, 0.5, 0.5),
            orientation=(0, 0, 90)
        )
        spinner = Lava(
            f"{self.name}_ThinLavaSpinner",
            (0.0, 0.5 + self.spinner_height / 2, self.platform_size / 2 + 2),
            (self.spinner_length, 0.3, 0.3),
            orientation=(0, self.starting_orientation, 0)
        )
        rotate_behavior = behaviors.PersistentRotateBehavior(
            (0.0, 0.5 + self.spinner_height, self.platform_size / 2 + 2),
            self.angular
        )
        after = Block(
            f"{self.name}_After",
            (0.0, 0.0, self.platform_size + 3),
            (1, PLATFORM_Y, 2)
        )
        spinner.add_behavior(rotate_behavior)
        checkpoint = place_checkpoint_after(after, self.checkpoint_name)
        parts = [before, platform, pole, spinner, after, checkpoint]

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class ThinLavaSpinnerDistribution(ObstacleDistribution):
    def __init__(
            self,
            spinner_length: Distribution,
            spinner_height: Distribution,
            platform_size: Distribution,
            angular: Distribution,
            starting_orientation: Distribution
    ):
        self.spinner_length = spinner_length
        self.spinner_height = spinner_height
        self.platform_size = platform_size
        self.angular = angular
        self.starting_orientation = starting_orientation

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return ThinLavaSpinner(
            name, position, orientation, checkpoint_name,
            self.spinner_length.sample(),
            self.spinner_height.sample(),
            self.platform_size.sample(),
            self.angular.sample(),
            self.starting_orientation.sample()
        )
    

class DoubleLavaSpinner(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            spinner_length_a: float,
            spinner_length_b: float,
            spinner_thickness_a: float,
            spinner_thickness_b: float,
            spinner_height_a: float,
            spinner_height_b: float,
            x_a: float,  # in (0, 1)
            y_a: float,  # in (0, 1)
            x_b: float,  # in (0, 1)
            y_b: float,  # in (0, 1)
            platform_size: float,
            angular_a: float,
            angular_b: float,
            starting_orientation_a: float,
            starting_orientation_b: float,
            base_score: float = 2
    ):
        super(DoubleLavaSpinner, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.spinner_length_a = spinner_length_a * platform_size
        self.spinner_length_b = spinner_length_b * platform_size # + platform_size # ensure spinner longer than platform
        self.spinner_thickness_a = spinner_thickness_a
        self.spinner_thickness_b = spinner_thickness_b
        self.spinner_height_a = spinner_height_a
        self.spinner_height_b = spinner_height_b
        self.platform_size = platform_size
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b
        self.angular_a = angular_a
        self.angular_b = angular_b
        self.starting_orientation_a = starting_orientation_a
        self.starting_orientation_b = starting_orientation_b

    def generate(self):
        before = Block(
            f"{self.name}_Before",
            (0.0, 0.0, 1.0),
            (1, PLATFORM_Y, 2)
        )
        platform = Block(
            f"{self.name}_Platform",
            (0.0, 0.0, self.platform_size / 2 + 2),
            (self.platform_size, PLATFORM_Y, self.platform_size)
        )

        # Spinner 1
        pole1 = Cylinder(
            f"{self.name}_Pole1",
            (self.platform_size * (self.x_a - 0.5), 0.25, self.platform_size * self.y_a + 2),
            (self.spinner_height_a, 0.5, 0.5),
            orientation=(0, 0, 90)
        )
        spinner1 = Lava(
            f"{self.name}_LavaSpinner1",
            (self.platform_size * (self.x_a - 0.5), 0.5 + self.spinner_height_a / 2, self.platform_size * self.y_a + 2),
            (self.spinner_length_a, self.spinner_thickness_a, 0.3),
            orientation=(0, self.starting_orientation_a, 0)
        )
        rotate_behavior1 = behaviors.PersistentRotateBehavior(
            (self.platform_size * (self.x_a - 0.5), 0.5 + self.spinner_height_a, self.platform_size * self.y_a + 2),
            self.angular_a
        )
        spinner1.add_behavior(rotate_behavior1)

        # Spinner 2
        pole2 = Cylinder(
            f"{self.name}_Pole2",
            (self.platform_size * (self.x_b - 0.5), 0.25, self.platform_size * self.y_b + 2),
            (self.spinner_height_b, 0.5, 0.5),
            orientation=(0, 0, 90)
        )
        spinner2 = Lava(
            f"{self.name}_LavaSpinner2",
            (self.platform_size * (self.x_b - 0.5), 0.5 + self.spinner_height_b / 2, self.platform_size * self.y_b + 2),
            (self.spinner_length_b, self.spinner_thickness_b, 0.3),
            orientation=(0, self.starting_orientation_b, 0)
        )
        rotate_behavior2 = behaviors.PersistentRotateBehavior(
            (self.platform_size * (self.x_b - 0.5), 0.5 + self.spinner_height_b, self.platform_size * self.y_b + 2),
            self.angular_b
        )
        spinner2.add_behavior(rotate_behavior2)
        after = Block(
            f"{self.name}_After",
            (0.0, 0.0, self.platform_size + 3),
            (1, PLATFORM_Y, 2)
        )
        checkpoint = place_checkpoint_after(after, self.checkpoint_name)
        parts = [before, platform, pole1, spinner1, pole2, spinner2, after, checkpoint]

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts
    

class DoubleLavaSpinnerDistribution(ObstacleDistribution):
    def __init__(
            self,
            spinner_length_a: Distribution,
            spinner_length_b: Distribution,
            spinner_thickness_a: Distribution,
            spinner_thickness_b: Distribution,
            spinner_height_a: Distribution,
            spinner_height_b: Distribution,
            x_a: Distribution,
            y_a: Distribution,
            x_b: Distribution,
            y_b: Distribution,
            platform_size: Distribution,
            angular_a: Distribution,
            angular_b: Distribution,
            starting_orientation_a: Distribution,
            starting_orientation_b: Distribution
    ):
        self.spinner_length_a = spinner_length_a
        self.spinner_length_b = spinner_length_b
        self.spinner_thickness_a = spinner_thickness_a
        self.spinner_thickness_b = spinner_thickness_b
        self.spinner_height_a = spinner_height_a
        self.spinner_height_b = spinner_height_b
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b
        self.platform_size = platform_size
        self.angular_a = angular_a
        self.angular_b = angular_b
        self.starting_orientation_a = starting_orientation_a
        self.starting_orientation_b = starting_orientation_b
        # print(self.spinner_lengths)

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return DoubleLavaSpinner(
            name, position, orientation, checkpoint_name,
            self.spinner_length_a.sample(), 
            self.spinner_length_b.sample(),
            self.spinner_thickness_a.sample(), 
            self.spinner_thickness_b.sample(),
            self.spinner_height_a.sample(),
            self.spinner_height_b.sample(),
            self.x_a.sample(),
            self.y_a.sample(),
            self.x_b.sample(),
            self.y_b.sample(),
            self.platform_size.sample(),
            self.angular_a.sample(),
            self.angular_b.sample(),
            self.starting_orientation_a.sample(),
            self.starting_orientation_b.sample()
        )


class LavaDoors(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_parts: int,
            platform_length: float,
            platform_width: float,
            lava_height: float,
            door_widths: list[float],
            door_centers: list[float],  # Range from (0, 1) as a ratio of the platform width
            base_score: float = 2
    ):
        super(LavaDoors, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_parts = n_parts
        self.platform_length = platform_length
        self.platform_width = platform_width
        self.lava_height = lava_height
        self.door_widths = door_widths
        self.door_centers = door_centers
        self.n_parts = n_parts

    def score(self, unit=1):
        return self.base_score * self.n_parts * self.lava_height

    def generate(self):
        total_platform_length = self.platform_length * (self.n_parts + 1) + self.n_parts
        platform = Block(
            f"{self.name}_Platform",
            (0, 0, total_platform_length / 2),
            (self.platform_width, PLATFORM_Y, total_platform_length)
        )
        parts = [platform]
        for n in range(self.n_parts):
            door_l = (self.door_centers[n] - 0.5) - self.door_widths[n] / (2 * self.platform_width)
            door_r = (self.door_centers[n] - 0.5) + self.door_widths[n] / (2 * self.platform_width)
            lava_r_width = max(0.1, 0.5 - door_r)
            lava_r_center = min(0.5, (door_r + 0.5) / 2)
            lava_l_width = max(0.1, 0.5 + door_l)
            lava_l_center = max(-0.5, (door_l - 0.5) / 2)
            lava_door_left = Lava(
                f"{self.name}_LavaLeft{n}",
                (lava_l_center * self.platform_width, self.lava_height / 2, self.platform_length * (n + 1) + n),
                (lava_l_width * self.platform_width, self.lava_height, 0.5)
            )
            lava_door_right = Lava(
                f"{self.name}_LavaRight{n}",
                (lava_r_center * self.platform_width, self.lava_height / 2, self.platform_length * (n + 1) + n),
                (lava_r_width * self.platform_width, self.lava_height, 0.5)
            )
            parts.append(lava_door_left)
            parts.append(lava_door_right)

        checkpoint = place_checkpoint_after(platform, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class LavaDoorsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_parts: Distribution,
            platform_length: Distribution,
            platform_width: Distribution,
            lava_height: Distribution,
            door_widths: Distribution,
            door_centers: Distribution
    ):
        self.n_parts = n_parts
        self.platform_length = platform_length
        self.platform_width = platform_width
        self.lava_height = lava_height
        self.door_widths = door_widths
        self.door_centers = door_centers

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_parts = self.n_parts.sample()
        # door_widths = [self.door_widths.sample() for _ in range(n_parts)]
        # door_centers = [self.door_centers.sample() for _ in range(n_parts)]
        door_widths = n_parts * [self.door_widths.sample()]
        door_centers = n_parts * [self.door_centers.sample()]
        for i in range(len(door_centers)):
            if i % 2 == 0:
                door_centers[i] = 1 - door_centers[i]
        return LavaDoors(
            name, position, orientation, checkpoint_name,
            n_parts,
            self.platform_length.sample(),
            self.platform_width.sample(),
            self.lava_height.sample(),
            door_widths,
            door_centers
        )
    

class WaterWheels(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            widths: list[float],
            lengths: list[float],
            angulars: list[float],
            starting_orientations: list[float],
            base_score=2
    ):
        assert len(widths) == n_blocks
        assert len(lengths) == n_blocks
        assert len(angulars) == n_blocks
        assert len(starting_orientations) == n_blocks

        super(WaterWheels, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_blocks = n_blocks
        self.widths = widths
        self.lengths = lengths
        self.angulars = angulars
        self.starting_orientations = starting_orientations

    def generate(self):
        parts = []
        
        curr_platform_z = 0
        for n in range(self.n_blocks):
            if n > 0:
                curr_platform_z += self.lengths[n - 1] / 2
            curr_platform_z += self.lengths[n] / 2
            block = Block(
                f'WaterWheel_{n}',
                position=(0, 0, curr_platform_z),
                size=(self.widths[n], PLATFORM_Y, self.lengths[n]),
                orientation=(self.starting_orientations[n], 0, 0)
            )
            rotate_behavior = behaviors.PersistentRotateBehavior(
                block.position,
                self.angulars[n],
                axis='right'
            )
            block.add_behavior(rotate_behavior)
            parts.append(block)

        # Add checkpoint.
        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts
    
class WaterWheelsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            widths: Distribution,
            lengths: Distribution,
            angulars: Distribution,
            starting_orientations: Distribution,
            uniform: bool
    ):
        self.n_blocks = n_blocks
        self.widths = widths
        self.lengths = lengths
        self.angulars = angulars
        self.starting_orientations = starting_orientations
        self.uniform = uniform

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_blocks_sample = self.n_blocks.sample()
        if self.uniform:
            widths_sample = n_blocks_sample * [self.widths.sample()]
            lengths_sample = n_blocks_sample * [self.lengths.sample()]
            angulars_sample = n_blocks_sample * [self.angulars.sample()]
            starting_orientations = n_blocks_sample * [self.starting_orientations.sample()]
        else:
            widths_sample = [self.widths.sample() for _ in range(n_blocks_sample)]
            lengths_sample = [self.lengths.sample() for _ in range(n_blocks_sample)]
            angulars_sample = [self.angulars.sample() for _ in range(n_blocks_sample)]
            starting_orientations = [self.starting_orientations.sample() for _ in range(n_blocks_sample)]

        return WaterWheels(
            name=name,
            position=position,
            orientation=orientation,
            checkpoint_name=checkpoint_name,
            widths=widths_sample,
            lengths=lengths_sample,
            angulars=angulars_sample,
            starting_orientations=starting_orientations,
        )
    

class TripleWaterWheels(WaterWheels):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            width_a: float,
            width_b: float,
            width_c: float,
            length_a: float,
            length_b: float,
            length_c: float,
            angular_a: float,
            angular_b: float,
            angular_c: float,
            starting_orientation_a: float,
            starting_orientation_b: float,
            starting_orientation_c: float,
            base_score=2
    ):
        super(TripleWaterWheels, self).__init__(
            name, 
            position, 
            orientation, 
            checkpoint_name, 
            n_blocks=3,
            widths=[width_a, width_b, width_c],
            lengths=[length_a, length_b, length_c],
            angulars=[angular_a, angular_b, angular_c],
            starting_orientations=[starting_orientation_a, starting_orientation_b, starting_orientation_c],
            base_score=base_score
        )

    def generate(self):
        parts = super(TripleWaterWheels, self).generate()
        return parts
    

class TripleWaterWheelsDistribution(ObstacleDistribution):
    def __init__(
            self,
            width_a: Distribution,
            width_b: Distribution,
            width_c: Distribution,
            length_a: Distribution,
            length_b: Distribution,
            length_c: Distribution,
            angular_a: Distribution,
            angular_b: Distribution,
            angular_c: Distribution,
            starting_orientation_a: Distribution,
            starting_orientation_b: Distribution,
            starting_orientation_c: Distribution,
    ):
        self.width_a = width_a
        self.width_b = width_b
        self.width_c = width_c
        self.length_a = length_a
        self.length_b = length_b
        self.length_c = length_c
        self.angular_a = angular_a
        self.angular_b = angular_b
        self.angular_c = angular_c
        self.starting_orientation_a = starting_orientation_a
        self.starting_orientation_b = starting_orientation_b
        self.starting_orientation_c = starting_orientation_c

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TripleWaterWheels(
            name=name,
            position=position,
            orientation=orientation,
            checkpoint_name=checkpoint_name,
            width_a=self.width_a.sample(),
            width_b=self.width_b.sample(),
            width_c=self.width_c.sample(),
            length_a=self.length_a.sample(),
            length_b=self.length_b.sample(),
            length_c=self.length_c.sample(),
            angular_a=self.angular_a.sample(),
            angular_b=self.angular_b.sample(),
            angular_c=self.angular_c.sample(),
            starting_orientation_a=self.starting_orientation_a.sample(),
            starting_orientation_b=self.starting_orientation_b.sample(),
            starting_orientation_c=self.starting_orientation_c.sample()
        )


class TrampolineCourse(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            width: float,
            length: float,
            bounce: float,
            n_parts: int,
            side_gap: float,
            length_gap: float,
            start_left: bool,
            base_score: float = 1
    ):
        super(TrampolineCourse, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.width = width
        self.length = length
        self.bounce = bounce
        self.n_parts = n_parts
        self.side_gap = side_gap  # gap from left to right
        self.length_gap = length_gap  # gap from left to left
        self.start_left = start_left

    def score(self, unit=1):
        return self.n_parts * (self.base_score * self.bounce * self.length / self.width)

    def generate(self):
        parts = []
        side = -1 if self.start_left else 1
        start_offset = self.length * 0.75  # How far the first block is from the start
        side_gap = max(self.side_gap, self.width / 2)  # left <-> right distance. Ensure no overlap
        length_gap = max(self.length_gap, 1.0)  # left <-> left distance. Ensure no overlap

        for n in range(2 * self.n_parts):
            x_pos = side_gap * side  # left or right
            z_pos = (length_gap * self.length * n) + start_offset  # distance from start

            # Tilt of the trampoline, bounce in direction of normal
            tilt = 20
            bounce_y = self.bounce * math.cos(math.radians(tilt))
            trampoline = Trampoline(
                f"{self.name}_BounceBlock{n}",
                position=(x_pos, 0.0, z_pos),
                size=(self.width, PLATFORM_Y, self.length),
                bounce=bounce_y,
                orientation=(0, 0, side * tilt)
            )

            parts.append(trampoline)
            # Change side
            side *= -1

        # Add checkpoint
        final_trampoline = parts[-1]
        checkpoint = Checkpoint(
            name=f"{self.name}_Checkpoint",
            position=(0, 0, final_trampoline.position[2] + self.length),
            size=(3, PLATFORM_Y, 3),
            orientation=(0, 0, 0)
        )
        parts.append(checkpoint)
        set_position(parts, self.position)

        return parts


class TrampolineCourseDistribution(ObstacleDistribution):
    def __init__(
            self,
            width: Distribution,
            length: Distribution,
            bounce: Distribution,
            parts: Distribution,
            side_gap: Distribution,
            length_gap: Distribution,
            start_left: Binary
    ):
        self.width = width
        self.length = length
        self.bounce = bounce
        self.parts = parts
        self.side_gap = side_gap
        self.length_gap = length_gap
        self.start_left = start_left

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TrampolineCourse(
            name=name,
            position=position,
            orientation=orientation,
            checkpoint_name=checkpoint_name,
            width=self.width.sample(),
            length=self.length.sample(),
            bounce=self.bounce.sample(),
            n_parts=self.parts.sample(),
            side_gap=self.side_gap.sample(),
            length_gap=self.length_gap.sample(),
            start_left=self.start_left.sample()
        )


class TrampolineObstacle(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            width: float,
            length: float,
            bounce: float,
            base_score: float = 1
    ):
        super(TrampolineObstacle, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.width = width
        self.length = length
        self.bounce = bounce

    def score(self, unit=1):
        return self.base_score * self.bounce * self.length / self.width

    def generate(self):
        parts = []
        trampoline = Trampoline(
            f"{self.name}_Trampoline",
            position=(0, 0, self.length / 2),
            size=(self.width, PLATFORM_Y, self.length),
            bounce=self.bounce
        )
        parts.append(trampoline)
        checkpoint = place_checkpoint_after(trampoline, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class TrampolineDistribution(ObstacleDistribution):
    def __init__(
            self,
            width: Distribution,
            length: Distribution,
            bounce: Distribution,
    ):
        self.width = width
        self.length = length
        self.bounce = bounce

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TrampolineObstacle(
            name=name,
            position=position,
            orientation=orientation,
            checkpoint_name=checkpoint_name,
            width=self.width.sample(),
            length=self.length.sample(),
            bounce=self.bounce.sample(),
        )


class TrampolineWallsObstacle(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            width: float,
            length: float,
            bounce: float,
            height_increment: float,
            platform_gap: float,
            base_score: float = 1
    ):
        super(TrampolineWallsObstacle, self).__init__(name, position, orientation, checkpoint_name,
                                                      base_score=base_score)
        self.n_blocks = n_blocks
        self.width = width
        self.length = length
        self.bounce = bounce
        self.height_increment = height_increment
        self.platform_gap = platform_gap

    def score(self, unit=1):
        return self.base_score * self.bounce * self.length / self.width

    def generate(self):
        parts = []
        # The maximum amount of height boost we can get from a trampoline
        max_height_increment = self.bounce ** 2 / GRAVITY
        height_inc = self.height_increment * max_height_increment
        for n in range(self.n_blocks):
            z_pos = self.length * n + self.length / 2
            x_pos = 1.5 * self.width * n + self.platform_gap * n
            platform = Block(
                f"{self.name}_Wall{n}",
                position=(self.width / 2, n * height_inc + self.width / 2, z_pos),
                size=(PLATFORM_Y, self.width, self.length)
            )
            trampoline = Trampoline(
                f"{self.name}_Trampoline{n}",
                position=(0, n * height_inc, z_pos),
                size=(self.width, PLATFORM_Y, self.length),
                bounce=self.bounce
            )

            parts.append(platform)
            parts.append(trampoline)

        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class TrampolineWallsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            width: Distribution,
            length: Distribution,
            bounce: Distribution,
            height_increment: Distribution,
            platform_gap: Distribution
    ):
        self.n_blocks = n_blocks
        self.width = width
        self.length = length
        self.bounce = bounce
        self.height_increment = height_increment
        self.platform_gap = platform_gap

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TrampolineWallsObstacle(
            name=name,
            position=position,
            orientation=orientation,
            checkpoint_name=checkpoint_name,
            n_blocks=self.n_blocks.sample(),
            width=self.width.sample(),
            length=self.length.sample(),
            bounce=self.bounce.sample(),
            height_increment=self.height_increment.sample(),
            platform_gap=self.platform_gap.sample()
        )


class LavaTrampoline(TrampolineObstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            width: float,
            length: float,
            bounce: float,
            n_lava_blocks: int,
            base_score: float = 2
    ):
        self.n_lava_blocks = n_lava_blocks
        super(LavaTrampoline, self).__init__(
            name, position, orientation, checkpoint_name,
            width, length, bounce, base_score=base_score
        )

    def score(self, unit=1):
        return self.base_score * self.bounce * self.n_lava_blocks * self.length / self.width

    def generate(self):
        parts = super(LavaTrampoline, self).generate()

        lava_parts = []
        lava_x = Uniform(-self.width / 4, self.width / 4)
        lava_z = Uniform(self.length / 4, 3 * self.length / 4)
        for n in range(self.n_lava_blocks):
            lava_block = Lava(
                f"{self.name}_Lava{n}",
                position=(lava_x.sample(), 0.1 * self.bounce, lava_z.sample()),
                size=(self.width / 2, PLATFORM_Y, self.width / 2),
            )
            lava_parts.append(lava_block)

        set_orientation(lava_parts, (0, 0, 0), self.orientation)
        set_position(lava_parts, self.position)

        for lava_part in lava_parts:
            parts.insert(len(parts) - 1, lava_part)

        return parts


class LavaTrampolineDistribution(ObstacleDistribution):
    def __init__(
            self,
            width: Distribution,
            length: Distribution,
            bounce: Distribution,
            n_lava_blocks: Distribution,
    ):
        self.width = width
        self.length = length
        self.bounce = bounce
        self.n_lava_blocks = n_lava_blocks

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaTrampoline(
            name, position, orientation, checkpoint_name,
            self.width.sample(),
            self.length.sample(),
            self.bounce.sample(),
            self.n_lava_blocks.sample()
        )


class WallJumps(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_walls: int,
            wall_depth: float,
            base_score: float = 10
    ):
        super(WallJumps, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_walls = n_walls
        self.wall_depth = wall_depth

    def score(self, unit=1):
        return self.base_score * self.n_walls * self.wall_depth

    def generate(self):
        wall_depth = self.wall_depth
        total_platform_length = 3 * (self.n_walls + 1) + self.n_walls * wall_depth
        platform = Block(
            f"{self.name}_Platform",
            (0, 0, total_platform_length / 2),
            (3, PLATFORM_Y, total_platform_length)
        )
        parts = [platform]
        for n in range(self.n_walls):
            wall = Block(
                f"{self.name}_Wall{n}",
                (0, 2, 3 * (n + 1) + n * wall_depth),
                (3, 5, wall_depth)
            )
            # The agent can learn to slide along the wall without this edge in place.
            safety_edge = Block(
                f"{self.name}_Edge{n}",
                (0, 2, 3 * (n + 1) + n * wall_depth + wall_depth * 0.5),
                (3.01, 5, 0.001)
            )
            parts.append(wall)
            parts.append(safety_edge)

        checkpoint = place_checkpoint_after(platform, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class WallJumpsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_walls: Distribution,
            wall_depth: Distribution,
    ):
        self.n_walls = n_walls
        self.wall_depth = wall_depth

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return WallJumps(
            name, position, orientation, checkpoint_name,
            self.n_walls.sample(), self.wall_depth.sample()
        )


class TrussObstacle(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            truss_height: float,
            base_score: float = 1
    ):
        super(TrussObstacle, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.truss_height = truss_height

    def score(self, unit=1):
        return self.base_score

    def generate(self):
        parts = []

        truss_height_sample = math.floor(self.truss_height)
        lower = Block(
            f"{self.name}_Lower",
            position=(0.0, 0.0, 1.0),
            size=(1, PLATFORM_Y, 2),
        )
        truss = Truss(
            f"{self.name}_Truss",
            position=(0, truss_height_sample / 2, 2),
            size=(1, truss_height_sample, 1)
        )
        upper = Block(
            f"{self.name}_Upper",
            position=(0.0, truss_height_sample, 3.0),
            size=(1, PLATFORM_Y, 2),
        )
        parts = [lower, truss, upper]

        checkpoint = place_checkpoint_after(upper, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class TrussDistribution(ObstacleDistribution):
    def __init__(
            self,
            truss_height: Distribution,
    ):
        self.truss_height = truss_height

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TrussObstacle(
            name, position, orientation, checkpoint_name,
            self.truss_height.sample()
        )


class TrussSiloObstacle(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            truss_height: float,
            hurdle_distance: float,
            gap_length: float,
            n_wide: int,
            n_hurdles: int,
            thickness_multiplier:float,
            base_score: float = 1
    ):
        super(TrussSiloObstacle, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.truss_height = truss_height
        self.hurdle_distance = hurdle_distance
        self.gap_length = gap_length
        self.n_hurdles = 2 * (n_hurdles  //  2) + 1
        self.thickness_multiplier = thickness_multiplier

    def score(self, unit=1):
        return self.base_score

    def generate(self):
        truss_height_sample = math.floor(self.truss_height)
        parts = []
        #n_hurdles is guaranteed to be odd
        for n_hurdle in range(self.n_hurdles):
            starting_z = n_hurdle * (self.hurdle_distance + self.gap_length + 1)
            block_height = (self.thickness_multiplier * truss_height_sample)

            mid_index = 1 + (self.n_hurdles // 2)
            local_height = 1 - (math.fabs(mid_index - (1 + n_hurdle)) / mid_index)
            block_height *= local_height
            truss = Truss(
                f"{self.name}_Truss_{n_hurdle}_{0}",
                position=(
                    0.0,
                    block_height * 0.5,
                    starting_z + self.gap_length * 0.5 + 0.5),
                size=(0.5, block_height, 0.5)
            )
            lower = Block(
                f"{self.name}_Lower_{n_hurdle}",
                position=(0.0, block_height * 0.5, starting_z + self.gap_length * 0.5 + 0.75 + self.hurdle_distance / 2),
                size=(self.hurdle_distance, block_height, self.hurdle_distance),
            )
            parts.append(truss)
            parts.append(lower)
        # Dummy part to place checkpoint
        next_lower = Block(
            f"{self.name}_NextLower",
            position=(0.0, 0.0, self.n_hurdles * (self.hurdle_distance + self.gap_length + 1) + self.hurdle_distance / 2),
            size=(0,0,0),
        )
        checkpoint = place_checkpoint_after(next_lower, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts

class TrussSiloDistribution(ObstacleDistribution):
    def __init__(
            self,
            truss_height: Distribution,
            hurdle_distance: Distribution,
            gap_length: Distribution,
            n_wide: Distribution,
            n_hurdles: Distribution,
            thickness_multiplier: Distribution
    ):
        self.truss_height = truss_height
        self.hurdle_distance = hurdle_distance
        self.gap_length = gap_length
        self.n_wide = n_wide
        self.n_hurdles = n_hurdles
        self.thickness_multiplier = thickness_multiplier

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TrussSiloObstacle(
            name, position, orientation, checkpoint_name,
            self.truss_height.sample(),
            self.hurdle_distance.sample(),
            self.gap_length.sample(),
            self.n_wide.sample(),
            self.n_hurdles.sample(),
            self.thickness_multiplier.sample()
        )


class TrussHurdlesObstacle(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            truss_height: float,
            hurdle_distance: float,
            gap_length: float,
            n_wide: int,
            n_hurdles: int,
            thickness_multiplier:float,
            base_score: float = 1
    ):
        super(TrussHurdlesObstacle, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.truss_height = truss_height
        self.hurdle_distance = hurdle_distance
        self.gap_length = gap_length
        self.n_wide = n_wide
        self.n_hurdles = n_hurdles
        self.thickness_multiplier = thickness_multiplier

    def score(self, unit=1):
        return self.base_score

    def generate(self):
        truss_height_sample = math.floor(self.truss_height)
        parts = []
        for n_hurdle in range(self.n_hurdles):
            starting_z = n_hurdle * (self.hurdle_distance + self.gap_length + 1)
            block_height = -0.5 * ((self.thickness_multiplier * 5 * truss_height_sample - 1) * PLATFORM_Y)
            lower = Block(
                f"{self.name}_Lower_{n_hurdle}",
                position=(0.0, block_height, starting_z + self.hurdle_distance / 2),
                size=(1, PLATFORM_Y * self.thickness_multiplier * 5 * truss_height_sample, self.hurdle_distance),
            )
            trusses = []
            for n in range(self.n_wide):
                truss = Truss(
                    f"{self.name}_Truss_{n_hurdle}_{n}",
                    position=(
                        0.5 * (-(self.n_wide - 1) / 2 + n),
                        (1 - self.thickness_multiplier * 0.5) * truss_height_sample,
                        starting_z + self.hurdle_distance + self.gap_length * 0.5 + 0.5),
                    size=(0.5, truss_height_sample * self.thickness_multiplier, 0.5)
                )
                trusses.append(truss)
            parts.extend([lower] + trusses)

        next_lower = Block(
            f"{self.name}_NextLower",
            position=(
                0.0, 0.0, self.n_hurdles * (self.hurdle_distance + self.gap_length + 1) + self.hurdle_distance / 2),
            size=(1, PLATFORM_Y, self.hurdle_distance),
        )
        parts.append(next_lower)

        checkpoint = place_checkpoint_after(next_lower, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class TrussHurdlesDistribution(ObstacleDistribution):
    def __init__(
            self,
            truss_height: Distribution,
            hurdle_distance: Distribution,
            gap_length: Distribution,
            n_wide: Distribution,
            n_hurdles: Distribution,
            thickness_multiplier: Distribution
    ):
        self.truss_height = truss_height
        self.hurdle_distance = hurdle_distance
        self.gap_length = gap_length
        self.n_wide = n_wide
        self.n_hurdles = n_hurdles
        self.thickness_multiplier = thickness_multiplier

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return TrussHurdlesObstacle(
            name, position, orientation, checkpoint_name,
            self.truss_height.sample(),
            self.hurdle_distance.sample(),
            self.gap_length.sample(),
            self.n_wide.sample(),
            self.n_hurdles.sample(),
            self.thickness_multiplier.sample()
        )


class WallWrapTightrope(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            height_increment: float,
            tightrope_width: float,
            wall_length: float,
            wall_width: float,
            n_turns: int,
            clockwise: bool
    ):
        super(WallWrapTightrope, self).__init__(name, position, orientation, checkpoint_name)
        self.height_increment = height_increment
        self.tightrope_width = tightrope_width
        self.wall_length = wall_length
        self.wall_width = wall_width
        self.n_turns = n_turns
        self.clockwise = clockwise

    def score(self, unit=1):
        return self.base_score * self.n_turns

    def generate(self):
        wall_height = self.height_increment * (self.n_turns + 1)
        padding = self.tightrope_width * 1.5
        starting_platform = Block(
            f"{self.name}_StartingPlatform",
            position=(0, 0, 1),
            size=(1, PLATFORM_Y, 2)
        )

        wall_pos = (0, wall_height / 2, 2 + self.wall_length / 2)
        wall = Block(
            f"{self.name}_Wall",
            position= wall_pos,
            size=(self.wall_width, wall_height, self.wall_length)
        )

        to_first_tightrope = Wedge(
            f"{self.name}_ToFirstTightrope",
            position=(0, self.height_increment / 2 + PLATFORM_Y / 2, 2 - self.tightrope_width - padding / 2),
            size=(self.tightrope_width, self.height_increment, padding),
            orientation=(0, 0, 0)
        )

        parts = [starting_platform, wall, to_first_tightrope]
        adjustment = self.tightrope_width - padding
        for i, n in enumerate(range(self.n_turns)):
            tightrope_x, _, tightrope_z = wall_pos
            tightrope_y = (i + 1) * self.height_increment
            wedge_y = tightrope_y + (PLATFORM_Y + self.height_increment) / 2

            if i % 4 == 0:
                size = (self.wall_width + adjustment, PLATFORM_Y, self.tightrope_width)
                tightrope_x -= self.tightrope_width - adjustment / 2
                tightrope_z -= (self.wall_length + self.tightrope_width) / 2
                wedge_x = -(self.wall_width + self.tightrope_width) / 2
                wedge_z = tightrope_z + self.tightrope_width / 2 + padding / 2
            elif i % 4 == 1:
                size = (self.tightrope_width, PLATFORM_Y, self.wall_length + adjustment)
                tightrope_x -= (self.wall_width + self.tightrope_width) / 2
                tightrope_z += self.tightrope_width - adjustment / 2
                wedge_x = -(self.wall_width - padding) / 2
                wedge_z = wall_pos[2] + self.wall_length / 2 + self.tightrope_width / 2
            elif i % 4 == 2:
                size = (self.wall_width + adjustment, PLATFORM_Y, self.tightrope_width)
                tightrope_x += self.tightrope_width - adjustment / 2
                tightrope_z += (self.wall_length + self.tightrope_width) / 2
                wedge_x = (self.wall_width + self.tightrope_width) / 2
                wedge_z = tightrope_z - self.tightrope_width / 2 - padding / 2
            elif i % 4 == 3:
                size = (self.tightrope_width, PLATFORM_Y, self.wall_length + adjustment)
                tightrope_x += (self.wall_width + self.tightrope_width) / 2
                tightrope_z -= self.tightrope_width - adjustment / 2
                wedge_x = (self.wall_width - padding) / 2
                wedge_z = wall_pos[2] - self.wall_length / 2 - self.tightrope_width / 2

            tightrope = Block(
                f"{self.name}_Tightrope{n}",
                position=(tightrope_x, tightrope_y, tightrope_z),
                size=size,
            )
            to_next_tightrope = Wedge(
                f"{self.name}_ToNextTightrope{n}",
                position=(wedge_x, wedge_y, wedge_z),
                size=(self.tightrope_width, self.height_increment, padding),
                orientation=(0, 90 * (i % 4), 0)
            )
            parts.append(tightrope)
            parts.append(to_next_tightrope)

        to_exit = Block(
            f"{self.name}_ToExit",
            position=(0, (self.n_turns + 1) * self.height_increment, 2 + self.wall_length),
            size=(1, PLATFORM_Y, 5)
        )

        # If clockwise, reflect all positions by the x=0 plane
        if self.clockwise:
            for part in parts:
                part.position = (-part.position[0], part.position[1], part.position[2])
                if part.orientation[1] % 180 != 0:
                    part.orientation = (part.orientation[0], part.orientation[1] + 180, part.orientation[2])

        checkpoint = place_checkpoint_after(to_exit, self.checkpoint_name)
        parts.append(to_exit)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts

class WallWrapTightropeDistribution(ObstacleDistribution):
    def __init__(
            self,
            height_increment: Distribution,
            tightrope_width: Distribution,
            wall_length: Distribution,
            wall_width: Distribution,
            n_turns: Distribution,
            clockwise: Binary
    ):
        self.height_increment = height_increment
        self.tightrope_width = tightrope_width
        self.wall_length = wall_length
        self.wall_width = wall_width
        self.n_turns = n_turns
        self.clockwise = clockwise

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return WallWrapTightrope(
            name, position, orientation, checkpoint_name,
            self.height_increment.sample(),
            self.tightrope_width.sample(),
            self.wall_length.sample(),
            self.wall_width.sample(),
            self.n_turns.sample(),
            self.clockwise.sample()
        )


class Drop(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            drop_height: float,
            base_score: float = 1
    ):
        super(Drop, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.drop_height = drop_height

    def score(self, unit=1):
        return self.base_score

    def generate(self):
        upper = Block(
            f"{self.name}_Upper",
            position=(0.0, 0.0, 1.0),
            size=(1, PLATFORM_Y, 2),
        )
        wall = Block(
            f"{self.name}_Wall",
            position=(0.0, -self.drop_height / 2, 2.0),
            size=(1, self.drop_height, PLATFORM_Y)
        )
        lower = Block(
            f"{self.name}_Lower",
            position=(0.0, -self.drop_height, 3.0),
            size=(1, PLATFORM_Y, 2),
        )

        parts = [upper, wall, lower]

        checkpoint = place_checkpoint_after(lower, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class DropDistribution(ObstacleDistribution):
    def __init__(
            self,
            drop_height: Distribution,
    ):
        self.drop_height = drop_height

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return Drop(
            name, position, orientation, checkpoint_name,
            self.drop_height.sample()
        )
    

class LavaRingDrop(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            drop_height: float,
            checkpoint_x: float, # absolute game distance offset from edge of platform
            checkpoint_y: float, # absolute game distance offset from edge of platform
            checkpoint_size: float, # expand area around checkpoint to this size
            ring_x: float, # absolute game distance offset from edge of platform
            ring_y: float, # absolute game distance offset from edge of platform
            ring_depth: float, # in [0, 1], fraction of drop_height
            ring_size: float, # ring hole side length
            ring_border: float, # outer radius, so inner hole of ring is always ring_size
            base_score: float = 1
    ):
        super(LavaRingDrop, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.drop_height = drop_height
        self.checkpoint_x = checkpoint_x
        self.checkpoint_y = checkpoint_y
        self.checkpoint_size = checkpoint_size
        self.ring_x = ring_x
        self.ring_y = ring_y
        self.ring_depth = ring_depth
        self.ring_size = ring_size
        self.ring_border = ring_border

    def score(self, unit=1):
        return self.base_score

    def generate(self):
        upper = Block(
            f"{self.name}_Upper",
            position=(0.0, 0.0, 1.0),
            size=(1, PLATFORM_Y, 2),
        )

        checkpoint = Checkpoint(
            f"{self.name}_Checkpoint",
            position=(self.checkpoint_x, -self.drop_height, self.checkpoint_y + self.checkpoint_size / 2),
            size=(self.checkpoint_size, PLATFORM_Y, self.checkpoint_size)
        )

        ring_x = checkpoint.position[0] + self.ring_x
        ring_y = checkpoint.position[2] + self.ring_y
        ring_depth = self.ring_depth * checkpoint.position[1]
        ring_center = (ring_x, ring_depth, ring_y)
        ring_r = (self.ring_border + self.ring_size) / 2

        ring_side_length = self.ring_size + 2 * self.ring_border

        ring_forward = Lava(
            f"{self.name}_RingForward",
            position=vadd(ring_center, (0, 0, ring_r)),
            size=(ring_side_length, PLATFORM_Y, self.ring_border)
        )
        ring_back = Lava(
            f"{self.name}_RingBack",
            position=vadd(ring_center, (0, 0, -ring_r)),
            size=(ring_side_length, PLATFORM_Y, self.ring_border)
        )
        ring_left = Lava(
            f"{self.name}_RingLeft",
            position=vadd(ring_center, (-ring_r, 0, 0)),
            size=(self.ring_border, PLATFORM_Y, ring_side_length)
        )
        ring_right = Lava(
            f"{self.name}_RingRight",
            position=vadd(ring_center, (ring_r, 0, 0)),
            size=(self.ring_border, PLATFORM_Y, ring_side_length)
        )

        parts = [upper, ring_forward, ring_back, ring_left, ring_right, checkpoint]

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts
    

class LavaRingDropDistribution(ObstacleDistribution):
    def __init__(
            self,
            drop_height: Distribution,
            checkpoint_x: Distribution, # absolute game distance offset from edge of platform
            checkpoint_y: Distribution, # absolute game distance offset from edge of platform
            checkpoint_size: Distribution, # expand area around checkpoint to this size
            ring_x: Distribution, # absolute game distance offset from edge of platform
            ring_y: Distribution, # absolute game distance offset from edge of platform
            ring_depth: Distribution, # in [0, 1], fraction of drop_height
            ring_size: Distribution, # ring hole side length
            ring_border: Distribution, # outer radius, so inner hole of ring is always ring_size
    ):
        self.drop_height = drop_height
        self.checkpoint_x = checkpoint_x
        self.checkpoint_y = checkpoint_y
        self.checkpoint_size = checkpoint_size
        self.ring_x = ring_x
        self.ring_y = ring_y
        self.ring_depth = ring_depth
        self.ring_size = ring_size
        self.ring_border = ring_border

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaRingDrop(
            name, position, orientation, checkpoint_name,
            self.drop_height.sample(),
            self.checkpoint_x.sample(),
            self.checkpoint_y.sample(),
            self.checkpoint_size.sample(),
            self.ring_x.sample(),
            self.ring_y.sample(),
            self.ring_depth.sample(),
            self.ring_size.sample(),
            self.ring_border.sample(),
        )


class LavaCheckerboard(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            rows: int,
            checkerboard_size: float,
            base_score: float = 2
    ):
        super(LavaCheckerboard, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.rows = rows
        self.checkerboard_size = checkerboard_size

    def score(self, unit=1):
        return self.base_score * 1.5 * self.rows

    def generate(self, return_solution: bool = False):
        parts = []
        waypoints = []
        platform = Block(
            f"{self.name}_Block",
            position=(0, 0, (self.checkerboard_size * self.rows + 2) / 2),
            size=(3 * self.checkerboard_size, PLATFORM_Y, (self.checkerboard_size * self.rows + 2))
        )
        parts.append(platform)

        l_idx = 0
        for r in range(self.rows):
            if r % 2 == 0:
                # Place two lava blocks.
                lava_block_l = Lava(
                    f"{self.name}_Lava{l_idx}",
                    position=(
                        -self.checkerboard_size,
                        PLATFORM_Y,
                        (self.checkerboard_size * r + 1 + self.checkerboard_size / 2)
                    ),
                    size=(self.checkerboard_size, PLATFORM_Y, self.checkerboard_size)
                )
                lava_block_r = Lava(
                    f"{self.name}_Lava{l_idx + 1}",
                    position=(
                        self.checkerboard_size,
                        PLATFORM_Y,
                        (self.checkerboard_size * r + 1 + self.checkerboard_size / 2)
                    ),
                    size=(self.checkerboard_size, PLATFORM_Y, self.checkerboard_size)
                )
                l_idx += 2
                parts.append(lava_block_l)
                parts.append(lava_block_r)

                waypoint_position = vscale(0.5, vadd(lava_block_l.position, lava_block_r.position))
                waypoint_action = Action.WALK if r == 0 else Action.JUMP
                waypoints.append(Waypoint(waypoint_position, waypoint_action))

                waypoint_position = vscale(0.5, vadd(lava_block_l.position, lava_block_r.position))
                waypoint_position = vadd(waypoint_position,
                                         (0.2 * self.checkerboard_size, 0, 0.2 * self.checkerboard_size))
                waypoint_action = Action.WALK
                waypoints.append(Waypoint(waypoint_position, waypoint_action))

            else:
                lava_block_m = Lava(
                    f"{self.name}_lava{l_idx}",
                    position=(0, PLATFORM_Y, (self.checkerboard_size * r + 1 + self.checkerboard_size / 2)),
                    size=(self.checkerboard_size, PLATFORM_Y, self.checkerboard_size)
                )
                l_idx += 1
                parts.append(lava_block_m)
                waypoint_position = vadd(lava_block_m.position, (self.checkerboard_size, 0, 0))
                waypoint_action = Action.JUMP
                waypoints.append(Waypoint(waypoint_position, waypoint_action))

                waypoint_position = vadd(lava_block_m.position, (0.8 * self.checkerboard_size, 0, 0.2 * self.checkerboard_size))
                waypoint_action = Action.WALK
                waypoints.append(Waypoint(waypoint_position, waypoint_action))

        checkpoint = place_checkpoint_after(platform, self.checkpoint_name)
        parts.append(checkpoint)
        waypoints.append(Waypoint(checkpoint.position, Action.JUMP))

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_orientation(waypoints, (0, 0, 0), self.orientation)
        set_position(parts, self.position)
        set_position(waypoints, self.position)

        if return_solution:
            return parts, waypoints
        else:
            return parts


class LavaCheckerboardDistribution(ObstacleDistribution):
    def __init__(
            self,
            rows: Distribution,
            checkerboard_size: Distribution,
    ):
        self.rows = rows
        self.checkerboard_size = checkerboard_size

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaCheckerboard(
            name, position, orientation, checkpoint_name,
            self.rows.sample(),
            self.checkerboard_size.sample()
        )


class Wedges(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_wedges: int,
            wedge_length: float,
            wedge_height: float,
            base_score: float = 1
    ):
        super(Wedges, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_wedges = n_wedges
        self.wedge_length = wedge_length
        self.wedge_height = wedge_height

    def score(self, unit=1):
        return self.base_score

    def generate(self):
        parts = []
        for n in range(self.n_wedges):
            wedge_down = Wedge(
                f"{self.name}_WedgeUp{n}",
                position=(0, self.wedge_height / 2, self.wedge_length / 2 + 2 * n * self.wedge_length),
                size=(2, self.wedge_height, self.wedge_length),
                orientation=(0, 0, 0)
            )
            wedge_up = Wedge(
                f"{self.name}_WedgeDown{n}",
                position=(0, self.wedge_height / 2, 1.5 * self.wedge_length + 2 * n * self.wedge_length),
                size=(2, self.wedge_height, self.wedge_length),
                orientation=(0, 180, 0)
            )
            parts.append(wedge_down)
            parts.append(wedge_up)

        checkpoint = place_checkpoint_after(wedge_down, self.checkpoint_name)
        checkpoint.position = vadd(checkpoint.position, (0, -self.wedge_height / 2, self.wedge_length))
        # checkpoint.orientation = (0, 0, 0)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class WedgesDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_wedges: Distribution,
            wedge_length: Distribution,
            wedge_height: Distribution
    ):
        self.n_wedges = n_wedges
        self.wedge_length = wedge_length
        self.wedge_height = wedge_height

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return Wedges(
            name, position, orientation, checkpoint_name,
            self.n_wedges.sample(),
            self.wedge_length.sample(),
            self.wedge_height.sample()
        )


class LavaWedges(Wedges):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_wedges: int,
            wedge_length: float,
            wedge_height: float,
            base_score: float = 1
    ):
        super(LavaWedges, self).__init__(
            name, position, orientation, checkpoint_name,
            n_wedges, wedge_length, wedge_height, base_score=base_score
        )

    def score(self):
        return self.base_score + 1

    def generate(self):
        parts = super(LavaWedges, self).generate()
        for i, part in enumerate(parts):
            if isinstance(part, Wedge):
                lava_block = Lava(
                    f'{self.name} Lava Block {i}',
                    position=part.position,
                    size=(part.size[0], 0.25, 0.25),
                    orientation=part.orientation
                )
                parts.insert(len(parts) - 1, lava_block)
        return parts


class LavaWedgesDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_wedges: Distribution,
            wedge_length: Distribution,
            wedge_height: Distribution
    ):
        self.n_wedges = n_wedges
        self.wedge_length = wedge_length
        self.wedge_height = wedge_height

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        return LavaWedges(
            name, position, orientation, checkpoint_name,
            self.n_wedges.sample(),
            self.wedge_length.sample(),
            self.wedge_height.sample()
        )


class SideToSideBlocks(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_blocks: int,
            velocities: list[float],
            gap_length: float,
            amplitudes: list[float],
            base_score: float = 4
    ):
        super(SideToSideBlocks, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_blocks = n_blocks
        self.velocities = velocities
        self.gap_length = gap_length
        self.amplitudes = amplitudes

    def score(self, unit=1):
        if self.gap_length * unit > MAX_JUMP_LENGTH:
            return math.inf

        total_times = 0
        for i, velocity in enumerate(self.velocities):
            total_times += self.amplitudes[i] / velocity

        total_amplitude = 0
        for amp in self.amplitudes:
            total_amplitude += abs(amp)

        return self.base_score * self.n_blocks * self.gap_length * total_amplitude / total_times

    def generate(self):
        parts = []
        for n in range(self.n_blocks):
            block = MovingBlock(
                f"{self.name}_Block{n}",
                position=(0, 0, 0.5 + n * (1 + self.gap_length)),
                size=(1, PLATFORM_Y, 1),
                motion_type='Persistent',
                # start_position=(-self.amplitudes[n], 0, 0.5 + n * (1 + self.gap_length)),
                # end_position=(self.amplitudes[n], 0, 0.5 + n * (1 + self.gap_length)),
                start_position=(abs(self.amplitudes[n]) * (-1)**n, 0, 0),
                end_position=(abs(self.amplitudes[n]) * (-1)**(n + 1), 0, 0),
                time=abs(self.amplitudes[n]) / self.velocities[n],
                easing_style='Linear'
            )
            parts.append(block)

        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class SideToSideBlocksDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_blocks: Distribution,
            velocities: Distribution,
            gap: Distribution,
            amplitudes: Distribution,
    ):
        self.n_blocks = n_blocks
        self.velocities = velocities
        self.gap = gap
        self.amplitudes = amplitudes

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_blocks_sample = self.n_blocks.sample()
        velocities_samples = [self.velocities.sample() for _ in range(n_blocks_sample)]
        gap_sample = self.gap.sample()
        amplitude_samples = [self.amplitudes.sample() for _ in range(n_blocks_sample)]

        return SideToSideBlocks(
            name, position, orientation, checkpoint_name,
            n_blocks_sample,
            velocities_samples,
            gap_sample,
            amplitude_samples
        )


class HiddenStairs(Obstacle):
    """To create, specify directions as a string of 'url' where 'u' indicates moving forward,
    'r' indicates moving positive local x and 'l' indicates moving negative local x. Make sure
    that there are no consecutive lr."""

    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            directions: str,
            gap_height: float,
            base_score: float = 1,
    ):
        super(HiddenStairs, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.directions = directions
        self.gap_height = gap_height

    def score(self):
        # Determine the tightest grid that is needed to fit the path.
        start = (0, 0)
        min_x = math.inf
        max_x = -math.inf
        for direction in self.directions:
            if direction == 'u':
                start[0] += 1
            elif direction == 'r':
                start[1] += 1
            elif direction == 'l':
                start[1] -= 1
            min_x = min(min_x, start[1])
            max_x = max(max_x, start[1])
        return start[0] * (max_x - min_x)

    def generate(self):
        # print("directions:", self.directions)
        start = [0, -1]
        min_x = math.inf
        max_x = -math.inf
        path_positions = []
        for direction in self.directions:
            if direction == 'u':
                start[1] += 1
            elif direction == 'r':
                start[0] += 1
            elif direction == 'l':
                start[0] -= 1
            min_x = min(min_x, start[0])
            max_x = max(max_x, start[0])
            path_positions.append(start.copy())

        n_blocks_width = max_x - min_x + 1
        n_blocks_forward = start[1] + 1
        block_idxs = [[0 for _ in range(n_blocks_width)] for _ in range(n_blocks_forward)]
        # print(block_idxs)

        # print(min_x, max_x)
        pos = [-1, -min_x]
        for i, direction in enumerate(self.directions):
            # print(pos[0], pos[1])
            if direction == 'u':
                pos[0] += 1
            elif direction == 'r':
                pos[1] += 1
            elif direction == 'l':
                pos[1] -= 1
            block_idxs[pos[0]][pos[1]] = i + 1
            # print(block_idxs)

        start_block = Block(
            f'{self.name} Start Block',
            position=(0, 0, 0.5),
            size=(n_blocks_width, PLATFORM_Y, 1),
            orientation=(0, 0, 0)
        )
        parts = [start_block]
        i = 0
        for c in range(n_blocks_width):
            for r in range(n_blocks_forward):
                # if r == path_positions[i][0] and c == path_positions[i][1]:
                if block_idxs[r][c] > 0:
                    block = MovingBlock(
                        f'{self.name} Block({c},{r})',
                        position=(-(n_blocks_width - 1) / 2 + + c, 0, 1.5 + r),
                        size=(1, PLATFORM_Y, 1),
                        motion_type='Touch',
                        start_position=(0, 0, 0),
                        end_position=(0, self.gap_height * block_idxs[r][c] / len(self.directions), 0),
                        time=11,
                        easing_style='Linear',
                        reverses=False,
                        loop=0
                    )
                    i += 1
                else:
                    block = Block(
                        f'{self.name} Block({c},{r})',
                        position=(-(n_blocks_width - 1) / 2 + c, 0, 1.5 + r),
                        size=(1, PLATFORM_Y, 1),
                        orientation=(0, 0, 0)
                    )
                parts.append(block)
        end_block = Block(
            f'{self.name} End Block',
            position=(0, self.gap_height, 1.5 + n_blocks_forward),
            size=(n_blocks_width, PLATFORM_Y, 1),
            orientation=(0, 0, 0)
        )
        parts.append(end_block)

        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class HiddenStairsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_turns: Distribution,
            turns: Distribution,
            gap: Distribution,
    ):
        self.n_turns = n_turns
        self.turns = turns
        self.gap = gap

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_turns_sample = self.n_turns.sample()
        directions_sample = generate_pretty_url_directions(n_turns_sample, self.turns)
        gap_sample = self.gap.sample()

        return HiddenStairs(
            name, position, orientation, checkpoint_name,
            directions_sample,
            gap_sample,
        )


class BlockPush(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            platform_width: float,
            platform_length: float,
            block_size: VectorType,
            block_position: VectorType,
            base_score: float = 1
    ):
        super(BlockPush, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.block_size = block_size
        self.block_position = block_position

    def score(self):
        return self.base_score

    def generate(self):
        platform = Block(
            f"{self.name}_Platform",
            position=(0, 0, self.platform_length / 2),
            size=(self.platform_width, PLATFORM_Y, self.platform_length),
        )
        unanchored_block = UnanchoredBlock(
            name=f"{self.name}",
            position=self.block_position,
            size=self.block_size
        )
        half_wall_height = self.block_size[1] + 0.5
        back_wall = Block(
            name=f"{self.name}_BackWall",
            position=(0, half_wall_height, self.platform_length),
            size=(self.platform_width, 2 * half_wall_height, PLATFORM_Y),
        )
        outway = Block(
            f"{self.name}_Outway",
            position=(0, 2 * half_wall_height - PLATFORM_Y * 0.5, self.platform_length + self.platform_width),
            size=(self.platform_width, PLATFORM_Y, self.platform_width * 2)
        )
        parts = [platform, unanchored_block, back_wall, outway]

        checkpoint = place_checkpoint_after(parts[-1], self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class BlockPushDistribution(ObstacleDistribution):
    def __init__(
            self,
            platform_width: Distribution,
            platform_length: Distribution,
            block_size: Distribution,
            block_position: Distribution,
    ):
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.block_size = block_size
        self.block_position = block_position

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        platform_width_sample = self.platform_width.sample()
        platform_length_sample = self.platform_length.sample()
        block_size_sample = self.block_size.sample()
        block_position = (
            (2 * self.block_position.sample() - 1) * (platform_width_sample - 2 * block_size_sample),
            block_size_sample / 2 + PLATFORM_Y,
            self.block_position.sample() * (platform_length_sample - 2 * block_size_sample)
        )
        return BlockPush(
            name, position, orientation, checkpoint_name,
            platform_width_sample,
            platform_length_sample,
            (block_size_sample, block_size_sample, block_size_sample),
            block_position,
        )


class ButtonHoldSpinner(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            platform_width: float,
            platform_length: float,
            platform_orientation: float,
            angular: float,
            base_score: float = 1
    ):
        super(ButtonHoldSpinner, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.platform_orientation = platform_orientation
        self.angular = angular

    def score(self):
        return self.base_score

    def generate(self):
        start = Block(
            f"{self.name}_Start",
            position=(0, 0, 0.5),
            size=(1, PLATFORM_Y, 1)
        )
        button = HoldButton(
            f"{self.name}_Button",
            position=(0, 0, 1.5),
            size=(1, PLATFORM_Y, 1),
        )
        rotating_plank = Block(
            f"{self.name}_Plank",
            position=(0, 0, 3 + self.platform_length / 2),
            size=(self.platform_width, PLATFORM_Y, self.platform_length),
            orientation=(0, self.platform_orientation, 0)
        )
        rotate_behavior = PersistentRotateBehavior(
            rotating_plank.position,
            angular=self.angular,
        )
        rotate_behavior.add_condition(button.state)
        rotating_plank.add_behavior(rotate_behavior)
        parts = [start, button, rotating_plank]

        checkpoint = Checkpoint(
            name=self.checkpoint_name,
            position=(0, 0, self.platform_length + 5.5),
            size=(1.5, PLATFORM_Y, 1.5)
        )
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class ButtonHoldSpinnerDistribution(ObstacleDistribution):
    def __init__(
            self,
            platform_width: Distribution,
            platform_length: Distribution,
            platform_orientation: Distribution,
            angular: Distribution,
    ):
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.platform_orientation = platform_orientation
        self.angular = angular

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        platform_width_sample = self.platform_width.sample()
        platform_length_sample = self.platform_length.sample()
        platform_orientation_sample = self.platform_orientation.sample()
        angular_sample = self.angular.sample()
        return ButtonHoldSpinner(
            name, position, orientation, checkpoint_name,
            platform_width_sample,
            platform_length_sample,
            platform_orientation_sample,
            angular_sample
        )


class DextrousButtonPressWall(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            platform_width: float,
            platform_length: float,
            button_position: VectorType,
            base_score: float = 1
    ):
        super(DextrousButtonPressWall, self).__init__(name, position, orientation, checkpoint_name,
                                                      base_score=base_score)
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.button_position = button_position

    def score(self):
        return self.base_score

    def generate(self):
        start = Block(
            f"{self.name}_Start",
            position=(0, 0, 0.5),
            size=(1, PLATFORM_Y, 1)
        )

        # Insert SideToSideBlocks here
        side_to_side_parts = SideToSideBlocks(
            "SideToSideToButton",
            position=(1.5, 0, 0),
            orientation=(0, 90, 0) + self.orientation,
            checkpoint_name="null",  # we'll delete this checkpoint below.
            n_blocks=4,
            velocities=[1.5, 1, 1.25, 1],
            gap_length=1,
            amplitudes=[3, 2, 5, 2]
        ).generate()

        # Grab the checkpoint off the part.
        # ckpt = side_to_side_parts[-1]

        # button_platform = Block(
        #     name=ckpt.name, # Keep the same name as the checkpoint so the observations work out for the agent.
        #     position=ckpt.position,
        #     size=ckpt.size,
        #     orientation=ckpt.orientation
        # )

        # side_to_side_parts[-1] = button_platform

        button_platform = side_to_side_parts[-1]

        button = PressButton(
            f"{self.name}_Button",
            position=(button_platform.position[0], self.button_position[1], button_platform.position[2]),
            size=(1, PLATFORM_Y, 1),
        )
        plank = Block(
            f"{self.name}_Plank",
            position=(0, 0, 1 + self.platform_length / 2),
            size=(self.platform_width, PLATFORM_Y, self.platform_length),
            orientation=(0, 0, 0)
        )
        wall = Block(
            f"{self.name}_Wall",
            position=(0, 2.5, self.platform_length + 1.25),
            size=(self.platform_width, 5, 0.5),
            orientation=(0, 0, 0)
        )
        move_behavior = PersistentTranslateBehavior(
            start_position=(0, 0, 0),
            end_position=(0, -wall.size[1], 0),
            loop=0,
            reverses=False,
        )
        move_behavior.add_condition(button.state)
        wall.add_behavior(move_behavior)

        parts = [start] + side_to_side_parts + [button, plank, wall]

        checkpoint = place_checkpoint_after(plank, self.checkpoint_name)
        checkpoint.position = vadd(checkpoint.position, (0, 0, 0.5))
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class DextrousButtonPressWallDistribution(ObstacleDistribution):
    def __init__(
            self,
            platform_width: Distribution,
            platform_length: Distribution,
            button_position: Distribution,
    ):
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.button_position = button_position

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        platform_width_sample = self.platform_width.sample()
        platform_length_sample = self.platform_length.sample()
        button_position_sample = ((0.5 - self.button_position.sample()) * platform_width_sample, PLATFORM_Y,
                                  self.button_position.sample() * platform_length_sample + 1)
        return DextrousButtonPressWall(
            name, position, orientation, checkpoint_name,
            platform_width_sample,
            platform_length_sample,
            button_position_sample,
        )


class OneButtonPressWall(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            platform_width: float,
            platform_length: float,
            button_position: VectorType,
            base_score: float = 1
    ):
        super(OneButtonPressWall, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.button_position = button_position

    def score(self):
        return self.base_score

    def generate(self):
        start = Block(
            f"{self.name}_Start",
            position=(0, 0, 0.5),
            size=(1, PLATFORM_Y, 1)
        )
        button = PressButton(
            f"{self.name}_Button",
            position=self.button_position,
            size=(1, PLATFORM_Y, 1),
        )
        plank = Block(
            f"{self.name}_Plank",
            position=(0, 0, 1 + self.platform_length / 2),
            size=(self.platform_width, PLATFORM_Y, self.platform_length),
            orientation=(0, 0, 0)
        )
        wall = Block(
            f"{self.name}_Wall",
            position=(0, 2.5, self.platform_length + 1.25),
            size=(self.platform_width, 5, 0.5),
            orientation=(0, 0, 0)
        )
        move_behavior = PersistentTranslateBehavior(
            start_position=(0, 0, 0),
            end_position=(0, -wall.size[1], 0),
            loop=0,
            reverses=False,
        )
        move_behavior.add_condition(button.state)
        wall.add_behavior(move_behavior)

        parts = [start, button, plank, wall]

        checkpoint = place_checkpoint_after(plank, self.checkpoint_name)
        checkpoint.position = vadd(checkpoint.position, (0, 0, 0.5))
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class OneButtonPressWallDistribution(ObstacleDistribution):
    def __init__(
            self,
            platform_width: Distribution,
            platform_length: Distribution,
            button_position: Distribution,
    ):
        self.platform_width = platform_width
        self.platform_length = platform_length
        self.button_position = button_position

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        platform_width_sample = self.platform_width.sample()
        platform_length_sample = self.platform_length.sample()
        button_position_sample = ((0.5 - self.button_position.sample()) * platform_width_sample, PLATFORM_Y,
                                  self.button_position.sample() * platform_length_sample + 1)
        return OneButtonPressWall(
            name, position, orientation, checkpoint_name,
            platform_width_sample,
            platform_length_sample,
            button_position_sample,
        )


class PathVsLilypads(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_pads: int,
            gap_length: float,
            base_score: float = 1
    ):
        super(PathVsLilypads, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
        self.n_pads = n_pads
        self.gap_length = gap_length

    def score(self):
        return self.base_score

    def generate(self):
        start = Block(
            f'{self.name}_Start',
            position=(0, 0, 0.5),
            size=(1, PLATFORM_Y, 1)
        )
        split = Block(
            f'{self.name}_Split',
            position=(0, 0, 1.5),
            size=(3, PLATFORM_Y, 1)
        )
        parts = [start, split]
        lilypads = Lilypads(self.name, (1, 0, 2), (0, 0, 0), "Checkpoint", Block, self.n_pads, self.gap_length)
        lilypad_parts = lilypads.generate()[:-1]
        parts.extend(lilypad_parts)
        path_length = self.n_pads + (self.n_pads - 1) * self.gap_length
        path = Block(
            f'{self.name}_Path',
            position=(-1, 0, path_length / 2 + 2),
            size=(1, PLATFORM_Y, path_length)
        )
        end = Block(
            f'{self.name}_End',
            position=(0, 0, path_length + 2.5),
            size=(1, PLATFORM_Y, 3)
        )
        parts.append(path)
        parts.append(end)

        checkpoint = place_checkpoint_after(end, self.checkpoint_name)
        parts.append(checkpoint)

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)

        return parts


class PathVsLilypadsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_pads: Distribution,
            gap_length: Distribution,
    ):
        self.n_pads = n_pads
        self.gap_length = gap_length

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_pads_sample = self.n_pads.sample()
        gap_length_sample = self.gap_length.sample()
        return PathVsLilypads(
            name, position, orientation, checkpoint_name,
            n_pads_sample,
            gap_length_sample
        )


class ConveyorVsLilypads(PathVsLilypads):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_pads: int,
            gap_length: float,
            velocity: float,
            base_score: float = 1
    ):
        super(ConveyorVsLilypads, self).__init__(name, position, orientation, checkpoint_name, n_pads, gap_length,
                                                 base_score=base_score)
        self.velocity = velocity

    def score(self):
        return self.base_score

    def generate(self):
        parts = super(ConveyorVsLilypads, self).generate()
        parts[-3].add_behavior(
            behaviors.ConveyorBehavior(0, self.velocity)
        )
        return parts


class ConveyorVsLilypadsDistribution(ObstacleDistribution):
    def __init__(
            self,
            n_pads: Distribution,
            gap_length: Distribution,
            velocity: Distribution
    ):
        self.n_pads = n_pads
        self.gap_length = gap_length
        self.velocity = velocity

    def sample(self, name, checkpoint_name, position=(0, 0, 0), orientation=(0, 0, 0)):
        n_pads_sample = self.n_pads.sample()
        gap_length_sample = self.gap_length.sample()
        velocity_sample = self.velocity.sample()
        return ConveyorVsLilypads(
            name, position, orientation, checkpoint_name,
            n_pads_sample,
            gap_length_sample,
            velocity_sample
        )


# class PlankVsConveyor(ObstacleDistribution):
#     def __init__(
#         self,
#         name: str,
#         position: VectorType, 
#         orientation: VectorType,
#         checkpoint_name: str,
#     )


# class TwoButtonPressWall(Obstacle):
#     def __init__(
#         self,
#         name: str,
#         position: VectorType, 
#         orientation: VectorType,
#         checkpoint_name: str,
#         platform_width: float,
#         platform_length: float,
#         up_button_position: float,
#         down_button_position: float,
#         base_score: float = 1
#     ):
#         super(TwoButtonPressWall, self).__init__(name, position, orientation, checkpoint_name, base_score=base_score)
#         self.platform_width = platform_width
#         self.platform_length = platform_length


# class BlockPushWithHole(Obstacle):
#     def __init__(
#         self,
#         name: str,
#         position: VectorType, 
#         orientation: VectorType,
#         checkpoint_name: str,
#         platform_size: VectorType,
#         block_location: VectorType,
#     ):
#         pass


# RANDOM


#### TEST
class CubeTest(Obstacle):
    def __init__(
            self,
            x: float,
            y: float,
            z: float,
            w: float,
            l: float,
            h: float,
    ):
        self.position = (x, y, z)
        self.size = (w, l, h)
        self.orientation = 0
        self.parts = []

    def __str__(self) -> str:
        rep_str = ""
        for part in self.parts:
            rep_str += str(part)
        return rep_str

    def generate(self) -> list[Part]:
        return make_cube(self.position, self.size, self.orientation)


class Cubes(Obstacle):
    def __init__(
            self,
            name: str,
            position: VectorType,
            orientation: VectorType,
            checkpoint_name: str,
            n_cubes: int,
            w: float,
            l: float,
            h: float,
            gap: float,
    ):
        self.name = name
        self.position = position
        self.orientation = orientation
        self.checkpoint_name = checkpoint_name

        self.n_cubes = n_cubes
        self.size = (w, h, l)
        self.gap = gap

    def __str__(self) -> str:
        rep_str = ""
        for part in self.parts:
            rep_str += str(part)
        return rep_str

    def generate(self) -> list[Part]:
        parts = []
        for n in range(self.n_cubes):
            parts.append(
                Block(
                    f"Cube {n}",
                    position=vadd(self.position, (0, 0, n * (self.gap + self.size[2]))),
                    size=self.size,
                    orientation=(0.0, 0.0, 0.0)
                )
            )

        set_orientation(parts, (0, 0, 0), self.orientation)
        set_position(parts, self.position)
        return parts
