# Basic part expressions.
# TODO:
# - add orientations
# - add basic parts like truss and wedges

from __future__ import annotations

import random
import typing as tp
from dataclasses import dataclass, field

import obby.grammar.behaviors as behaviors
from obby.grammar.behaviors import NoneState
from obby.grammar.utils import make_cube, orient
from obby.utils.vector import *
from obby.grammar.type_aliases import *
from obby.grammar.distribution import Distribution


@dataclass
class Part:
    name: str
    position: Vector
    size: Vector
    orientation: Vector = (0, 0, 0)
    part_type: str = "Part"
    behaviors: list[behaviors.Behavior] = field(default_factory=lambda: [])
    state: behaviors.State = field(default_factory=lambda: NoneState())
    color: Vector = (0.5, 0.5, 0.5)

    def add_behavior(self, behavior: behaviors.Behavior):
        self.behaviors.append(behavior)

    # NOTE: Relative jitter.
    def randomize(self, position_sampler: Distribution, size_sampler: Distribution, orientation_sampler: Distribution, randomize_behaviors: bool = False):
        position_jitter = (position_sampler.sample(), position_sampler.sample(), position_sampler.sample())
        size_jitter = (size_sampler.sample(), size_sampler.sample(), size_sampler.sample())
        orientation_jitter = (orientation_sampler.sample(), orientation_sampler.sample(), orientation_sampler.sample())
        self.position = vadd(self.position, position_jitter)
        self.size = vadd(self.size, size_jitter)
        self.orientation = vadd(self.orientation, orientation_jitter)

        if randomize_behaviors:
            for behavior in self.behaviors:
                behavior.randomize(position_sampler, size_sampler, orientation_sampler)


    def __str__(self) -> str:
        rep_str = f"{self.part_type} {self.name}\n"
        rep_str += f"  Position: {self.position}\n"
        rep_str += f"  Size: {self.size}\n"
        rep_str += f"  Orientation: {self.orientation}\n"
        rep_str += f"  State: {self.state}\n"
        rep_str += f"  Behaviors:\n"
        for behavior in self.behaviors:
            rep_str += f"    {behavior}\n"
        return rep_str
    
    def jsonify(self, unit: float = 1) -> tuple[str, dict[str, tp.Any]]:
        return self.name, {
            "type": self.part_type,
            "class": self.__class__.__name__,
            "position": (unit * self.position[0], unit * self.position[1], unit * self.position[2]),
            "size": (unit * self.size[0], unit * self.size[1], unit * self.size[2]),
            "orientation": self.orientation,
            "state": self.state.jsonify(),
            "behaviors": [behavior.jsonify(unit=unit) for behavior in self.behaviors],
        }
    
    def geometry(self, unit: float = 1, v_offset: int = 0):
        v, f, c = make_cube(
            position=vscale(unit, self.position),
            size=vscale(unit, self.size),
            color=self.color,
            orientation=self.orientation,
            v_offset=v_offset
        )
        return v, f, c
    
    def sample_point_above(self, unit: float = 1, above=0.2) -> Vector:
        # first, sample a random point above in canonical space.
        random_factor = (random.random() - 0.5, 0.5, random.random() - 0.5)
        random_point = vmul(random_factor, self.size)
        random_point = orient(random_point, self.orientation)
        random_point = vadd(random_point, self.position)
        random_point = vadd(random_point, (0, above, 0)) # Raise it above the ground a little
        return random_point
    
    def is_safe(self) -> bool:
        for behavior in self.behaviors:
            if isinstance(behavior, behaviors.DamageBehavior):
                return False
        return True

################################################################################
# HIGHER-LEVEL PARTS
################################################################################

class Block(Part):
    """A flat platform block."""
    part_type = "Block"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Block, self).__init__(name, position, size, orientation, part_type="Block")


class Cylinder(Part):
    """A cylinder block."""
    part_type = "Cylinder"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Cylinder, self).__init__(name, position, size, orientation, part_type="Cylinder")


class Wall(Part):
    """A tall wall."""
    part_type = "Block"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Wall, self).__init__(name, position, size, orientation, part_type="Block")


class Truss(Part):
    """A truss part."""
    part_type = "TrussPart"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Truss, self).__init__(name, position, size, orientation, behaviors=[behaviors.TrussBehavior()], part_type="TrussPart")


class Wedge(Part):
    """A wedge part."""
    part_type = "Wedge"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Wedge, self).__init__(name, position, size, orientation, part_type="Wedge")

class Ball(Part):
    """A sphere part."""
    part_type = "Ball"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Ball, self).__init__(name, position, size, orientation, part_type="Ball")
    

class SpawnLocation(Part):
    """A spawn location block."""
    part_type = "SpawnLocation"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(SpawnLocation, self).__init__(name, position, size, orientation=orientation, behaviors=[behaviors.SpawnBehavior()], part_type="SpawnLocation")
        self.color = (0.0, 1.0, 0.0)


class Checkpoint(Part):
    """A checkpoint block."""
    part_type = "Part"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Checkpoint, self).__init__(name, position, size, orientation=orientation, behaviors=[behaviors.CheckpointBehavior()])
        self.color = (0.0, 0.0, 1.0)


class Goal(Part):
    """A goal block."""
    part_type = "Part"

    def __init__(self, name, position, size, orientation=(0, 0, 0)):
        super(Goal, self).__init__(name, position, size, orientation=orientation, behaviors=[behaviors.CheckpointBehavior()])
        self.color = (0.0, 0.0, 1.0)
    

class Lava(Part):
    """A lava block."""
    part_type = "Part"
    # behaviors = None

    def __init__(self, name, position, size, orientation=(0, 0, 0), damage=100.0):
        super(Lava, self).__init__(name, position, size, orientation=orientation)
        self.color = (1.0, 0.0, 0.0)
        self.add_behavior(behaviors.DamageBehavior(amount=damage))


class Conveyor(Part):
    """A conveyor block."""
    part_type = "Part"
    # behaviors = field(default_factory=lambda: [])

    def __init__(self, name, position, size, orientation=(0, 0, 0), velocity: Vector = (1, 0, 0)):
        super(Conveyor, self).__init__(name, position, size, orientation=orientation)
        self.add_behavior(behaviors.ConveyorBehavior(x_velocity=velocity[0], z_velocity=velocity[2]))
        self.color = (0.75, 0.0, 0.75)


class Trampoline(Part):
    """A trampoline block."""
    part_type = "Part"

    def __init__(self, name, position, size, orientation=(0, 0, 0), bounce: float = 10):
        super(Trampoline, self).__init__(name, position, size, orientation=orientation)
        self.add_behavior(behaviors.TrampolineBehavior(bounce=bounce))
        self.color = (0.75, 0.0, 0.75)


class DisappearBlock(Part):
    """A block which disappears when touched and reappears after some time."""
    part_type = "Part"

    def __init__(self, name, position, size, orientation=(0, 0, 0), disappear_delay: float = 10, reappear_delay: float = 10):
        super(DisappearBlock, self).__init__(name, position, size, orientation=orientation)
        self.add_behavior(
            behaviors.DisappearBehavior(
                disappear_delay=disappear_delay, 
                reappear_delay=reappear_delay
            )
        )

class MovingBlock(Part):
    """A block that moves between two points."""
    part_type = "Part"

    def __init__(
        self, 
        name, 
        position, 
        size, 
        motion_type: tp.Literal['Persistent', 'Touch'],
        start_position: Vector,
        end_position: Vector,
        orientation: Vector = (0, 0, 0),
        time: float = 1,
        easing_style: tp.Literal[
            'Linear', 'Sine',
            'Quad', 'Cubic', 'Quart', 'Quint', 'Exponential',
            'Circular', 'Back', 'Bounce', 'Elastic'
        ] = 'Quad',
        easing_direction: tp.Literal['In', 'Out', 'InOut'] = 'Out',
        carry: bool = True,
        reverses: bool = True,
        loop: int = -1
    ):
        super(MovingBlock, self).__init__(name, position, size, orientation=orientation)
        if motion_type == 'Persistent':
            self.add_behavior(
                behaviors.PersistentTranslateBehavior(
                    start_position=start_position, 
                    end_position=end_position, 
                    time=time, 
                    easing_style=easing_style, 
                    easing_direction=easing_direction,
                    reverses=reverses,
                )
            )
        elif motion_type == 'Touch':
            self.add_behavior(
                behaviors.TouchTranslateBehavior(
                    start_position=start_position, 
                    end_position=end_position, 
                    time=time, 
                    easing_style=easing_style, 
                    easing_direction=easing_direction,
                    reverses=reverses,
                    loop=loop
                )
            )
        if carry:
            self.add_behavior(behaviors.CarryBehavior())
        else:
            raise TypeError("Invalid motion_type:", motion_type)
        

class UnanchoredBlock(Part):
    """A block that can be pushed around."""
    part_type = "Part"
    
    def __init__(
        self,
        name: str,
        position: Vector,
        size: Vector,
        orientation: Vector = (0, 0, 0)
    ):
        super(UnanchoredBlock, self).__init__(name, position, size, orientation=orientation)
        self.add_behavior(behaviors.UnanchoredBehavior())


class HoldButton(Part):
    """A block that can be pressed."""
    part_type = "Part"

    def __init__(
        self,
        name: str,
        position: Vector,
        size: Vector,
        orientation: Vector = (0, 0, 0),
        state: bool = False
    ):
        super(HoldButton, self).__init__(name, position , size, orientation=orientation)
        self.state = behaviors.State(self.name, state)
        self.add_behavior(behaviors.HoldStateBehavior())


class PressButton(Part):
    """A block that can be pressed."""
    part_type = "Part"

    def __init__(
        self,
        name: str,
        position: Vector,
        size: Vector,
        orientation: Vector = (0, 0, 0),
        state: bool = False
    ):
        super(PressButton, self).__init__(name, position , size, orientation=orientation)
        self.state = behaviors.State(self.name, state)
        self.add_behavior(behaviors.PressStateBehavior())


class RandomPart(Distribution):
    def __init__(
        self, 
        part_type: Distribution, 
        width: Distribution,  # x
        height: Distribution, # y
        length: Distribution, # z
        pitch: Distribution,  # x
        yaw: Distribution,    # y
        roll : Distribution   # z
    ):
        self.part_type = part_type
        self.width = width
        self.height = height
        self.length = length
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def sample(self) -> Part:
        part_type_sample = self.part_type.sample()
        if part_type_sample == Ball:
            radius = self.width.sample()
            part_size = (radius, radius, radius)
        elif part_type_sample == Cylinder:
            radius = self.width.sample()
            height = self.height.sample()
            part_size = (radius, height, radius)
        else:
            part_size = (self.width.sample(), self.height.sample(), self.length.sample())
        
        return part_type_sample(
            f"{part_type_sample.part_type}",
            (self.pitch.sample(), self.yaw.sample(), self.roll.sample()),
            part_size
        )