import typing as tp

# from dataclasses import dataclass, field
from obby.grammar.type_aliases import *
from obby.grammar.distribution import *


class State:
    name: str
    value: bool

    def __init__(self, name: str, value: bool):
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"State {self.name}"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "value": self.value
        }


class NoneState(State):
    def __init__(self):
        super(NoneState, self).__init__("None", False)


class Behavior:
    name: tp.ClassVar[str]

    def __init__(self, conditions: list[State] = None):
        self.conditions = conditions if conditions else []
        # print("init:", self.conditions)

    def randomize(self, sampler):
        # TODO: implement here
        return

    def __str__(self) -> str:
        return f"Behavior {self.name}"

    def add_condition(self, condition: State):
        # print("before add:", self.conditions)
        self.conditions.append(condition)
        # print("after add:", self.conditions)

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "conditions": [condition.name for condition in self.conditions]
        }


####################################################################################################
# Spawn/Save Behaviors
####################################################################################################

class SpawnBehavior(Behavior):
    name = "Spawn"

    def __init__(self):
        super(SpawnBehavior, self).__init__()


class CheckpointBehavior(Behavior):
    name = "Checkpoint"

    def __init__(self, conditions: list[State] = None):
        super(CheckpointBehavior, self).__init__(conditions)


####################################################################################################
# Waypoint/Avoid Behaviors
####################################################################################################

class WaypointBehavior(Behavior):
    name = "Waypoint"

    def __init__(self, index: int, jumps: int):
        super(WaypointBehavior, self).__init__()
        self.index = index
        self.jumps = jumps
    
    def __str__(self) -> str:
        return f"{self.type} Behavior {self.name} (index={self.index})"
    
    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "jumps": self.jumps,
            "conditions": [condition.name for condition in self.conditions]
        }

class AvoidBehavior(Behavior):
    name = "Avoid"

    def __init__(self):
        super(AvoidBehavior, self).__init__()

class NoCollideBehavior(Behavior):
    name = "NoCollide"

    def __init__(self):
        super(NoCollideBehavior, self).__init__()

####################################################################################################
# Health Behaviors
####################################################################################################

class DamageBehavior(Behavior):
    name = "Damage"

    def __init__(self, amount: float, conditions: list[State] = None):
        super(DamageBehavior, self).__init__(conditions)
        self.amount = amount

    def __str__(self) -> str:
        return f"{self.type} Behavior {self.name} (damage={self.amount})"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "damage": self.amount,
            "conditions": [condition.name for condition in self.conditions]
        }


####################################################################################################
# Physics Behaviors
####################################################################################################

class VelocityBehavior(Behavior):
    name = "Velocity"

    def __init__(self, velocity: Vector, conditions: list[State] = None):
        super(VelocityBehavior, self).__init__(conditions)
        self.velocity = velocity

    def __str__(self) -> str:
        return f"{self.type} Behavior {self.name} (velocity={self.velocity})"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "velocity": (
            kwargs["unit"] * self.velocity[0], kwargs["unit"] * self.velocity[1], kwargs["unit"] * self.velocity[2]),
            "conditions": [condition.name for condition in self.conditions]
        }


class TrampolineBehavior(VelocityBehavior):
    name = "Trampoline"

    def __init__(self, bounce: float, conditions: list[State] = None):
        super(TrampolineBehavior, self).__init__(velocity=(0, bounce, 0), conditions=conditions)


class ConveyorBehavior(VelocityBehavior):
    name = "Conveyor"

    def __init__(self, x_velocity: float, z_velocity: float, conditions: list[State] = None):
        super(ConveyorBehavior, self).__init__(velocity=(x_velocity, 0, z_velocity), conditions=conditions)


class FrictionBehavior(Behavior):
    name = "Friction"

    def __init__(self, friction: float, weight: float, conditions: list[State] = None):
        super(FrictionBehavior, self).__init__(conditions)
        self.friction = friction
        self.weight = weight

    def __str__(self) -> str:
        return f"{self.type} Behavior {self.name} (weight={self.weight})"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "friction": self.friction,
            "weight": self.weight,
            "conditions": [condition.name for condition in self.conditions]
        }


class TrussBehavior(Behavior):
    name = "Truss"

    def __init__(self, conditions: list[State] = None):
        super(TrussBehavior, self).__init__(conditions)


class UnanchoredBehavior(Behavior):
    name = "Unanchored"

    def __init__(self, conditions: list[State] = None):
        super(UnanchoredBehavior, self).__init__(conditions)


####################################################################################################
# Motion Behaviors
####################################################################################################

class CarryBehavior(Behavior):
    name = "Carry"

    def __init__(self, conditions: list[State] = None):
        super(CarryBehavior, self).__init__(conditions)


class PersistentTranslateBehavior(Behavior):
    name = "PersistentTranslate"

    def __init__(
            self,
            start_position: Vector,
            end_position: Vector,
            time: float = 1,
            easing_style: tp.Literal[
                'Linear', 'Sine',
                'Quad', 'Cubic', 'Quart', 'Quint', 'Exponential',
                'Circular', 'Back', 'Bounce', 'Elastic'
            ] = 'Quad',
            easing_direction: tp.Literal['In', 'Out', 'InOut'] = 'Out',
            reverses: bool = True,
            loop: int = -1,
            conditions: list[State] = None
    ):
        super(PersistentTranslateBehavior, self).__init__(conditions)
        self.start_position = start_position
        self.end_position = end_position
        self.time = time
        self.easing_style = easing_style
        self.easing_direction = easing_direction
        self.loop = loop
        self.reverses = reverses

    def __str__(self) -> str:
        return f"Behavior {self.name} (" + \
            f"start_position={self.start_position}, " + \
            f"end_position={self.end_position}, " + \
            f"time={self.time}, " + \
            f"easing_style={self.easing_style}, " + \
            f"easing_direction={self.easing_direction}, " + \
            f"reverses={self.reverses}), " + \
            f"loop={self.loop}"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "conditions": [condition.name for condition in self.conditions],
            "start_position": (
                kwargs["unit"] * self.start_position[0],
                kwargs["unit"] * self.start_position[1],
                kwargs["unit"] * self.start_position[2]
            ),
            "end_position": (
                kwargs["unit"] * self.end_position[0],
                kwargs["unit"] * self.end_position[1],
                kwargs["unit"] * self.end_position[2]
            ),
            "time": self.time,
            "easing_style": self.easing_style,
            "easing_direction": self.easing_direction,
            "reverses": self.reverses,
            "loop": self.loop
        }


class TouchTranslateBehavior(PersistentTranslateBehavior):
    name = "TouchTranslate"

    def __init__(
            self,
            start_position: Vector,
            end_position: Vector,
            time: float = 1,
            easing_style: tp.Literal[
                'Linear', 'Sine',
                'Quad', 'Cubic', 'Quart', 'Quint', 'Exponential',
                'Circular', 'Back', 'Bounce', 'Elastic'
            ] = 'Quad',
            easing_direction: tp.Literal['In', 'Out', 'InOut'] = 'Out',
            reverses: bool = False,
            loop: int = -1,
            conditions: list[State] = None
    ):
        super(TouchTranslateBehavior, self).__init__(
            start_position, end_position, time,
            easing_style, easing_direction, reverses, loop, conditions
        )
    #     self.loop = loop

    # def __str__(self) -> str:
    #     return f"Behavior {self.name} (" + \
    #         f"start_position={self.start_position}, " + \
    #         f"end_position={self.end_position}, " + \
    #         f"time={self.time}, " + \
    #         f"easing_style={self.easing_style}, " + \
    #         f"easing_direction={self.easing_direction}, " + \
    #         f"reverses={self.reverses}), " + \
    #         f"loop={self.loop}"

    # def jsonify(self, **kwargs) -> dict:
    #     return {
    #         "name": self.name,
    #         "conditions": [condition.name for condition in self.conditions],
    #         "start_position": (
    #             kwargs["unit"] * self.start_position[0],
    #             kwargs["unit"] * self.start_position[1],
    #             kwargs["unit"] * self.start_position[2]
    #         ),
    #         "end_position": (
    #             kwargs["unit"] * self.end_position[0],
    #             kwargs["unit"] * self.end_position[1],
    #             kwargs["unit"] * self.end_position[2]
    #         ),
    #         "time": self.time,
    #         "easing_style": self.easing_style,
    #         "easing_direction": self.easing_direction,
    #         "reverses": self.reverses,
    #         "loop": self.loop
    #     }


class PersistentScaleBehavior(Behavior):
    name = "PersistentScale"

    def __init__(
            self,
            start_scale: Vector,
            end_scale: Vector,
            time: float = 1,
            easing_style: tp.Literal[
                'Linear', 'Sine',
                'Quad', 'Cubic', 'Quart', 'Quint', 'Exponential',
                'Circular', 'Back', 'Bounce', 'Elastic'
            ] = 'Quad',
            easing_direction: tp.Literal['In', 'Out', 'InOut'] = 'Out',
            reverses: bool = True,
            loop: int = -1,
            conditions: list[State] = None
    ):
        super(PersistentScaleBehavior, self).__init__(conditions=conditions)
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.time = time
        self.easing_style = easing_style
        self.easing_direction = easing_direction
        self.reverses = reverses
        self.loop = loop

    def __str__(self) -> str:
        return f"Behavior {self.name} (" + \
            f"start_scale={self.start_scale}, " + \
            f"end_scale={self.end_scale}, " + \
            f"time={self.time}, " + \
            f"easing_style={self.easing_style}, " + \
            f"easing_direction={self.easing_direction}, " + \
            f"reverses={self.reverses}), " + \
            f"loop={self.loop}"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "conditions": [condition.name for condition in self.conditions],
            "start_scale": self.start_scale,
            "end_scale": self.end_scale,
            "time": self.time,
            "easing_style": self.easing_style,
            "easing_direction": self.easing_direction,
            "reverses": self.reverses,
            "loop": self.loop
        }


class TouchScaleBehavior(PersistentScaleBehavior):
    name = "TouchScale"
    conditions: list[State] = None

    def __init__(
            self,
            start_scale: Vector,
            end_scale: Vector,
            time: float = 1,
            easing_style: tp.Literal[
                'Linear', 'Sine',
                'Quad', 'Cubic', 'Quart', 'Quint', 'Exponential',
                'Circular', 'Back', 'Bounce', 'Elastic'
            ] = 'Quad',
            easing_direction: tp.Literal['In', 'Out', 'InOut'] = 'Out',
            reverses: bool = True,
            loop: int = -1,
            conditions: list[State] = None
    ):
        super(TouchScaleBehavior, self).__init__(
            start_scale, end_scale, time,
            easing_style, easing_direction, reverses, loop, conditions
        )

    # def __str__(self) -> str:
    #     return f"Behavior {self.name} (" + \
    #         f"start_scale={self.start_scale}, " + \
    #         f"end_scale={self.end_scale}, " + \
    #         f"time={self.time}, " + \
    #         f"easing_style={self.easing_style}, " + \
    #         f"easing_direction={self.easing_direction}, " + \
    #         f"reverses={self.reverses}), " + \
    #         f"loop={self.loop}"

    # def jsonify(self, **kwargs) -> dict:
    #     return {
    #         "name": self.name,
    #         "conditions": [condition.name for condition in self.conditions],
    #         "start_scale": self.start_scale,
    #         "end_scale": self.end_scale,
    #         "time": self.time,
    #         "easing_style": self.easing_style,
    #         "easing_direction": self.easing_direction,
    #         "reverses": self.reverses,
    #         "loop": self.loop
    #     }


class PersistentRotateBehavior(Behavior):
    name = "PersistentRotate"

    def __init__(self, about: Vector, angular: float, axis: tp.Literal['up', 'right', 'forward'] = 'up', loop: int = -1, conditions: list[State] = None):
        super(PersistentRotateBehavior, self).__init__(conditions)
        self.about = about
        self.angular = angular
        self.axis = axis
        self.loop = loop

    def __str__(self) -> str:
        return f"Behavior {self.name} (" + \
            f"about={self.about}, " + \
            f"angular={self.angular}, " + \
            f"axis={self.axis}, " + \
            f"loop={self.loop}"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "conditions": [condition.name for condition in self.conditions],
            "about": (
                kwargs["unit"] * self.about[0],
                kwargs["unit"] * self.about[1],
            ),
            "angular": self.angular,
            "axis": self.axis,
            "loop": self.loop
        }


class TouchRotateBehavior(PersistentRotateBehavior):
    name = "TouchRotate"

    def __init__(self, about: Vector, angular: float, loop: int = -1, conditions: list[State] = None):
        super(TouchRotateBehavior, self).__init__(about, angular, loop, conditions)

    # def __str__(self) -> str:
    #     return f"Behavior {self.name} (" + \
    #         f"about={self.about}, " + \
    #         f"angular={self.angular}, " + \
    #         f"loop={self.loop}"

    # def jsonify(self, **kwargs) -> dict:
    #     return {
    #         "name": self.name,
    #         "conditions": [condition.name for condition in self.conditions],
    #         "about": (
    #             kwargs["unit"] * self.about[0],
    #             kwargs["unit"] * self.about[1],
    #         ),
    #         "angular": self.angular,
    #         "loop": self.loop
    #     }


####################################################################################################
# Miscellaneous Behaviors
####################################################################################################

class DisappearBehavior(Behavior):
    name = "Disappear"

    def __init__(self, disappear_delay: float, reappear_delay: float, conditions: list[State] = None):
        super(DisappearBehavior, self).__init__(conditions)
        self.disappear_delay = disappear_delay
        self.reappear_delay = reappear_delay

    def __str__(self) -> str:
        return f"Behavior {self.name} (disappear={self.disappear_delay}, reappear={self.reappear_delay})"

    def jsonify(self, **kwargs) -> dict:
        return {
            "name": self.name,
            "conditions": [condition.name for condition in self.conditions],
            "disappear_delay": self.disappear_delay,
            "reappear_delay": self.reappear_delay,
        }


class HoldStateBehavior(Behavior):
    name = "HoldState"


class PressStateBehavior(Behavior):
    name = "PressState"
