from __future__ import annotations

import json
from dataclasses import dataclass

from obby.grammar.constants import *
from obby.utils.vector import *


@dataclass
class PhysicsDistribution:
    def __init__(self, gravity: Distribution, move_force: Distribution, jump_force: Distribution, air_resistance: Distribution, spawn_position: Distribution, spawn_velocity: Distribution):
        self.gravity = gravity
        self.move_force = move_force
        self.jump_force = jump_force
        self.air_resistance = air_resistance
        self.spawn_position = spawn_position
        self.spawn_velocity = spawn_velocity

    def generate(self):
        # spawn_position and spawn_velocity are not sampled (no Vector3 distributions).
        return Physics(gravity=self.gravity.sample(), 
                       move_force=self.move_force.sample(),
                       jump_force=self.jump_force.sample(),
                       air_resistance=self.air_resistance.sample())


@dataclass
class Physics:
    gravity: float = 196.2
    move_force: float = 216.034
    jump_force: float = 55.0 # Raise to 65 if the jump height does not cover the sim mismatch.
    air_resistance: float = 4.0
    spawn_position: tuple[float, float, float] = (-1,-1,-1)
    spawn_velocity: tuple[float, float, float] = (0, 0, 0)
    name: str = "Physics"

    def jsonify(self) -> tuple[str, dict]:
        info = {
            "gravity": self.gravity,
            "move_force": self.move_force,
            "jump_force": self.jump_force,
            "air_resistance": self.air_resistance,
            "spawn_position": self.spawn_position,
            "spawn_velocity": self.spawn_velocity
        }
        return json.dumps(info, indent=4)
