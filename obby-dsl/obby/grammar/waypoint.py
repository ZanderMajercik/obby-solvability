import json

from enum import Enum

from obby.utils.vector import *
from obby.grammar.distribution import Gaussian
from obby.grammar.type_aliases import Vector as VectorType

# TODO: Add Wait action. May need to be a class.
Action = Enum('Action', ['WALK', 'JUMP', 'CLIMB'])


class Waypoint:
    def __init__(self, position: VectorType, size: VectorType):
        self.position = position
        self.size = size

    def jsonify(self, unit: float = 1.0):
        return {
            "position": vscale(unit, self.position),
            "size": vscale(unit, self.size),
        }


class AvoidRegion:
    def __init__(self, position: VectorType, size: VectorType):
        self.position = position
        self.size = size

    def jsonify(self, unit: float = 1.0):
        return {
            "position": vscale(unit, self.position),
            "size": vscale(unit, self.size),
        }


class Path:
    def __init__(self, start: VectorType = None, end: VectorType = None, waypoints: list[Waypoint] = None):
        self.start = start if start is not None else (0, 0, 0)
        self.end = end if end is not None else (0, 0, 0)
        self.waypoints = waypoints if waypoints is not None else []

    def append_waypoints(self, waypoints: Waypoint | list[Waypoint]):
        if isinstance(waypoints, Waypoint):
            self.waypoints.append(waypoints)
        else:
            for waypoint in waypoints:
                self.waypoints.append(waypoint)

    def perturb(self, radius: float, idxs: list[int] = None):
        """Apply perturbation to the locations of waypoints."""
        perturbed_path = Path(start=self.start, end=self.end)
        sampler = Gaussian(0, radius)
        for i, w in enumerate(self.waypoints):
            if idxs is None or i in idxs:
                perturbed_waypoint = Waypoint(
                    position=vadd(w.position, (sampler.sample(), sampler.sample(), sampler.sample())),
                    action=w.action
                )
            else:
                perturbed_waypoint = Waypoint(w.position, w.action)
            perturbed_path.append_waypoints(perturbed_waypoint)
        return perturbed_path


    def jsonify(self, unit: float = 1):
        info = {
            'start': vscale(unit, self.start),
            'end': vscale(unit, self.end),
        }
        info['path'] = [waypoint.jsonify(unit=unit)
            for idx, waypoint in enumerate(self.waypoints)
        ]
        j = json.dumps(info, indent=4)
        return j