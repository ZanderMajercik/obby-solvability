"""Describes different levels of obbies.

NOTE: I'm using (X, Y, Z) order in my tuples for the coordinate space. However, Roblox uses the Y-axis to describe 
vertical movement, so you will notice this switch in the loadObby script.
"""

from __future__ import annotations

import json
# import math
import collections

import typing as tp
from dataclasses import dataclass

from obby.grammar.part import *
from obby.grammar.obstacles import *


@dataclass
class Obby(object):
    name: str
    unit: float = 1
    p_idx: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    parts: list[Part] = field(default_factory=lambda: [])
    waypoints: list[Waypoint] = field(default_factory=lambda: [])
    avoid: list[AvoidRegion] = field(default_factory=lambda: [])

    def __str__(self) -> str:
        rep_str = f"Obby: {self.name}\n"
        rep_str += "START\n"
        for i, part in enumerate(self.parts):
            rep_str += f"{i}: "
            rep_str += str(part)
        rep_str += "END"
        return rep_str

    def append_part(self, part: Part):
        self.parts.append(part)

    def append_parts(self, parts: list[Part]):
        for part in parts:
            self.append_part(part)

    def append_waypoint(self, waypoint: Waypoint):
        self.waypoints.append(waypoint)

    def append_avoid(self, avoid: AvoidRegion):
        self.avoid.append(avoid)
    
    def append_waypoints(self, waypoints: list[Waypoint]):
        for waypoint in waypoints:
            self.append_waypoint(waypoint)
    
    def append_avoids(self, avoids: list[AvoidRegion]):
        for avoid in avoids:
            self.append_avoid(avoid)

    def geometry(self):#, unit: float = 1):
        all_v = []
        all_f = []
        v_offset = 0
        for part in self.parts:
            part_geometry = part.geometry(unit=self.unit, v_offset=v_offset)
            v = part_geometry[0]
            f = part_geometry[1]
            v_offset += v.shape[0]
            all_v.append(v)
            all_f.append(f)
        return np.concatenate(all_v), np.concatenate(all_f)
    
    def jsonify(self):#, unit: float = 1):
        info = {}
        for idx, part in enumerate(self.parts):
            part_name, part_info = part.jsonify(unit=self.unit)
            info[part_name] = part_info
            info[part_name]["idx"] = idx
        j = json.dumps(info, indent=4)
        return j

    def spawn(self, position: tuple[float], size: float, orientation: tuple[float] = (0,0,0)) -> SpawnLocation:
        block_position = position
        block_size = (size, PLATFORM_Y, size)
        part = SpawnLocation(
            f"SpawnLocation {self.p_idx['spawn']}",
            position=block_position,
            size=block_size,
            orientation=orientation
        )
        self.append_part(part)

        # Update block idx.
        self.p_idx['spawn'] += 1

        return part
    
    def randomize(self, position_sampler: Distribution, size_sampler: Distribution, orientation_sampler: Distribution):
        for part in self.parts:
            part.randomize(position_sampler, size_sampler, orientation_sampler)

    # def checkpoint(
    #         self, 
    #         position: tuple[float], 
    #         size: float, 
    #     ) -> Part:
    #     block_position = position
    #     block_size = (size, size, PLATFORM_Y)    
    #     part = Checkpoint(
    #         f"Checkpoint {self.p_idx['checkpoint']}",
    #         position=block_position,
    #         size=block_size,
    #     )
    #     self.append_part(part)
    #     self.p_idx['checkpoint'] += 1

    #     return part

    # def part(
    #         self, 
    #         part_type: tp.Literal['Block', 'Lava', 'Conveyor', 'Trampoline', 'DisappearBlock', 'MovingBlock'],
    #         position: tuple[float],
    #         size: tp.Union[float, tuple[float]],
    #         **kwargs
    #     ) -> Part:
    #     # Compute dimensions and placement of block.
    #     if isinstance(size, tuple):
    #         if len(size) == 2:
    #             block_size = (size[0], size[1], PLATFORM_Y)
    #         else:
    #             block_size = size
    #     else:
    #         block_size = (size, size, PLATFORM_Y)
    #     block_position = position
    
    #     # Place block.
    #     # TODO: Make this simpler.
    #     if part_type == 'Block':
    #         part = Block(
    #             f"BlockPart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #         )
    #     elif part_type == 'Lava':
    #         part = Lava(
    #             f"LavaPart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #         )
    #     elif part_type == 'Conveyor':
    #         part = Conveyor(
    #             f"ConveyorPart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #             velocity=kwargs["velocity"],
    #         )
    #     elif part_type == 'Trampoline':
    #         part = Trampoline(
    #             f"TrampolinePart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #             bounce=kwargs["bounce"],
    #         )
    #     elif part_type == 'BounceBlock':
    #         part = BounceBlock(
    #             f"BounceBlockPart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #             bounce=kwargs["bounce"],
    #         )
    #     elif part_type == 'DisappearBlock':
    #         part = DisappearBlock(
    #             f"DisappearBlockPart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #             disappear_delay=kwargs["disappear_delay"],
    #             reappear_delay=kwargs["reappear_delay"],
    #         )
    #     elif part_type == 'MovingBlock':
    #         part = MovingBlock(
    #             f"MovingBlockPart {self.p_idx[part_type]}",
    #             position=block_position,
    #             size=block_size,
    #             motion_type=kwargs["motion_type"],
    #             start_position=kwargs["start_position"],
    #             end_position=kwargs["end_position"],
    #             time=kwargs["time"],
    #             easing_style=kwargs["easing_style"],
    #             easing_direction=kwargs["easing_direction"]
    #         )
    #     else:
    #         raise TypeError("Invalid part type:", part_type)
    #     self.append_part(part)
    #     self.p_idx[part_type] += 1
        
    #     return part