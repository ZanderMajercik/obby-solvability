from enum import IntEnum

class BehaviorType(IntEnum):
    NONE = 0
    SPAWN = 1 << 0
    CHECKPOINT = 1 << 1
    DAMAGE = 1 << 2
    ROTATE = 1 << 3
    MOTION = 1 << 4
    TRAMPOLINE = 1 << 5
    CONVEYOR = 1 << 6
    TRUSS = 1 << 7
    PRESS_STATE = 1 << 8
    HOLD_STATE = 1 << 9
    WAYPOINT = 1 << 10
    NOCOLLIDE = 1 << 11
    UNANCHORED = 1 << 12

class SimObject(IntEnum):
    Cube = 1  # colored cubes, ids 2-9
    Sphere = 10  # colored spheres, ids 11-13
    Wedge = 14  # colored spheres, ids 15-18
    Cylinder = 18


# From https://create.roblox.com/docs/reference/engine/enums/EasingStyle
class EasingStyle(IntEnum):
    Linear = 0,
    Sine = 1,  # NOT IMPLEMENTED
    Back = 2,  # NOT IMPLEMENTED
    Quad = 3,  # NOT IMPLEMENTED
    Quart = 4,  # NOT IMPLEMENTED
    Quint = 5,  # NOT IMPLEMENTED
    Bounce = 6,  # NOT IMPLEMENTED
    Elastic = 7,  # NOT IMPLEMENTED
    Exponential = 8,  # NOT IMPLEMENTED
    Circular = 9,  # NOT IMPLEMENTED
    Cubic = 10  # NOT IMPLEMENTED


# From https://create.roblox.com/docs/reference/engine/enums/EasingDirection
class EasingDirection(IntEnum):
    In = 0,  # NOT IMPLEMENTED
    Out = 1,
    InOut = 2  # NOT IMPLEMENTED