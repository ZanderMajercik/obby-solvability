import math

def vsub(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def vadd(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

def vdot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def magnitude(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def vscale(a, v):
    return (a * v[0], a * v[1], a * v[2])

def vmul(v1, v2):
    return (v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2])

def vangle(v1, v2):
    math.acos(vdot(v1, v2) / (magnitude(v1) * magnitude(v2)))