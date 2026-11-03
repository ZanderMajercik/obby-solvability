import math
import torch

import numpy as np

def orient(v, orientation):
    if not isinstance(v, np.ndarray):
        v_ = np.array(v)
    else:
        v_ = v.copy()
    rotations = []
    for i in range(3):
        theta = math.radians(orientation[i])
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        if i == 0:
            rotation = torch.tensor([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ], dtype=v_.dtype) if torch.is_tensor(v_) else np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])
        elif i == 1:
            rotation = torch.tensor([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta],
            ], dtype=torch.tensor) if torch.is_tensor(v_) else np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta],
            ])
        else:
            rotation = torch.tensor([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=v_.dtype) if torch.is_tensor(v_) else np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
        rotations.append(rotation)

    # if len(v.shape) < 2:
    #     return rotations[2] @ rotations[0] @ rotations[1] @ v
    # else:
    # print(v)
    return (rotations[2] @ rotations[0] @ rotations[1] @ v_.T).T

def inv_orient(v, orientation):
    rotations = []
    for i in range(3):
        theta = math.radians(-orientation[i])
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        if i == 0:
            rotation = torch.tensor([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ], dtype=v.dtype) if torch.is_tensor(v) else np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])
        elif i == 1:
            rotation = torch.tensor([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta],
            ], dtype=v.dtype) if torch.is_tensor(v) else np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta],
            ])
        else:
            rotation = torch.tensor([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=v.dtype) if torch.is_tensor(v) else np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
        rotations.append(rotation)

    # if len(v.shape) < 2:
    #     return rotations[1] @ rotations[0] @ rotations[2] @ v
    # else:
    return (rotations[1] @ rotations[0] @ rotations[2] @ v.T).T

def make_cube(position, size, color, orientation=(0, 0, 0), v_offset=0):
    v = np.array([
        [-0.5 * size[0], -0.5 * size[1], -0.5 * size[2]],
        [-0.5 * size[0], -0.5 * size[1],  0.5 * size[2]],
        [-0.5 * size[0],  0.5 * size[1], -0.5 * size[2]],
        [-0.5 * size[0],  0.5 * size[1],  0.5 * size[2]],
        [ 0.5 * size[0], -0.5 * size[1], -0.5 * size[2]],
        [ 0.5 * size[0], -0.5 * size[1],  0.5 * size[2]],
        [ 0.5 * size[0],  0.5 * size[1], -0.5 * size[2]],
        [ 0.5 * size[0],  0.5 * size[1],  0.5 * size[2]]
    ], dtype=np.float32)
    v = orient(v, orientation)
    v = v + np.array(position)

    f = np.array([
        [1, 3, 0],
        [4, 1, 0],
        [0, 3, 2],
        [2, 4, 0],
        [1, 7, 3],
        [5, 1, 4],
        [5, 7, 1],
        [3, 7, 2],
        [6, 4, 2],
        [2, 7, 6],
        [6, 5, 4],
        [7, 5, 6]
    ], dtype=np.int32) + v_offset

    c = np.tile(color, (f.shape[0], 1))

    return v, f, c