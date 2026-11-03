"""
Draws trajectory heatmaps from the position files in scripts/heatmaps/positions,
optionally over a top-down view of the level, and saves them to
scripts/heatmaps/visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import ListedColormap

from parameters.enums import BehaviorType

parser = argparse.ArgumentParser()
parser.add_argument('--level-type', type=str,
                    help="Obstacle to collate pictures of.")
parser.add_argument('--level-npy', type=str, help="Binary of level (npy file).")
parser.add_argument('--draw-jumps', action='store_true', help="Draw jumps.")


def draw_part(pos, size, orientation, behavior_type):
    cx, cy, cz = pos
    sx, sy, sz = size
    line_width = 2
    alpha = 0.3
    zorder = 1
    if behavior_type == BehaviorType.SPAWN:
        color = "green"
    elif behavior_type == BehaviorType.CHECKPOINT:
        color = "blue"
    elif behavior_type == BehaviorType.DAMAGE:
        color = "red"
        zorder = 10
    else:
        color = "dimgray"
        zorder = -1

    corner_offsets = np.array([[-sx / 2, -sy / 2],
                               [sx / 2, -sy / 2],
                               [sx / 2, sy / 2],
                               [-sx / 2, sy / 2]])
    # Rotate the corners about z.
    z_rotation = orientation[2] * np.pi / 180
    rotation_matrix = np.array([[np.cos(z_rotation), -np.sin(z_rotation)],
                                [np.sin(z_rotation), np.cos(z_rotation)]])
    corner_offsets = np.dot(corner_offsets, rotation_matrix.T)

    corners = corner_offsets + np.array([cx, cy])

    x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
    y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
    # Outline
    plt.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]],
             color=color, linewidth=line_width, zorder=zorder)
    plt.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]],
             color=color, linewidth=line_width, zorder=zorder)
    plt.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]],
             color=color, linewidth=line_width, zorder=zorder)
    plt.plot([corners[3, 0], corners[0, 0]], [corners[3, 1], corners[0, 1]],
             color=color, linewidth=line_width, zorder=zorder)
    # Fill, no border
    plt.fill(corners[:, 0], corners[:, 1], color=color, alpha=alpha,
             edgecolor=None, zorder=zorder)

    return x_min, x_max, y_min, y_max


def create_heatmap(data, bounds=None):
    x_data, y_data = data[:, :, 1], data[:, :, 2]
    x_data = x_data.flatten()
    y_data = y_data.flatten()

    # Colormap in which 0 is white.
    base_cmap = plt.get_cmap("plasma")
    colors = [base_cmap(i) for i in range(base_cmap.N)]
    custom_cmap = ListedColormap(['white'] + colors[1:])
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        plt.hist2d(x_data, y_data, bins=100, cmap=custom_cmap,
                   range=[[x_min, x_max], [y_min, y_max]])
    else:
        plt.hist2d(x_data, y_data, bins=100, cmap=custom_cmap)

    return

def draw_traces(data, color):
    x_data, y_data = data[:, :, 1], data[:, :, 2]
    x_data, y_data = x_data.flatten(), y_data.flatten()

    # Plot traces. Whenever there is a discontinuity, plot a new line
    ind_dis = np.where(np.abs(np.diff(y_data)) > 3)
    ind_dis = np.concatenate((np.array([0]), ind_dis[0] + 1))
    for i in range(len(ind_dis) - 1):
        plt.plot(x_data[ind_dis[i]:ind_dis[i + 1]],
                 y_data[ind_dis[i]:ind_dis[i + 1]],
                 color=color,
                 alpha=0.05,
                 linewidth=1)
    return

def draw_dones(data):
    x_data, y_data, dones = data[:, :, 1], data[:, :, 2], data[:, :, 5]
    worlds_done, t_done = np.where(dones == 1)
    x_done, y_done = x_data[worlds_done, t_done - 1], y_data[worlds_done, t_done - 1]
    plt.scatter(x_done, y_done, color='red', marker="x", s=10, alpha=0.2)


def draw_jumps(data):
    x_data, y_data, jumped = data[:, :, 1], data[:, :, 2], data[:, :, 4]
    # Only jumps below y = -6.
    ind = np.where(y_data < -6)
    x_data, y_data, jumped = x_data[ind], y_data[ind], jumped[ind]

    x_jump, y_jump = x_data[jumped == 1], y_data[jumped == 1]
    plt.scatter(x_jump, y_jump, color='purple', marker="s", s=10, alpha=0.2)
    return


def draw_level_overlay(level_data):
    x_min, y_min = 99999, 99999
    x_max, y_max = -99999, -99999
    # Level 0. Rows 0 and 1 of the level tensor hold the physics parameters and
    # the trajectory info, so the parts start at row 2.
    level_parts = level_data[0, 2:, :]
    for part in level_parts:
        pos = part[0:3]
        size = part[3:6]
        orientation = part[6:9]
        behavior = part[12]
        if np.sum(size) == 0 or behavior < -1:
            continue
        behavior = 0 if behavior == -1 else behavior
        px_min, px_max, py_min, py_max = draw_part(pos, size, orientation,
                                                   BehaviorType(behavior))
        x_min = min(x_min, px_min)
        x_max = max(x_max, px_max)
        y_min = min(y_min, py_min)
        y_max = max(y_max, py_max)
    # Pad the bounds by 1.
    x_min, y_min = x_min - 1, y_min - 1
    x_max, y_max = x_max + 1, y_max + 1
    bounds = (x_min, x_max, y_min, y_max)
    return bounds


def main():
    args = parser.parse_args()
    files = os.listdir("scripts/heatmaps/positions")
    for file in sorted(files):
        if f"{args.level_type}" not in file:
            continue
        if "_" in file:
            level_type = file.split("_")[0]
        else:
            level_type = file.split(".")[0]

        data = np.load(os.path.join("scripts/heatmaps/positions", file))
        success_rate = np.mean(data[:, :, 0])

        plt.figure()

        bounds = None

        # The level parts (drawn with their z rotation only).
        if args.level_npy is not None:
            level = np.load(args.level_npy)
            bounds = draw_level_overlay(level)

        draw_traces(data, (1.0, 0.5, 0))
        draw_jumps(data)
        draw_dones(data)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.yticks([bounds[2], bounds[3]])
        plt.xticks([bounds[0], bounds[1]])

        base_file, _ = os.path.splitext(file)
        plt.savefig(f"scripts/heatmaps/visualization/{base_file}.png", dpi=500)
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
