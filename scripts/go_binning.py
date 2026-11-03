import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize_positions import draw_level_overlay
from generateLevels import json_parts_to_xyz_bounds, transform_coordinates


class BinningFunction:
    """ Binning function that maps points to bins. """
    rep_points = None

    def apply(self, pts):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def index_to_coord(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def coord_to_index(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_adjacent_bins(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def num_bins(self) -> int:
        raise NotImplementedError("Subclasses should implement this method.")

    def get_grid(self):
        raise NotImplementedError("Subclasses should implement this method.")


def get_part_bounds(pos, size, orientation):
    """ Returns the 2D bounding of a part in the level """
    cx, cy, cz = pos
    sx, sy, sz = size
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

    # Axis-aligned bounds of the rotated part.
    x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
    y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
    return x_min, x_max, y_min, y_max


def create_grid(level_data, grid_size, z_grid_size, t_grid_size, t_max, cell_size=None):
    """ Given a level, creates a grid that covers the entire level. """
    # Level 0 only. Parts start at index 2: the first two entries are
    # placeholder objects that carry physics parameters and trajectory data.

    level_parts = level_data[0, 2:, :]

    # Bounding box of the whole level.
    x_min, y_min = 99999, 99999
    x_max, y_max = -99999, -99999
    for part in level_parts:
        pos = part[0:3]
        size = part[3:6]
        orientation = part[6:9]
        behavior = part[12]
        if np.sum(size) == 0 or behavior < -1:
            continue
        px_min, px_max, py_min, py_max = get_part_bounds(pos, size, orientation)
        x_min = min(x_min, px_min)
        x_max = max(x_max, px_max)
        y_min = min(y_min, py_min)
        y_max = max(y_max, py_max)

    if cell_size:
        num_cells_x = int(np.ceil((x_max - x_min) / cell_size))
        num_cells_y = int(np.ceil((y_max - y_min) / cell_size))
        x_grid = torch.linspace(x_min, x_min + cell_size * num_cells_x, num_cells_x + 1)
        y_grid = torch.linspace(y_min, y_min + cell_size * num_cells_y, num_cells_y + 1)
    else:
        x_grid = torch.linspace(x_min, x_max, grid_size)
        y_grid = torch.linspace(y_min, y_max, grid_size)
    
    z_grid = torch.linspace(0, z_max, z_grid_size)
    t_grid = torch.linspace(0, t_max, t_grid_size)

    return x_grid, y_grid, z_grid, t_grid


def create_binning_function(x_grid, y_grid, z_grid, t_grid, j_grid, dyn_grid, dev):
    """ Given a grid, creates a binning function that maps points to bins. """
    x_len = len(x_grid) - 1
    y_len = len(y_grid) - 1
    z_len = len(z_grid) - 1
    t_len = len(t_grid) - 1
    j_len = len(j_grid) - 1
    d_len = len(dyn_grid) - 1

    n_valid_bins = x_len * y_len * z_len * t_len * j_len * d_len

    def coord_to_index(x_bin, y_bin, z_bin, t_bin, j_bin, dyn_bin):
        return dyn_bin * j_len * t_len * z_len * y_len * x_len + \
               j_bin * t_len * z_len * y_len * x_len + \
               t_bin * z_len * y_len * x_len + \
               z_bin * y_len * x_len + \
               y_bin * x_len + \
               x_bin

    def index_to_coord(indices):
        return (
            indices.remainder(x_len), 
            indices.remainder(y_len * x_len) // x_len, 
            indices.remainder(z_len * y_len * x_len) // (y_len * x_len),
            indices.remainder(t_len * z_len * y_len * x_len) // (z_len * y_len * x_len),
            indices.remainder(j_len * t_len * z_len * y_len * x_len) // (t_len * z_len * y_len * x_len),
            indices // (j_len * t_len * z_len * y_len * x_len)
        )
    
    def get_adjacent_bins(indices):
        x, y, z, t, j, dyn = index_to_coord(indices)
        assert not torch.any((x >= len(x_grid) - 1) | (x < 0))
        assert not torch.any((y >= len(y_grid) - 1) | (y < 0))
        assert not torch.any((z >= len(z_grid) - 1) | (z < 0))
        assert not torch.any((t >= len(t_grid) - 1) | (t < 0))
        assert not torch.any((j >= len(j_grid) - 1) | (j < 0))
        assert not torch.any((dyn >= len(dyn_grid) - 1) | (dyn < 0))

        def _clamp(dimension, grid):
            grid_size = len(grid) - 1
            return torch.maximum(torch.minimum(dimension, torch.ones_like(dimension) * (grid_size - 1)), torch.zeros_like(dimension)).int()
        adjacent_bins = torch.cat(
            (coord_to_index(_clamp(x+1, x_grid), y, z, t, j, dyn).unsqueeze(1),
             coord_to_index(_clamp(x-1, x_grid), y, z, t, j, dyn).unsqueeze(1),
             coord_to_index(x, _clamp(y+1, y_grid), z, t, j, dyn).unsqueeze(1),
             coord_to_index(x, _clamp(y-1, y_grid), z, t, j, dyn).unsqueeze(1),
             coord_to_index(x, y, _clamp(z+1, z_grid), t, j, dyn).unsqueeze(1),
             coord_to_index(x, y, _clamp(z-1, z_grid), t, j, dyn).unsqueeze(1),
             coord_to_index(x, y, z, _clamp(t+1, t_grid), j, dyn).unsqueeze(1),
             coord_to_index(x, y, z, _clamp(t-1, t_grid), j, dyn).unsqueeze(1)),
            dim=1
        )
        return adjacent_bins

    def digitize(pts, steps, jumps_remaining, dyn_block_pos):
        if len(pts.shape) == 1:
            pts = pts.unsqueeze(0)
            steps = steps.unsqueeze(0)
        out = torch.zeros(pts.shape[0], dtype=torch.int64, device=dev)
        # Bin only the forward (y) coord of dynamic objects.
        x, y, z, t, j, dyn = pts[:, 0], pts[:, 1], pts[:, 2], steps[:, 0], jumps_remaining[:, 0], dyn_block_pos[:, 1]

        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        t = t.contiguous()
        j = j.contiguous()
        dyn = dyn.contiguous()

        x_bin = torch.bucketize(x, x_grid) - 1
        y_bin = torch.bucketize(y, y_grid) - 1
        z_bin = torch.bucketize(z, z_grid) - 1
        t_bin = torch.bucketize(t, t_grid) - 1
        j_bin = torch.bucketize(j, j_grid) - 1
        dyn_bin = torch.bucketize(dyn, dyn_grid) - 1

        # bucketize returns len(grid) for values >= grid[-1], so clamp the
        # upper end too.
        x_bin = x_bin.clamp(0, x_len - 1)
        y_bin = y_bin.clamp(0, y_len - 1)
        z_bin = z_bin.clamp(0, z_len - 1)
        t_bin = t_bin.clamp(0, t_len - 1)
        j_bin = j_bin.clamp(0, j_len - 1)
        dyn_bin = dyn_bin.clamp(0, d_len - 1)

        out = coord_to_index(
            x_bin, 
            y_bin, 
            z_bin, 
            t_bin, 
            j_bin,
            dyn_bin
        )
        # Points outside the grid get bin -1, so they are not archived.
        valid_ind = (
            (x >= x_grid[0]) & (x <= x_grid[-1]) &
            (y >= y_grid[0]) & (y <= y_grid[-1]) &
            (z >= z_grid[0]) & (z <= z_grid[-1]) &
            (t >= t_grid[0]) & (t <= t_grid[-1]) &
            (j >= j_grid[0]) & (j <= j_grid[-1]) &
            (dyn >= dyn_grid[0]) & (dyn <= dyn_grid[-1])
        )
        out[~valid_ind] = -1
        return out
    
    # Representative state (x, y, z, t) of each bin.
    rep_states = torch.zeros((n_valid_bins, 4), dtype=torch.float, device=dev)
    bin_size = torch.tensor([abs(grid[0] - grid[-1]) / max(1, len(grid) - 1) for grid in (x_grid, y_grid, z_grid, t_grid)], device=dev)
    bin_indices = torch.arange(n_valid_bins, dtype=torch.int64, device=dev)
    x_coord, y_coord, z_coord, t_coord, j_coord, dyn_coord = index_to_coord(bin_indices)
    
    rep_states[..., 0] = x_coord
    rep_states[..., 1] = y_coord
    rep_states[..., 2] = z_coord
    rep_states[..., 3] = t_coord

    rep_states[..., :] = (rep_states[..., :] + 0.5) * bin_size
    rep_states[..., :] += torch.tensor([x_grid[0], y_grid[0], z_grid[0], t_grid[0]], device=dev)

    binning_function = BinningFunction()
    binning_function.apply = digitize
    binning_function.coord_to_index = coord_to_index
    binning_function.index_to_coord = index_to_coord
    binning_function.get_adjacent_bins = get_adjacent_bins
    binning_function.num_bins = lambda: n_valid_bins
    binning_function.get_grid = lambda: (x_grid, y_grid, z_grid, t_grid, j_grid, dyn_grid)
    binning_function.rep_points = rep_states
    print("Number of bins:", n_valid_bins)
    return binning_function

def get_level_binning_function(level_json, 
                               num_x_bins, 
                               num_y_bins, 
                               num_z_bins, 
                               num_t_bins, 
                               t_max, 
                               t_min, 
                               num_j_bins,
                               j_max,
                               j_min,
                               num_dyn_bins,
                               horizontal_mobility, 
                               jump_height,
                               bounds,
                               dev,
                               verbose=False):
    
    x_min, x_max, y_min, y_max, z_min, z_max = \
        json_parts_to_xyz_bounds(level_json)

    scene_x_min, scene_x_max, scene_y_min, scene_y_max, scene_z_min, scene_z_max = x_min, x_max, y_min, y_max, z_min, z_max
    
    if bounds is not None:
        pos = transform_coordinates(bounds["position"])
        extents = [abs(x) / 2 for x in transform_coordinates(bounds["size"])]
        if verbose:
            print(pos)
            print(extents)
        x_min = pos[0] - extents[0]
        x_max = pos[0] + extents[0]
        y_min = pos[1] - extents[1]
        y_max = pos[1] + extents[1]
        z_min = pos[2] - extents[2]
        z_max = pos[2] + extents[2]
        horizontal_mobility = 0
        jump_height = 0

    

    x_grid = torch.linspace(x_min - horizontal_mobility, x_max + horizontal_mobility, num_x_bins + 1).to(dev)
    y_grid = torch.linspace(y_min - horizontal_mobility, y_max + horizontal_mobility, num_y_bins + 1).to(dev)
    z_grid = torch.linspace(z_min, z_max + jump_height, num_z_bins + 1).to(dev)
    t_grid = torch.linspace(t_min, t_max, num_t_bins + 1).to(dev)
    j_grid = torch.linspace(j_min, j_max, num_j_bins + 1).to(dev)
    # The dyn grid always contains 0, so levels without a dynamic object
    # (whose position defaults to (0, 0, 0)) still get a valid bin.
    dyn_y_min = min(scene_y_min, -1.0)
    dyn_y_max = max(scene_y_max, 1.0)
    dyn_grid = torch.linspace(dyn_y_min, dyn_y_max, num_dyn_bins + 1).to(dev)

    binning_function = create_binning_function(x_grid, y_grid, z_grid, t_grid, j_grid, dyn_grid, dev)
    return binning_function


# Interactive 2D view of the binning function (debugging aid).
def main():
    level = np.load("../obby-dsl/latest.npy", allow_pickle=True)
    n_bins = 100
    grid_size = int(np.sqrt(n_bins)) + 1
    x_grid, y_grid = create_grid(level, grid_size)
    binning_function = create_binning_function(x_grid, y_grid,
                                               torch.device("cpu"))
    setup_interactive_plot(level=level, x_grid=x_grid, y_grid=y_grid,
                           binning_function=binning_function,
                           draw_fn=redraw)
    return


def draw_grid(x_grid, y_grid):
    for x in x_grid:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    for y in y_grid:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    return


def setup_interactive_plot(level, x_grid, y_grid, binning_function, draw_fn):
    fig = plt.figure()
    draw_fn(level, x_grid, y_grid, binning_function, ix=0, iy=0)

    def handle_click(event):
        if event.xdata is not None and event.ydata is not None:
            ix, iy = onclick(event)
            plt.clf()
            draw_fn(level, x_grid, y_grid, binning_function, ix, iy)
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', handle_click)
    plt.show()


def redraw(level, x_grid, y_grid, binning_function, ix, iy):
    plt.clf()
    draw_level(level)
    draw_grid(x_grid, y_grid)
    draw_binning_function(x_grid, y_grid, binning_function, ix, iy)
    return


def onclick(event):
    ix, iy = event.xdata, event.ydata
    return ix, iy


def draw_level(level_data):
    plot_bounds = draw_level_overlay(level_data)
    plt.xlim(plot_bounds[0], plot_bounds[1])
    plt.ylim(plot_bounds[2], plot_bounds[3])
    plt.gca().set_aspect('equal', adjustable='box')
    return


def draw_binning_function(x_grid, y_grid, binning_function, ix, iy):
    # Center of each grid cell.
    x_centers = (x_grid[:-1] + x_grid[1:]) / 2
    y_centers = (y_grid[:-1] + y_grid[1:]) / 2
    # N x N x 2 array of cell-center coordinates.
    xx, yy = torch.meshgrid(x_centers, y_centers)
    pts = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    bin_ids = binning_function.apply(pts)

    highlighted_bin = binning_function.apply(torch.tensor([[ix, iy]]))
    print("highlighted bin:", highlighted_bin)
    for i, pt in enumerate(pts):
        bin_id = bin_ids[i]
        if bin_id == highlighted_bin:
            plt.scatter(pt[0], pt[1], color='red', marker="o", s=10)
        plt.text(pt[0], pt[1], str(bin_id.item()), fontsize=8, ha='center',
                 va='center')
    return


def draw_binning_function_random(x_grid, y_grid, binning_function):
    N_points = 30
    for _ in range(N_points):
        x = np.random.uniform(x_grid[0], x_grid[-1]) * 1.1
        y = np.random.uniform(y_grid[0], y_grid[-1]) * 1.1
        bin_id = binning_function.apply(torch.tensor([[x, y]]))
        plt.scatter(x, y, color='red', marker="o", s=10)
        plt.text(x, y, str(bin_id.item()), fontsize=8, ha='center', va='center')


if __name__ == "__main__":
    main()
