# Roblox design tool (robloxTestHarness)

The front end of the design tool: a Roblox Studio plugin that loads obstacle
courses, sends solvability queries to the backend, and plays back the
trajectories it finds. The backend is `scripts/slider_backend.py` in this
repository.

## 1. Start the backend

On a Linux machine with an NVIDIA GPU, install this repository first (see the
top-level [README](../README.md)). Then, from the repository root:

```bash
python scripts/slider_backend.py --gpu-sim \
    --ckpt-path ckpts/policy.pth \
    --host 0.0.0.0 --port 3000
```

`--method PBGE` runs PBGE instead of the default search (GE-PS).

## 2. Set up Roblox Studio

1. Create a Roblox account and install [Roblox Studio](https://create.roblox.com/).
2. Install [Rojo](https://rojo.space/docs/v7/getting-started/installation/) 7.4.4
   (the version in `ObbyTest/aftman.toml`) and the
   [Rojo Studio plugin](https://create.roblox.com/store/asset/6415005344/Rojo-7).
3. In Studio, create a new place. In the Explorer, delete everything in
   `Workspace` except `Camera` and `Terrain`.
4. Save the place. Then set `File > Game Settings > Security > Allow HTTP Requests`.
5. In `ObbyTest/`, run `rojo serve`, and connect from the Rojo plugin in Studio.

## 3. Install the design-tool plugin

1. In `ObbyTest/src/ServerStorage/ParamSlidersPlugin/ObbyEditorPlugin.server.lua`,
   set `backendUrl` (line 2) to the address of the backend machine, for example
   `http://192.0.2.10:3000`.
2. Enable `File > Studio Settings > Debugging > Plugin debugging enabled`.
3. In the Explorer, right-click
   `ServerStorage/ParamSlidersPlugin/ObbyEditorPlugin.server.lua` and select
   **Save as Local Plugin...**. Studio saves a copy of the script, so save it
   again after each change to the file.

## 4. Use the tool

1. Press **Play**, then switch to **Server** mode.
2. Open **Plugins > Obby Editor**.
3. Load an obstacle course from the **Select obstacle...** menu, and choose a
   search method from the **Select agent...** menu.

## License

MIT; see [`LICENSE`](LICENSE).
