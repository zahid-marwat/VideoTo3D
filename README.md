# VideoTo3D

Pipeline that turns a single 180° video into a stitched panorama and a COLMAP-based 3D point cloud.

## Features

- Frame extraction with configurable sampling rate.
- Cylindrical panorama stitching using OpenCV.
- Optional COLMAP automatic reconstruction to produce a dense point cloud.
- Built-in point cloud viewer (uses Open3D when available, falls back to your OS default app).
- Simple CLI wrapper: `python -m videoto3d <video>` (or `python -m run_pipeline <video>`).
- Optional desktop GUI (`python -m videoto3d.gui` or simply run `start.bat`).

## Prerequisites

1. Python 3.10+.
2. Either install [COLMAP](https://colmap.github.io/install.html) and ensure the `colmap` CLI is available, **or** `pip install pycolmap` (the project now falls back to the Python bindings when the CLI binary is missing).
3. (Windows) Install Visual C++ runtime and CUDA toolkit if you plan to run COLMAP on GPU.
4. (Optional) `pip install open3d` to use the interactive point cloud viewer; without it the project falls back to your OS default viewer.

## Installation

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

COLMAP is not available via `pip`. Download a prebuilt binary or compile from source and ensure the executable directory is in your `PATH`. Alternatively, install [pycolmap](https://github.com/colmap/pycolmap) to run reconstruction directly from Python without a standalone executable.

## Usage

```pwsh
python -m videoto3d path\to\video.mp4 --workspace outputs\room --fps 2
```

Key optional flags:

- `--no-crop`: keep black borders around the panorama.
- `--skip-colmap`: generate only the panorama.
- `--colmap-bin PATH`: provide an explicit COLMAP executable instead of relying on your `PATH`. Pass `pycolmap` to force the Python bindings even when a CLI binary is present.
- `--quality {low,medium,high}`: balance speed vs accuracy for COLMAP.
- `--cpu`: disable GPU usage when running COLMAP.
- `--max-frames N`: stop sampling after `N` frames (default 150, set 0 for unlimited).
- `--pano-width W`: downscale frames to width `W` px before stitching (default 1600, set 0 for full res).
- `--pano-images N`: feed at most `N` frames into the panorama step (default 60, set 0 to use all frames).
- `--pano-max-pixels N`: automatically downscale the final panorama if it would exceed `N` pixels (default 150M, set 0 to disable).
- `--pano-existing PATH`: reuse an already stitched panorama image instead of creating a new one.
- `--pano-cpu`: force panorama stitching to stay on CPU even if CUDA is available.
- `--view-cloud`: launch the point cloud viewer after reconstruction finishes.

Outputs are written to the workspace folder:

```
workspace/
  frames/           # Extracted frame JPEGs
  panorama/
    panorama.jpg    # Final stitched panorama
  colmap/
    ...             # COLMAP project files
    dense/0/fused.ply  # Dense point cloud when reconstruction succeeds
```

## Visualization

- Panorama: open `panorama.jpg` in any image viewer or load as a skybox texture in a 3D engine.
- Point cloud: use the built-in viewer (`--view-cloud` flag or the GUI button). Install [Open3D](http://www.open3d.org/) (`pip install open3d`) for an interactive window; otherwise the OS default application opens the `.ply` file.

## GUI Workflow

Launch the GUI with `python -m videoto3d.gui` (or double-click `start.bat`). The app walks through:

1. Selecting the source video and output workspace.
2. Choosing sampling, frame limits, and panorama width (tune these if stitching is slow).
  - Use the new panorama frame cap to downsample long videos before stitching.
3. Running the pipeline with live status updates, per-step progress bars, and automatic panorama fallback logic (manual stitching with OpenCV as backup).
4. Previewing the resulting panorama and opening the generated files in external viewers.
  - The GUI enables panorama GPU acceleration automatically when OpenCV CUDA support is detected.
  - Provide a custom COLMAP executable via the new picker if the binary is not on your `PATH`. When pycolmap is installed the pipeline automatically falls back to it.
  - Extremely large panoramas are downscaled automatically so they can be previewed safely.
  - When you already have a panorama, enable *Use existing panorama* to skip the stitching stage (the GUI copies it into the workspace and continues with later steps).

## Next Steps

- Build a Three.js or Unity viewer to display the panorama and fused point cloud.
- Add lens undistortion if your camera metadata is available (OpenCV `fisheye` module).
- Integrate depth-aware panorama rendering with WebGL for a Google Street View style experience.
