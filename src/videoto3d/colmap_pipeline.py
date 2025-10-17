"""Helpers for driving COLMAP from Python."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


class ColmapNotInstalledError(RuntimeError):
    """Raised when COLMAP binary cannot be located."""


class ColmapRunner:
    """Thin wrapper around the COLMAP command line tools."""

    def __init__(self, *, colmap_bin: str = "colmap") -> None:
        self.colmap_bin = colmap_bin
        if not shutil.which(self.colmap_bin):
            raise ColmapNotInstalledError(
                "COLMAP executable not found. Install COLMAP and ensure it is "
                "on your PATH or pass the explicit binary path."
            )

    def _run(self, args: List[str], *, cwd: Optional[Path] = None) -> None:
        cmd = [self.colmap_bin] + args
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "COLMAP command failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    def automatic_reconstruction(
        self,
        images_dir: Path,
        workspace: Path,
        *,
        quality: str = "medium",
        use_gpu: bool = True,
        data_type: str = "photometric",
    ) -> Path:
        """Run the COLMAP automatic reconstructor over ``images_dir``."""

        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        args = [
            "automatic_reconstructor",
            f"--image_path={images_dir}",
            f"--workspace_path={workspace}",
            f"--quality={quality}",
            f"--data_type={data_type}",
        ]
        if not use_gpu:
            args.append("--Mapper.ba_gpu=0")
            args.append("--SiftExtraction.use_gpu=0")
            args.append("--SiftMatching.use_gpu=0")
            args.append("--PatchMatchStereo.use_gpu=0")
        else:
            args.append("--Mapper.ba_gpu=1")
            args.append("--SiftExtraction.use_gpu=1")
            args.append("--SiftMatching.use_gpu=1")
            args.append("--PatchMatchStereo.use_gpu=1")

        self._run(args)

        dense_folder = workspace / "dense" / "0"
        if not dense_folder.exists():
            raise RuntimeError("COLMAP finished but dense reconstruction folder missing.")
        return dense_folder

    def export_model(
        self,
        model_dir: Path,
        output_path: Path,
        *,
        format: str = "PLY",
    ) -> Path:
        """Export a sparse or dense COLMAP model to a standard format."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "model_converter",
            f"--input_path={model_dir}",
            f"--output_path={output_path}",
            f"--output_type={format}",
        ]
        self._run(args)
        if not output_path.exists():
            raise RuntimeError(f"Model export failed: {output_path}")
        return output_path
