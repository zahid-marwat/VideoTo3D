"""Helpers for driving COLMAP from Python."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


class ColmapNotInstalledError(RuntimeError):
    """Raised when COLMAP binary cannot be located."""


try:
    import pycolmap  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pycolmap = None


class ColmapRunner:
    """Thin wrapper around the COLMAP command line tools."""

    def __init__(self, *, colmap_bin: str = "colmap") -> None:
        self.colmap_bin: Optional[str] = None
        self._mode: str = "cli"

        if colmap_bin.lower() == "pycolmap":
            if pycolmap is None:
                raise ColmapNotInstalledError(
                    "pycolmap is not installed. Install it via 'pip install pycolmap' or provide a COLMAP binary."
                )
            self._mode = "pycolmap"
            return

        if _is_explicit_path(colmap_bin):
            executable = Path(colmap_bin)
            if not executable.exists():
                raise ColmapNotInstalledError(
                    f"COLMAP executable not found at: {executable}"
                )
            if not os.access(executable, os.X_OK):
                raise ColmapNotInstalledError(
                    f"COLMAP executable is not runnable: {executable}"
                )
            self.colmap_bin = str(executable)
        else:
            resolved = shutil.which(colmap_bin)
            if resolved:
                self.colmap_bin = resolved
            elif pycolmap is not None:
                self._mode = "pycolmap"
            else:
                raise ColmapNotInstalledError(
                    "COLMAP executable not found. Install COLMAP and ensure it is "
                    "on your PATH, pass the explicit binary path, or install pycolmap."
                )

    def _run(self, args: List[str], *, cwd: Optional[Path] = None) -> None:
        if not self.colmap_bin:
            raise RuntimeError("CLI COLMAP runner invoked without executable path.")
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

        if self._mode == "pycolmap":
            self._automatic_reconstruction_pycolmap(
                images_dir,
                workspace,
                quality=quality,
                use_gpu=use_gpu,
                data_type=data_type,
            )
        else:
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
        if self._mode == "pycolmap":
            if pycolmap is None:
                raise ColmapNotInstalledError(
                    "pycolmap is not installed. Please install it or provide the COLMAP binary."
                )
            reconstruction = pycolmap.Reconstruction()
            reconstruction.read(str(model_dir))
            if format.upper() == "PLY":
                reconstruction.export_PLY(str(output_path))
            elif format.upper() == "BIN":
                reconstruction.write(str(output_path))
            elif format.upper() == "TXT":
                reconstruction.write_text(str(output_path))
            else:
                raise ValueError(f"Unsupported export format for pycolmap: {format}")
        else:
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

    def _automatic_reconstruction_pycolmap(
        self,
        images_dir: Path,
        workspace: Path,
        *,
        quality: str,
        use_gpu: bool,
        data_type: str,
    ) -> None:
        if pycolmap is None:
            raise ColmapNotInstalledError(
                "pycolmap is not installed. Please install it or provide the COLMAP binary."
            )

        normalized_data_type = data_type.lower()
        if normalized_data_type != "photometric":
            raise ValueError("pycolmap backend currently supports only photometric data type")

        image_names = _collect_image_names(images_dir)
        if not image_names:
            raise RuntimeError(f"No images found for COLMAP in {images_dir}")

        workspace.mkdir(parents=True, exist_ok=True)
        database_path = workspace / "database.db"
        sparse_dir = workspace / "sparse"
        dense_root = workspace / "dense"
        dense_workspace = dense_root / "0"

        for path in (database_path,):
            if path.exists():
                path.unlink()
        for folder in (sparse_dir, dense_workspace):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

        has_cuda = bool(pycolmap.has_cuda)
        use_gpu_effective = use_gpu and has_cuda
        if use_gpu and not has_cuda:
            print("pycolmap: CUDA not available, falling back to CPU execution.")

        device = pycolmap.Device.cuda if use_gpu_effective else pycolmap.Device.cpu
        gpu_index = "0" if use_gpu_effective else "-1"

        quality_presets = _quality_presets()
        presets = quality_presets.get(quality.lower(), quality_presets["medium"])

        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "SIMPLE_RADIAL"

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.use_gpu = use_gpu_effective
        sift_options.gpu_index = gpu_index
        sift_options.max_num_features = presets.sift_max_features
        sift_options.peak_threshold = presets.sift_peak_threshold

        pycolmap.extract_features(
            str(database_path),
            str(images_dir),
            image_names=image_names,
            camera_mode=pycolmap.CameraMode.AUTO,
            camera_model="SIMPLE_RADIAL",
            reader_options=reader_options,
            sift_options=sift_options,
            device=device,
        )

        matching_sift_opts = pycolmap.SiftMatchingOptions()
        matching_sift_opts.use_gpu = use_gpu_effective
        matching_sift_opts.gpu_index = gpu_index
        matching_sift_opts.max_num_matches = presets.match_max_num_matches

        matching_opts = pycolmap.ExhaustiveMatchingOptions()

        pycolmap.match_exhaustive(
            str(database_path),
            sift_options=matching_sift_opts,
            matching_options=matching_opts,
            device=device,
        )

        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_num_matches = presets.mapper_min_num_matches
        mapper_options.ba_use_gpu = use_gpu_effective
        mapper_options.ba_gpu_index = gpu_index

        reconstructions = pycolmap.incremental_mapping(
            str(database_path),
            str(images_dir),
            str(sparse_dir),
            options=mapper_options,
        )

        if not reconstructions:
            raise RuntimeError("pycolmap incremental mapping produced no reconstructions")

        reconstruction_id, reconstruction = max(
            reconstructions.items(), key=lambda item: item[1].num_reg_images()
        )

        sparse_output = sparse_dir / str(reconstruction_id)
        sparse_output.mkdir(parents=True, exist_ok=True)
        reconstruction.write(str(sparse_output))

        undistort_output = dense_workspace
        pycolmap.undistort_images(
            str(undistort_output),
            str(sparse_output),
            str(images_dir),
        )

        patch_match_opts = pycolmap.PatchMatchOptions()
        patch_match_opts.gpu_index = gpu_index
        patch_match_opts.num_iterations = presets.patchmatch_num_iterations
        if hasattr(patch_match_opts, "use_gpu"):
            patch_match_opts.use_gpu = use_gpu_effective

        fused_path = undistort_output / "fused.ply"
        patchmatch_completed = True

        try:
            pycolmap.patch_match_stereo(
                str(undistort_output),
                "COLMAP",
                "option-all",
                patch_match_opts,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "requires CUDA" in message or "No CUDA" in message:
                print(
                    "pycolmap: PatchMatch stereo requires CUDA. Dense reconstruction will be "
                    "skipped and the sparse point cloud will be exported instead."
                )
                patchmatch_completed = False
            else:
                raise RuntimeError(f"PatchMatch stereo failed via pycolmap: {exc}")

        if patchmatch_completed:
            fusion_opts = pycolmap.StereoFusionOptions()
            fusion_opts.max_reproj_error = presets.fusion_max_reproj_error

            fused = pycolmap.stereo_fusion(
                str(fused_path),
                str(undistort_output),
                "COLMAP",
                "option-all",
                normalized_data_type,
                fusion_opts,
            )
            if fused is None:
                raise RuntimeError("pycolmap stereo fusion failed")
        else:
            fused_path.parent.mkdir(parents=True, exist_ok=True)
            reconstruction.export_PLY(str(fused_path))
            print(
                f"pycolmap: Sparse reconstruction exported to {fused_path} as a fallback point cloud."
            )

    @property
    def using_pycolmap(self) -> bool:
        return self._mode == "pycolmap"


def _is_explicit_path(command: str) -> bool:
    return any(sep in command for sep in ("/", "\\")) or Path(command).is_absolute()


class _PycolmapQualityPreset:
    def __init__(
        self,
        *,
        sift_max_features: int,
        sift_peak_threshold: float,
        match_max_num_matches: int,
        mapper_min_num_matches: int,
        patchmatch_num_iterations: int,
        fusion_max_reproj_error: float,
    ) -> None:
        self.sift_max_features = sift_max_features
        self.sift_peak_threshold = sift_peak_threshold
        self.match_max_num_matches = match_max_num_matches
        self.mapper_min_num_matches = mapper_min_num_matches
        self.patchmatch_num_iterations = patchmatch_num_iterations
        self.fusion_max_reproj_error = fusion_max_reproj_error


def _quality_presets() -> dict[str, _PycolmapQualityPreset]:
    return {
        "low": _PycolmapQualityPreset(
            sift_max_features=4000,
            sift_peak_threshold=0.04,
            match_max_num_matches=10000,
            mapper_min_num_matches=15,
            patchmatch_num_iterations=3,
            fusion_max_reproj_error=2.0,
        ),
        "medium": _PycolmapQualityPreset(
            sift_max_features=8000,
            sift_peak_threshold=0.02,
            match_max_num_matches=20000,
            mapper_min_num_matches=20,
            patchmatch_num_iterations=5,
            fusion_max_reproj_error=1.0,
        ),
        "high": _PycolmapQualityPreset(
            sift_max_features=16000,
            sift_peak_threshold=0.01,
            match_max_num_matches=40000,
            mapper_min_num_matches=25,
            patchmatch_num_iterations=7,
            fusion_max_reproj_error=0.7,
        ),
    }


def _collect_image_names(images_dir: Path) -> Sequence[str]:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(
        str(path.name)
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_suffixes
    )


def _run_subprocess(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
