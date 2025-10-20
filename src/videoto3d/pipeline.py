"""High-level orchestration for panorama + point cloud generation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .colmap_pipeline import ColmapRunner, ColmapNotInstalledError
from .frame_extractor import extract_frames
from .panorama import (
    DEFAULT_MAX_OUTPUT_PIXELS,
    make_panorama,
    is_gpu_available as panorama_gpu_available,
)


@dataclass
class PipelineOutputs:
    frames_dir: Path
    panorama_path: Path
    dense_folder: Optional[Path]
    point_cloud_path: Optional[Path]

    def to_json(self) -> str:
        return json.dumps(
            {
                "frames_dir": str(self.frames_dir),
                "panorama_path": str(self.panorama_path),
                "dense_folder": str(self.dense_folder) if self.dense_folder else None,
                "point_cloud_path": str(self.point_cloud_path)
                if self.point_cloud_path
                else None,
            },
            indent=2,
        )


def _linspace_indices(start: int, end: int, count: int) -> Iterable[float]:
    if count <= 1 or start == end:
        return [float(start)]
    step = (end - start) / (count - 1)
    return [start + step * i for i in range(count)]


@dataclass
class PipelineConfig:
    video_path: Path
    workspace: Path
    frame_rate: Optional[float] = 2.0
    panorama_crop: bool = True
    panorama_width: Optional[int] = 1600
    panorama_max_images: Optional[int] = 60
    panorama_max_output_pixels: Optional[int] = DEFAULT_MAX_OUTPUT_PIXELS
    panorama_use_gpu: bool = True
    existing_panorama: Optional[Path] = None
    run_colmap: bool = True
    colmap_binary: Optional[str] = None
    colmap_quality: str = "medium"
    use_gpu: bool = True
    max_frames: Optional[int] = 150

    @property
    def frames_dir(self) -> Path:
        return self.workspace / "frames"

    @property
    def panorama_path(self) -> Path:
        return self.workspace / "panorama" / "panorama.jpg"

    @property
    def colmap_workspace(self) -> Path:
        return self.workspace / "colmap"

    @property
    def point_cloud_path(self) -> Path:
        return self.colmap_workspace / "dense" / "0" / "fused.ply"


def run_pipeline(
    config: PipelineConfig,
    *,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> PipelineOutputs:
    def notify(message: str) -> None:
        print(message)
        if status_callback:
            status_callback(message)

    def report_progress(stage: str, value: float) -> None:
        clamped = max(0.0, min(1.0, value))
        if progress_callback:
            progress_callback(stage, clamped)

    frames_dir = config.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)

    notify("Extracting frames...")
    max_frames = config.max_frames if config.max_frames and config.max_frames > 0 else None
    if max_frames:
        notify(f"Limiting extraction to {max_frames} frame(s).")
    if config.frame_rate:
        notify(f"Target sampling rate: {config.frame_rate} fps.")
    report_progress("extract", 0.0)
    frames = list(
        extract_frames(
            config.video_path,
            frames_dir,
            target_frame_rate=config.frame_rate,
            max_frames=max_frames,
            progress_callback=lambda value: report_progress("extract", value),
        )
    )
    if not frames:
        raise RuntimeError("No frames were extracted; check the video file and sampling settings.")
    notify(f"Captured {len(frames)} frame(s).")
    report_progress("extract", 1.0)

    pano_frame_paths: Iterable[Path] = frames
    if not config.existing_panorama:
        if config.panorama_max_images and config.panorama_max_images > 0:
            limit = config.panorama_max_images
            if len(frames) > limit:
                indices = [int(round(i)) for i in _linspace_indices(0, len(frames) - 1, limit)]
                unique_indices = sorted(set(max(0, min(len(frames) - 1, idx)) for idx in indices))
                if len(unique_indices) < 2 and len(frames) >= 2:
                    unique_indices = [0, len(frames) - 1]
                pano_frame_paths = [frames[idx] for idx in unique_indices]
                notify(
                    "Panorama will use %d frames (downsampled from %d)." % (len(unique_indices), len(frames))
                )
            else:
                notify(f"Panorama will use all {len(frames)} frames.")
        else:
            notify(f"Panorama will use all {len(frames)} frames.")

    report_progress("panorama", 0.0)

    pano_path: Path
    if config.existing_panorama:
        existing = config.existing_panorama
        if not existing.exists():
            raise FileNotFoundError(f"Existing panorama not found: {existing}")
        notify(f"Using existing panorama: {existing}.")
        notify("Skipping panorama stitching stage.")
        pano_path = config.panorama_path
        pano_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            same_file = existing.resolve() == pano_path.resolve()
        except OSError:
            same_file = False
        if not same_file:
            shutil.copy2(existing, pano_path)
            notify(f"Copied panorama to {pano_path}.")
        else:
            notify("Existing panorama already resides in workspace; no copy needed.")
        report_progress("panorama", 1.0)
    else:
        notify("Building panorama...")
        if config.panorama_width:
            notify(f"Using panorama resize width: {config.panorama_width}px.")
        else:
            notify("Using full-resolution frames for panorama (slower).")

        panorama_gpu = False
        if config.panorama_use_gpu:
            if panorama_gpu_available():
                panorama_gpu = True
                notify("Panorama stage will use GPU acceleration.")
            else:
                notify("Panorama GPU requested but CUDA not available; falling back to CPU.")

        pano_path = make_panorama(
            pano_frame_paths,
            config.panorama_path,
            crop=config.panorama_crop,
            resize_width=config.panorama_width,
            use_gpu=panorama_gpu,
            progress_callback=lambda value: report_progress("panorama", value),
            max_output_pixels=config.panorama_max_output_pixels,
        )
        report_progress("panorama", 1.0)

    notify(f"Panorama saved to {pano_path}.")

    dense_folder = None
    point_cloud_path = None

    report_progress("colmap", 0.0)
    if config.run_colmap:
        try:
            runner = ColmapRunner(colmap_bin=config.colmap_binary or "colmap")
        except ColmapNotInstalledError as exc:
            notify(f"COLMAP not available: {exc}. Skipping 3D reconstruction.")
            report_progress("colmap", 1.0)
        else:
            if config.colmap_binary:
                if config.colmap_binary.lower() == "pycolmap" and runner.using_pycolmap:
                    notify("Using pycolmap Python bindings for reconstruction.")
                else:
                    notify(f"Using COLMAP executable: {config.colmap_binary}")
            elif runner.using_pycolmap:
                notify("COLMAP binary not found; using pycolmap Python bindings instead.")
            notify("Running COLMAP automatic reconstruction...")
            report_progress("colmap", 0.0)
            dense_folder = runner.automatic_reconstruction(
                frames_dir,
                config.colmap_workspace,
                quality=config.colmap_quality,
                use_gpu=config.use_gpu,
            )
            # COLMAP produces fused.ply inside dense/0 by default.
            default_cloud = dense_folder / "fused.ply"
            point_cloud_path = default_cloud if default_cloud.exists() else None
            if point_cloud_path:
                notify(f"Point cloud saved to {point_cloud_path}.")
            else:
                notify("COLMAP finished but fused point cloud not found.")
            report_progress("colmap", 1.0)
    else:
        report_progress("colmap", 1.0)

    return PipelineOutputs(
        frames_dir=frames_dir,
        panorama_path=pano_path,
        dense_folder=dense_folder,
        point_cloud_path=point_cloud_path,
    )
