"""Command-line interface helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .panorama import DEFAULT_MAX_OUTPUT_PIXELS
from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a panorama and optionally a point cloud from a video."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Output workspace directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target frame sampling rate for extraction (frames per second).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Optional hard cap on the number of frames to sample (0 = unlimited).",
    )
    parser.add_argument(
        "--pano-width",
        type=int,
        default=1600,
        help="Downscale frames to this width before stitching (0 = full resolution).",
    )
    parser.add_argument(
        "--pano-images",
        type=int,
        default=60,
        help="Maximum number of frames to feed into panorama stitching (0 = use all).",
    )
    parser.add_argument(
        "--pano-max-pixels",
        type=int,
        default=DEFAULT_MAX_OUTPUT_PIXELS,
        help="Pixel cap for the panorama image (0 = disable automatic downscaling).",
    )
    parser.add_argument(
        "--pano-existing",
        type=Path,
        help="Path to an existing panorama image to reuse (skips stitching).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable automatic panorama border cropping.",
    )
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Skip COLMAP reconstruction stage.",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="COLMAP automatic reconstructor quality preset.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force COLMAP stages to run on CPU only.",
    )
    parser.add_argument(
        "--pano-cpu",
        action="store_true",
        help="Force panorama stitching to run on CPU even if CUDA is available.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    config = PipelineConfig(
        video_path=parsed.video,
        workspace=parsed.workspace,
        frame_rate=parsed.fps,
        panorama_crop=not parsed.no_crop,
        panorama_width=parsed.pano_width or None,
        panorama_max_images=parsed.pano_images or None,
        panorama_max_output_pixels=parsed.pano_max_pixels or None,
        panorama_use_gpu=not parsed.pano_cpu,
        existing_panorama=parsed.pano_existing,
        run_colmap=not parsed.skip_colmap,
        colmap_quality=parsed.quality,
        use_gpu=not parsed.cpu,
        max_frames=parsed.max_frames or None,
    )

    outputs = run_pipeline(config)
    print("Pipeline completed. Outputs:")
    print(outputs.to_json())
