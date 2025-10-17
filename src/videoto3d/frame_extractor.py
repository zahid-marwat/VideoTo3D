"""Utilities for sampling frames from input videos."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    every_n_frames: int = 10,
    target_frame_rate: Optional[float] = None,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Iterable[Path]:
    """Extract frames from video and write them to ``output_dir``.

    Args:
        video_path: Path to the video file.
        output_dir: Folder that will receive the extracted frames.
        every_n_frames: Keep one frame out of every n frames when
            ``target_frame_rate`` is not provided.
        target_frame_rate: If provided, compute a frame skipping factor that
            approximates the requested FPS. Overrides ``every_n_frames``.
        max_frames: Optional cap on the number of frames to extract.

    Yields:
        Paths to the frames written in chronological order.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if target_frame_rate and target_frame_rate > 0:
            every_n_frames = max(1, int(round(fps / target_frame_rate)))

        total_target = 0
        if frame_count > 0:
            total_target = frame_count // every_n_frames if every_n_frames > 1 else frame_count
        if max_frames and max_frames > 0:
            total_target = total_target or max_frames
            if total_target:
                total_target = min(total_target, max_frames)

        def report_progress(value: float) -> None:
            if progress_callback:
                progress_callback(max(0.0, min(1.0, value)))

        report_progress(0.0)

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n_frames == 0:
                frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                success = cv2.imwrite(str(frame_path), frame)
                if not success:
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
                saved += 1
                if total_target:
                    report_progress(min(1.0, saved / total_target))
                elif max_frames and max_frames > 0:
                    report_progress(min(1.0, saved / max_frames))
                yield frame_path
                if max_frames and saved >= max_frames:
                    break

            frame_idx += 1

        # Provide basic logging for diagnostics when running interactively.
        print(
            f"Extracted {saved} frames from {frame_count} total "
            f"(skip={every_n_frames})."
        )
        report_progress(1.0)
    finally:
        cap.release()
