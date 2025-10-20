"""Helpers for previewing point cloud outputs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class PointCloudViewerError(RuntimeError):
    """Raised when the point cloud viewer cannot be launched."""


def _open_with_default_app(path: Path) -> str:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return "system"


def view_point_cloud(point_cloud_path: Path, *, window_name: str = "VideoTo3D Point Cloud") -> str:
    """Open the given point cloud using Open3D when available.

    Returns a string describing the backend used ("open3d" or "system").
    Raises PointCloudViewerError when both strategies fail.
    """

    point_cloud_path = Path(point_cloud_path)
    if not point_cloud_path.exists():
        raise PointCloudViewerError(f"Point cloud not found: {point_cloud_path}")

    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except ImportError:
        try:
            return _open_with_default_app(point_cloud_path)
        except Exception as exc:  # noqa: BLE001
            raise PointCloudViewerError(str(exc)) from exc

    try:
        cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    except Exception as exc:  # noqa: BLE001
        raise PointCloudViewerError(f"Failed to read point cloud: {exc}") from exc

    if cloud.is_empty():
        raise PointCloudViewerError("The point cloud contains no points.")

    try:
        o3d.visualization.draw_geometries([cloud], window_name=window_name)
        return "open3d"
    except Exception as exc:  # noqa: BLE001
        # Fall back to the default application if Open3D visualization fails.
        try:
            return _open_with_default_app(point_cloud_path)
        except Exception as fallback_exc:  # noqa: BLE001
            raise PointCloudViewerError(str(fallback_exc)) from exc
