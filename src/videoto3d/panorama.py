"""Panorama construction from extracted frames."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional

import cv2
import numpy as np

# Disable OpenCL to prevent driver-related crashes on some systems.
if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
    try:
        cv2.ocl.setUseOpenCL(False)
    except cv2.error:  # pragma: no cover - best-effort guard
        pass


DEFAULT_MAX_OUTPUT_PIXELS = 150_000_000


class PanoramaError(RuntimeError):
    """Raised when panorama generation fails."""


def load_images(
    frame_paths: Iterable[Path], *, resize_width: Optional[int] = None
) -> List:
    """Load images into memory with OpenCV, optionally downscaling."""

    images = []
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        if resize_width and resize_width > 0 and img.shape[1] > resize_width:
            scale = resize_width / img.shape[1]
            new_size = (resize_width, int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    if len(images) < 2:
        raise ValueError("Need at least two frames to stitch a panorama.")
    return images


def make_panorama(
    frame_paths: Iterable[Path],
    output_path: Path,
    *,
    crop: bool = False,
    mode: int = cv2.Stitcher_PANORAMA,
    resize_width: Optional[int] = None,
    use_gpu: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
    method: str = "auto",
    max_output_pixels: Optional[int] = DEFAULT_MAX_OUTPUT_PIXELS,
) -> Path:
    """Stitch the provided frames into a panorama image."""

    images = load_images(frame_paths, resize_width=resize_width)

    methods = [method] if method != "auto" else ["manual", "opencv"]
    last_error: Exception | None = None
    for selected in methods:
        try:
            if selected == "manual":
                pano = _manual_panorama(images, progress_callback)
            elif selected == "opencv":
                pano = _opencv_stitch(images, mode, use_gpu, progress_callback)
            else:
                raise ValueError(f"Unknown panorama method: {selected}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Panorama method '{selected}' failed: {exc}")
            if progress_callback:
                progress_callback(0.0)
            continue
        else:
            if crop:
                pano = _autocrop_black_borders(pano)

            if max_output_pixels:
                pano = _enforce_pixel_limit(pano, max_output_pixels)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), pano)
            if not success:
                raise RuntimeError(f"Failed to write panorama: {output_path}")
            print(f"Panorama generated using {selected} method.")
            return output_path

    if last_error:
        raise PanoramaError(f"Panorama generation failed: {last_error}") from last_error
    raise PanoramaError("Panorama generation failed with unknown error.")


def _autocrop_black_borders(image):
    """Remove empty borders by finding the largest non-black contour."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y : y + h, x : x + w]


def _create_stitcher(mode: int, try_use_gpu: bool):
    """Create a Stitcher instance, handling OpenCV API differences."""

    # OpenCV 4.8+ exposes Stitcher_create with try_use_gpu
    create = getattr(cv2, "Stitcher_create", None)
    if create:
        try:
            return create(mode, try_use_gpu)
        except TypeError:
            return create(mode)

    # Older OpenCV exposes Stitcher.create
    if hasattr(cv2, "Stitcher") and hasattr(cv2.Stitcher, "create"):
        try:
            return cv2.Stitcher.create(mode, try_use_gpu)
        except TypeError:
            return cv2.Stitcher.create(mode)  # type: ignore[call-arg]

    legacy = getattr(cv2, "createStitcher", None)
    if legacy:
        return legacy(bool(try_use_gpu))

    default_creator = getattr(cv2, "Stitcher_createDefault", None)
    if default_creator:
        return default_creator(bool(try_use_gpu))

    raise RuntimeError("OpenCV Stitcher API not available in this build.")


def _is_cuda_available() -> bool:
    if not hasattr(cv2, "cuda"):
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except cv2.error:
        return False


def is_gpu_available() -> bool:
    """Return True if OpenCV CUDA support is available."""

    return _is_cuda_available()


def _opencv_stitch(
    images: List,
    mode: int,
    use_gpu: bool,
    progress_callback: Optional[Callable[[float], None]] = None,
):
    if progress_callback:
        progress_callback(0.05)

    try_use_gpu = use_gpu and _is_cuda_available()
    stitcher = _create_stitcher(mode, try_use_gpu)
    stitcher.setPanoConfidenceThresh(0.6)
    stitcher.setSeamEstimationResol(-1)  # Use default resolution
    stitcher.setCompositingResol(-1)
    stitcher.setPanoConfidenceThresh(0.6)
    if hasattr(stitcher, "setWarper"):
        stitcher.setWarper(cv2.PyRotationWarper("cylindrical", 500))

    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise PanoramaError(f"OpenCV stitcher failed with status code {status}")

    if progress_callback:
        progress_callback(1.0)
    return pano


def _manual_panorama(
    images: List,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    if len(images) < 2:
        raise PanoramaError("Need at least two frames for manual panorama.")

    h, w = images[0].shape[:2]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    detector = _create_feature_detector()
    matcher: Optional[cv2.BFMatcher] = None

    transforms = [np.eye(3, dtype=np.float64)]
    total_pairs = len(images) - 1

    for idx in range(1, len(images)):
        kp_prev, desc_prev = detector.detectAndCompute(gray_images[idx - 1], None)
        kp_curr, desc_curr = detector.detectAndCompute(gray_images[idx], None)

        if desc_prev is None or desc_curr is None:
            raise PanoramaError("Feature detection failed.")

        if matcher is None:
            norm_type = cv2.NORM_L2 if desc_prev.dtype == np.float32 else cv2.NORM_HAMMING
            matcher = cv2.BFMatcher(norm_type, crossCheck=False)

        matches = matcher.knnMatch(desc_curr, desc_prev, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            raise PanoramaError("Not enough feature matches between frames.")

        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None:
            raise PanoramaError("Homography estimation failed.")

        transforms.append(transforms[-1] @ H)

        if progress_callback and total_pairs > 0:
            progress_callback(0.1 + 0.4 * (idx / total_pairs))

    corners = np.float32(
        [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]
    )
    transformed_corners = []
    for T in transforms:
        pts = (T @ corners.T).T
        pts /= pts[:, 2:3]
        if not np.all(np.isfinite(pts)):
            raise PanoramaError("Homography produced invalid coordinates.")
        transformed_corners.append(pts[:, :2])

    all_points = np.concatenate(transformed_corners, axis=0)
    min_x, min_y = np.floor(all_points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_points.max(axis=0)).astype(int)

    offset = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64
    )
    pano_width = int(max_x - min_x)
    pano_height = int(max_y - min_y)
    if pano_width <= 0 or pano_height <= 0:
        raise PanoramaError("Computed panorama dimensions are invalid.")

    accumulator = np.zeros((pano_height, pano_width, 3), dtype=np.float32)
    weight = np.zeros((pano_height, pano_width), dtype=np.float32)

    base_mask = np.ones((h, w), dtype=np.float32)

    for idx, (image, T) in enumerate(zip(images, transforms)):
        warp_matrix = offset @ T
        warped = cv2.warpPerspective(image, warp_matrix, (pano_width, pano_height))
        mask = cv2.warpPerspective(base_mask, warp_matrix, (pano_width, pano_height))

        accumulator += warped.astype(np.float32) * mask[..., None]
        weight += mask

        if progress_callback:
            progress_callback(0.5 + 0.5 * ((idx + 1) / len(images)))

    weight[weight == 0] = 1.0
    panorama = accumulator / weight[..., None]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    return panorama


def _create_feature_detector():
    try:
        return cv2.SIFT_create()
    except AttributeError:
        return cv2.ORB_create(4000)


def _enforce_pixel_limit(image: np.ndarray, max_pixels: int) -> np.ndarray:
    """Downscale large panoramas to avoid Pillow decompression limits."""

    height, width = image.shape[:2]
    total_pixels = height * width
    if total_pixels <= max_pixels or max_pixels <= 0:
        return image

    scale = (max_pixels / float(total_pixels)) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    print(
        "Panorama exceeds %d pixels (actual: %d); downscaling to %dx%d." %
        (max_pixels, total_pixels, new_width, new_height)
    )
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
