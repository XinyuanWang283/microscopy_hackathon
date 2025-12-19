#!/usr/bin/env python3
"""
Prepare MoS2 Nanowire patches for DDPM training.

This script finds the largest 2D dataset in each HDF5 file, applies
percentile normalization, extracts fixed-size patches, and saves them
as PNG/TIFF/NPY for easy ingestion in PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from scipy.ndimage import gaussian_filter, zoom
except ImportError:  # pragma: no cover - optional dependency
    gaussian_filter = None
    zoom = None

DEFAULT_EXCLUDE_DIRS: Tuple[str, ...] = ("kmr_doped", "MoS2_0606_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DDPM training patches from MoS2 Nanowire HDF5 files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "MoS2_0510_1",
        help="Root directory containing HDF5 files (default: MoS2_0510_1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for patches (default: ./ddpm_training_data_128).",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "tif", "npy"),
        default="tif",
        help="Output format for patches (default: tif).",
    )
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size.")
    parser.add_argument("--stride", type=int, default=64, help="Stride between patches.")
    parser.add_argument(
        "--p-low",
        type=float,
        default=2.0,
        help="Low percentile for normalization (default: 2).",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=98.0,
        help="High percentile for normalization (default: 98).",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=0.1,
        help="Drop patches with std below this threshold (default: 0.1).",
    )
    parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=0.02,
        help="Skip images with (p_high - p_low) below this threshold.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.5,
        help="Gaussian blur sigma to reduce noise (0 disables).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample images to a target pixel size to align lattice periodicity.",
    )
    parser.add_argument(
        "--target-pixel-size",
        type=float,
        default=None,
        help="Target pixel size (nm/px) used when --resample is set.",
    )
    parser.add_argument(
        "--min-pixel-size",
        type=float,
        default=None,
        help="Skip images with pixel size below this value.",
    )
    parser.add_argument(
        "--max-pixel-size",
        type=float,
        default=None,
        help="Skip images with pixel size above this value.",
    )
    parser.add_argument(
        "--resample-order",
        type=int,
        default=1,
        help="Interpolation order for resampling (default: 1).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=list(DEFAULT_EXCLUDE_DIRS),
        help="Directory name to exclude (can be repeated).",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Process files in excluded directories.",
    )
    return parser.parse_args()


def normalize_percentile(
    image: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> Tuple[np.ndarray, float, float]:
    if p_low < 0 or p_high > 100 or p_low >= p_high:
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100.")

    low_val = np.nanpercentile(image, p_low)
    high_val = np.nanpercentile(image, p_high)

    if not np.isfinite(low_val) or not np.isfinite(high_val) or high_val <= low_val:
        return np.zeros_like(image, dtype=np.float32), low_val, high_val

    clipped = np.clip(image, low_val, high_val)
    normalized = (clipped - low_val) / (high_val - low_val)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return normalized.astype(np.float32), low_val, high_val


def find_largest_2d_dataset(h5_file: h5py.File) -> Optional[h5py.Dataset]:
    best_name = None
    best_size = 0

    def visitor(name: str, obj: h5py.Dataset) -> None:
        nonlocal best_name, best_size
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
            if not np.issubdtype(obj.dtype, np.number):
                return
            size = obj.size
            if size > best_size:
                best_size = size
                best_name = name

    h5_file.visititems(visitor)
    if best_name is None:
        return None
    return h5_file[best_name]


def infer_pixel_size(dataset: h5py.Dataset) -> Optional[Tuple[float, float]]:
    parent = dataset.parent
    if not isinstance(parent, h5py.Group):
        return None
    if "x" not in parent or "y" not in parent:
        return None
    x_coords = parent["x"][()]
    y_coords = parent["y"][()]
    if x_coords.ndim != 1 or y_coords.ndim != 1:
        return None
    if x_coords.size < 2 or y_coords.size < 2:
        return None
    height, width = dataset.shape
    pixel_size_x = float((x_coords.max() - x_coords.min()) / width)
    pixel_size_y = float((y_coords.max() - y_coords.min()) / height)
    return pixel_size_x, pixel_size_y


def iter_h5_files(data_root: Path) -> Iterable[Path]:
    return sorted(data_root.rglob("*.h5"))


def is_excluded(path: Path, exclude_dirs: Sequence[str]) -> bool:
    parts = set(path.parts)
    return any(ex_dir in parts for ex_dir in exclude_dirs)


def save_patch(patch: np.ndarray, output_path: Path, output_format: str) -> None:
    if output_format == "png":
        patch_uint8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(patch_uint8, mode="L").save(output_path)
    elif output_format == "tif":
        Image.fromarray(patch.astype(np.float32), mode="F").save(output_path)
    elif output_format == "npy":
        np.save(output_path, patch.astype(np.float32))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def extract_patches(
    image: np.ndarray,
    patch_size: int,
    stride: int,
    std_threshold: float,
    output_dir: Path,
    start_idx: int,
    output_format: str,
) -> int:
    height, width = image.shape
    saved = 0
    extension = "tif" if output_format == "tif" else output_format
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            if std_threshold > 0 and float(np.std(patch)) < std_threshold:
                continue
            output_path = output_dir / f"patch_{start_idx + saved:06d}.{extension}"
            save_patch(patch, output_path, output_format=output_format)
            saved += 1
    return saved


def process_dataset(
    data_root: Path,
    output_dir: Path,
    patch_size: int,
    stride: int,
    p_low: float,
    p_high: float,
    std_threshold: float,
    min_dynamic_range: float,
    gaussian_sigma: float,
    exclude_dirs: Sequence[str],
    output_format: str,
    resample: bool,
    target_pixel_size: Optional[float],
    min_pixel_size: Optional[float],
    max_pixel_size: Optional[float],
    resample_order: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_files = iter_h5_files(data_root)

    patch_idx = 0
    processed = 0
    pixel_sizes = []

    if resample and target_pixel_size is None:
        for h5_path in h5_files:
            if exclude_dirs and is_excluded(h5_path, exclude_dirs):
                continue
            try:
                with h5py.File(h5_path, "r") as h5_file:
                    dataset = find_largest_2d_dataset(h5_file)
                    if dataset is None:
                        continue
                    inferred = infer_pixel_size(dataset)
                    if inferred is None:
                        continue
                    pixel_sizes.append(inferred[0])
            except Exception:
                continue
        if pixel_sizes:
            target_pixel_size = float(np.median(pixel_sizes))
            print(f"Auto target pixel size set to {target_pixel_size:.5f} nm/px")
        else:
            raise RuntimeError("Unable to infer pixel size for resampling.")

    for h5_path in tqdm(h5_files, desc="Processing H5 files"):
        if exclude_dirs and is_excluded(h5_path, exclude_dirs):
            continue

        try:
            with h5py.File(h5_path, "r") as h5_file:
                dataset = find_largest_2d_dataset(h5_file)
                if dataset is None:
                    continue

                image = np.asarray(dataset, dtype=np.float32)
                pixel_size = infer_pixel_size(dataset)

        except Exception as exc:
            print(f"Failed to read {h5_path}: {exc}")
            continue

        if pixel_size is None and (resample or min_pixel_size or max_pixel_size):
            print(f"Skipping {h5_path}: missing pixel size metadata.")
            continue

        if pixel_size is not None:
            px = pixel_size[0]
            if min_pixel_size is not None and px < min_pixel_size:
                print(f"Skipping {h5_path}: pixel size {px:.5f} below min.")
                continue
            if max_pixel_size is not None and px > max_pixel_size:
                print(f"Skipping {h5_path}: pixel size {px:.5f} above max.")
                continue

        if resample:
            if zoom is None:
                raise RuntimeError(
                    "scipy is required for resampling. Install scipy or disable --resample."
                )
            if pixel_size is None or target_pixel_size is None:
                print(f"Skipping {h5_path}: cannot resample without pixel size.")
                continue
            scale_y = pixel_size[1] / target_pixel_size
            scale_x = pixel_size[0] / target_pixel_size
            image = zoom(image, zoom=(scale_y, scale_x), order=resample_order)

        if image.shape[0] < patch_size or image.shape[1] < patch_size:
            print(f"Skipping {h5_path}: image smaller than patch size.")
            continue

        normalized, low_val, high_val = normalize_percentile(
            image, p_low=p_low, p_high=p_high
        )
        if (
            not np.isfinite(low_val)
            or not np.isfinite(high_val)
            or high_val <= low_val
        ):
            print(f"Skipping {h5_path}: invalid percentile range.")
            continue
        if (high_val - low_val) < min_dynamic_range:
            print(
                f"Skipping {h5_path}: dynamic range {high_val - low_val:.4f} too low."
            )
            continue

        if gaussian_sigma > 0:
            if gaussian_filter is None:
                raise RuntimeError(
                    "scipy is required for gaussian blur. Install scipy or set "
                    "--gaussian-sigma 0."
                )
            normalized = gaussian_filter(normalized, sigma=gaussian_sigma)

        saved = extract_patches(
            normalized,
            patch_size=patch_size,
            stride=stride,
            std_threshold=std_threshold,
            output_dir=output_dir,
            start_idx=patch_idx,
            output_format=output_format,
        )
        patch_idx += saved
        processed += 1

    print(
        f"Done. Processed {processed} files and saved {patch_idx} patches to {output_dir}."
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (Path.cwd() / "ddpm_training_data_128")

    exclude_dirs = [] if args.include_excluded else args.exclude_dir
    process_dataset(
        data_root=args.data_root,
        output_dir=output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        p_low=args.p_low,
        p_high=args.p_high,
        std_threshold=args.std_threshold,
        min_dynamic_range=args.min_dynamic_range,
        gaussian_sigma=args.gaussian_sigma,
        exclude_dirs=exclude_dirs,
        output_format=args.output_format,
        resample=args.resample,
        target_pixel_size=args.target_pixel_size,
        min_pixel_size=args.min_pixel_size,
        max_pixel_size=args.max_pixel_size,
        resample_order=args.resample_order,
    )


if __name__ == "__main__":
    main()
