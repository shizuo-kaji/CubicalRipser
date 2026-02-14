#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert image/volume files between NumPy and common formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
from typing import Any

import numpy as np

from cripser.image_loader import load_dipha_complex, load_image, load_series, save_image
from cripser.transform import SUPPORTED_TRANSFORMS, preprocess_image


_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


def _looks_like_series_input(inputs: list[str]) -> bool:
    if len(inputs) > 1:
        return True
    if len(inputs) != 1:
        return False
    one = inputs[0]
    return any(ch in one for ch in "*?[]") or Path(one).is_dir()


def _load_dipha_persistence(path: str) -> np.ndarray:
    """Load DIPHA persistence output (.output/.diagram) as (n, 3)."""
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise ValueError(f"File too short for DIPHA persistence format: {path}")

    _magic, _dtype, n_pairs = struct.unpack_from("qqq", data, 0)
    offset = 8 * 3
    expected_bytes = offset + n_pairs * (8 + 8 + 8)  # qdd per pair
    if len(data) < expected_bytes:
        raise ValueError(f"Truncated DIPHA persistence file: {path}")

    raw = struct.unpack_from("qdd" * n_pairs, data, offset)
    return np.asarray(raw, dtype=np.float64).reshape((-1, 3))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="img2npy",
        description="Convert image/volume files to/from NumPy, DIPHA complex, DICOM, NRRD, etc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Positional arguments are: one or more input paths, followed by output path.\n"
            "Examples:\n"
            "  python demo/img2npy.py input.jpg output.npy\n"
            "  python demo/img2npy.py input00.dcm input01.dcm input02.dcm volume.npy\n"
            "  python demo/img2npy.py img.npy img.complex\n"
            "  python demo/img2npy.py img.complex img.npy\n"
            "  python demo/img2npy.py result.output result.npy\n"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input file(s) followed by output file",
    )

    parser.add_argument("--scaling_factor", "-sf", type=float, default=1.0)
    parser.add_argument("--forceSpacing", "-fs", type=float, default=None)
    parser.add_argument("--tile", "-tl", type=int, default=1)
    parser.add_argument(
        "--transform",
        "-tr",
        choices=list(SUPPORTED_TRANSFORMS),
        default=None,
        help="Apply preprocessing transform",
    )
    parser.add_argument("--transpose", "-tp", type=int, nargs="*", default=None)
    parser.add_argument(
        "--origin",
        "-o",
        type=int,
        nargs="*",
        default=None,
        help="Origin for radial/geodesic transform (z,y,x for 3D)",
    )
    parser.add_argument(
        "--origin_mask",
        "-om",
        type=str,
        default=None,
        help="Mask image path for geodesic transform origin region",
    )
    parser.add_argument("--threshold", "-th", type=float, default=None)
    parser.add_argument("--threshold_upper_limit", "-thu", type=float, default=None)
    parser.add_argument("--shift_value", "-sv", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        "-d",
        type=str,
        default=None,
        choices=[None, "uint8", "uint16", "float", "double", "float32", "float64"],
        help="Cast output dtype",
    )
    parser.add_argument(
        "--sort",
        "-s",
        action="store_true",
        help="Numeric sort of filenames when stacking series",
    )
    parser.add_argument("--zrange", type=int, nargs=2, default=None)
    parser.add_argument(
        "--input_ext",
        "-it",
        type=str,
        default=None,
        help="Optional extension filter when input is directory/glob (e.g. dcm, .png)",
    )
    return parser.parse_args()


def _resolve_dtype(name: str | None) -> np.dtype[Any] | None:
    if name is None:
        return None
    return _DTYPE_MAP[name]


def _print_stats(tag: str, arr: np.ndarray) -> None:
    print(
        f"{tag}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={np.min(arr):.6g}, max={np.max(arr):.6g}"
    )


def main() -> None:
    args = _parse_args()

    if len(args.paths) < 2:
        raise SystemExit("Provide at least one input path and one output path.")

    input_paths = args.paths[:-1]
    output_path = args.paths[-1]

    if len(input_paths) == 1:
        in_ext = Path(input_paths[0]).suffix.lower()
        if in_ext in {".output", ".diagram"}:
            if args.transform is not None:
                print("Warning: ignoring --transform for DIPHA persistence conversion.")
            out_arr = _load_dipha_persistence(input_paths[0])
            np.save(output_path, out_arr)
            print(f"Converted DIPHA persistence to NumPy: {output_path} shape={out_arr.shape}")
            return

    dicom_ref = None
    if _looks_like_series_input(input_paths):
        loaded = load_series(
            input_paths if len(input_paths) > 1 else input_paths[0],
            input_extension=args.input_ext,
            numeric_sort=args.sort,
            force_spacing=args.forceSpacing,
            squeeze=True,
            return_metadata=True,
        )
        arr, metadata_list = loaded
        for meta in metadata_list:
            if "dicom_dataset" in meta:
                dicom_ref = meta["dicom_dataset"]
                break
    else:
        input_ext = Path(input_paths[0]).suffix.lower()
        if input_ext == ".complex":
            arr = load_dipha_complex(input_paths[0], transpose_to_numpy=True)
            dicom_ref = None
        else:
            loaded = load_image(
                input_paths[0],
                force_spacing=args.forceSpacing,
                return_metadata=True,
            )
            arr, meta = loaded
            dicom_ref = meta.get("dicom_dataset")

    arr = np.asarray(arr)
    _print_stats("input", arr)

    origin_mask = None
    if args.origin_mask is not None:
        origin_mask = np.asarray(load_image(args.origin_mask))

    arr = preprocess_image(
        arr,
        transpose=tuple(args.transpose) if args.transpose is not None else None,
        zrange=tuple(args.zrange) if args.zrange is not None else None,
        scaling_factor=args.scaling_factor,
        tile=args.tile,
        transform=args.transform,
        threshold=args.threshold,
        threshold_upper_limit=args.threshold_upper_limit,
        origin=tuple(args.origin) if args.origin is not None else None,
        origin_mask=origin_mask,
        shift_value=args.shift_value,
        dtype=_resolve_dtype(args.dtype),
    )
    arr = np.asarray(arr)
    _print_stats("processed", arr)

    out_ext = Path(output_path).suffix.lower()
    if out_ext == ".dcm" and dicom_ref is None:
        raise ValueError(
            "Saving .dcm requires DICOM input metadata. "
            "Use DICOM input files as reference or save to another format."
        )

    save_kwargs: dict[str, Any] = {}
    if out_ext == ".dcm":
        save_kwargs["dicom_reference"] = dicom_ref
    save_image(arr, output_path, **save_kwargs)

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
