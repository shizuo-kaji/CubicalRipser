"""Image and volume loading/saving utilities for CubicalRipser workflows.

The helpers in this module are extracted from and compatible with the legacy
`demo/dipha2npy.py` and `demo/img2npy.py` scripts, but exposed as reusable
Python APIs.
"""

from __future__ import annotations

from pathlib import Path
import copy
import glob
import re
import struct
from typing import Any, Sequence

import numpy as np

PathLike = str | Path

_DIPHA_MAGIC = 8067171840
_DIPHA_COMPLEX_TYPE = 1
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def _require_pil():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Pillow is required for image file loading/saving.") from exc
    return Image


def _require_nrrd():
    try:
        import nrrd  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pynrrd is required for .nrrd files. Install via `pip install pynrrd`.") from exc
    return nrrd


def _require_pydicom():
    try:
        import pydicom  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pydicom is required for DICOM files. Install via `pip install pydicom`.") from exc
    return pydicom


def _require_skimage_rescale():
    try:
        from skimage.transform import rescale  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-image is required for DICOM force-spacing rescaling.") from exc
    return rescale


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    digits = re.sub(r"\D", "", path.name)
    return (int(digits) if digits else -1, path.name)


def _normalise_ext(path: PathLike) -> str:
    return Path(path).suffix.lower()


def _perseus_expected_size(shape: Sequence[int]) -> int:
    size = 1
    for s in shape:
        size *= int(s)
    return size


def load_dipha_complex(path: PathLike, *, transpose_to_numpy: bool = False) -> np.ndarray:
    """Load a DIPHA `.complex` file into a NumPy array."""
    data = Path(path).read_bytes()
    magic, data_type, size, dimensions = struct.unpack_from("qqqq", data, 0)
    if magic != _DIPHA_MAGIC:
        raise ValueError(f"Invalid DIPHA magic number in {path}: {magic}")
    if data_type != _DIPHA_COMPLEX_TYPE:
        raise ValueError(f"Unsupported DIPHA data type in {path}: {data_type}")

    offset = 8 * 4
    shape = struct.unpack_from("q" * dimensions, data, offset)
    values = np.array(struct.unpack_from("d" * size, data, offset + 8 * dimensions), dtype=np.float64)
    if not transpose_to_numpy:
        return values.reshape(shape)

    if dimensions == 3:
        perm = (2, 1, 0)
    elif dimensions == 2:
        perm = (1, 0)
    else:
        return values.reshape(shape)

    transposed_shape = tuple(shape[i] for i in perm)
    return values.reshape(transposed_shape).transpose(np.argsort(perm))


def save_dipha_complex(
    arr: np.ndarray | Sequence[Any],
    path: PathLike,
    *,
    transpose_for_dipha: bool = True,
) -> None:
    """Save an array as DIPHA `.complex`."""
    data = np.asarray(arr, dtype=np.float64)
    shape = data.shape
    size = int(np.prod(shape))
    dimensions = len(shape)

    payload = data
    if transpose_for_dipha:
        if dimensions == 3:
            payload = data.transpose((2, 1, 0))
        elif dimensions == 2:
            payload = data.transpose((1, 0))

    with Path(path).open("wb") as f:
        f.write(struct.pack("qqqq", _DIPHA_MAGIC, _DIPHA_COMPLEX_TYPE, size, dimensions))
        f.write(struct.pack("q" * dimensions, *shape))
        f.write(struct.pack("d" * size, *payload.ravel()))


def load_perseus(path: PathLike, *, replace_minus_one_with: float = 0.0) -> np.ndarray:
    """Load a PERSEUS text file into a NumPy array."""
    with Path(path).open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError("Invalid PERSEUS format: insufficient header lines.")

    dim = int(lines[0])
    if dim < 1 or dim > 4:
        raise ValueError(f"PERSEUS dimension must be 1-4, got {dim}.")

    header_shape = [int(lines[i + 1]) for i in range(dim)]
    expected_size = _perseus_expected_size(header_shape)
    data_lines = lines[dim + 1 :]

    values: list[float] = []
    for line in data_lines[:expected_size]:
        val = float(line)
        values.append(replace_minus_one_with if val == -1.0 else val)

    while len(values) < expected_size:
        values.append(replace_minus_one_with)

    return np.array(values, dtype=np.float64).reshape(tuple(header_shape), order="F")


def save_perseus(arr: np.ndarray | Sequence[Any], path: PathLike) -> None:
    """Save an array to PERSEUS text format."""
    data = np.asarray(arr)
    if data.ndim < 1 or data.ndim > 4:
        raise ValueError("PERSEUS format supports arrays with dimension 1-4.")

    with Path(path).open("w", encoding="utf-8") as f:
        f.write(f"{data.ndim}\n")
        for size in data.shape:
            f.write(f"{int(size)}\n")
        for value in data.reshape(-1, order="F"):
            f.write(f"{float(value)}\n")


def _load_npz(path: PathLike, *, key: str | None = None) -> np.ndarray:
    z = np.load(path)
    if key is None:
        key = z.files[0]
    return np.asarray(z[key])


def _load_dicom(path: PathLike, *, force_spacing: float | None = None) -> tuple[np.ndarray, Any]:
    pydicom = _require_pydicom()
    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds.file_meta, "TransferSyntaxUID"):
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    if ds.file_meta.TransferSyntaxUID != pydicom.uid.ImplicitVRLittleEndian:
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = True

    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = np.asarray(ds.pixel_array) + intercept

    if force_spacing is not None and hasattr(ds, "PixelSpacing"):
        rescale = _require_skimage_rescale()
        scaling = float(ds.PixelSpacing[0]) / float(force_spacing)
        arr = rescale(arr, scaling, mode="reflect", preserve_range=True)

    return arr, ds


def load_image(
    path: PathLike,
    *,
    force_spacing: float | None = None,
    npz_key: str | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Load one file into a NumPy array.

    Supported input formats include:
    `.npy`, `.npz`, `.complex` (DIPHA), `.txt` (PERSEUS), image files,
    `.dcm`, and `.nrrd`.
    """
    p = Path(path)
    ext = p.suffix.lower()
    metadata: dict[str, Any] = {}

    if ext == ".npy":
        arr = np.load(p)
    elif ext == ".npz":
        arr = _load_npz(p, key=npz_key)
    elif ext == ".complex":
        arr = load_dipha_complex(p)
    elif ext == ".txt":
        arr = load_perseus(p)
    elif ext == ".csv":
        arr = np.loadtxt(p, delimiter=",")
    elif ext == ".nrrd":
        nrrd = _require_nrrd()
        arr, header = nrrd.read(str(p), index_order="C")
        metadata["nrrd_header"] = header
    elif ext == ".dcm":
        arr, ds = _load_dicom(p, force_spacing=force_spacing)
        metadata["dicom_dataset"] = ds
    else:
        Image = _require_pil()
        arr = np.asarray(Image.open(p).convert("L"))

    if return_metadata:
        return np.asarray(arr), metadata
    return np.asarray(arr)


def _expand_series_inputs(inputs: PathLike | Sequence[PathLike]) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        one = str(inputs)
        if any(c in one for c in "*?[]"):
            return [Path(p) for p in glob.glob(one)]
        p = Path(one)
        if p.is_dir():
            return [x for x in p.iterdir() if x.is_file()]
        return [p]

    files: list[Path] = []
    for item in inputs:
        files.extend(_expand_series_inputs(item))
    return files


def load_series(
    inputs: PathLike | Sequence[PathLike],
    *,
    input_extension: str | None = None,
    sort: bool = False,
    numeric_sort: bool = False,
    force_spacing: float | None = None,
    squeeze: bool = True,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """Load a series of files and stack them along the first axis."""
    files = _expand_series_inputs(inputs)
    if input_extension is not None:
        ext = input_extension if input_extension.startswith(".") else f".{input_extension}"
        ext = ext.lower()
        files = [f for f in files if f.suffix.lower() == ext]
    if not files:
        raise ValueError("No input files found.")

    if numeric_sort:
        files.sort(key=_numeric_sort_key)
    elif sort:
        files.sort(key=lambda x: x.name)

    arrays: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for path in files:
        loaded = load_image(
            path,
            force_spacing=force_spacing,
            return_metadata=return_metadata,
        )
        if return_metadata:
            assert isinstance(loaded, tuple)
            arr, meta = loaded
            arrays.append(arr)
            metas.append(meta)
        else:
            assert isinstance(loaded, np.ndarray)
            arrays.append(loaded)

    out = np.stack(arrays, axis=0)
    if squeeze:
        out = np.squeeze(out)

    if return_metadata:
        return out, metas
    return out


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_image_slices(arr: np.ndarray, path: PathLike) -> None:
    Image = _require_pil()
    p = Path(path)
    out_dir = p.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(arr.shape[0]):
        slice_path = out_dir / f"{i:04d}{p.suffix.lower()}"
        Image.fromarray(_to_uint8(arr[i])).save(slice_path)


def _resolve_dicom_reference(dicom_reference: PathLike | Any | None) -> Any:
    if dicom_reference is None:
        raise ValueError(
            "Saving DICOM requires `dicom_reference` (path or loaded pydicom dataset)."
        )
    if isinstance(dicom_reference, (str, Path)):
        pydicom = _require_pydicom()
        return pydicom.dcmread(str(dicom_reference), force=True)
    return dicom_reference


def _save_dicom(arr: np.ndarray, path: PathLike, *, dicom_reference: PathLike | Any | None) -> None:
    ref = _resolve_dicom_reference(dicom_reference)
    intercept = float(getattr(ref, "RescaleIntercept", 0.0))
    try:
        pixel_dtype = ref.pixel_array.dtype
    except Exception:  # pragma: no cover - fallback when pixel decoding unavailable
        pixel_dtype = np.int16

    p = Path(path)
    if arr.ndim == 3:
        out_dir = p.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(arr.shape[0]):
            ds = copy.deepcopy(ref)
            stored = np.asarray(arr[i] - intercept).astype(pixel_dtype)
            ds.PixelData = stored.tobytes()
            ds.Rows, ds.Columns = stored.shape
            ds.InstanceNumber = i + 1
            ds.save_as(str(out_dir / f"{i:04d}.dcm"), write_like_original=False)
    elif arr.ndim == 2:
        ds = copy.deepcopy(ref)
        stored = np.asarray(arr - intercept).astype(pixel_dtype)
        ds.PixelData = stored.tobytes()
        ds.Rows, ds.Columns = stored.shape
        ds.save_as(str(p), write_like_original=False)
    else:
        raise ValueError("DICOM saving expects a 2D image or 3D series.")


def save_image(
    arr: np.ndarray | Sequence[Any],
    path: PathLike,
    *,
    dicom_reference: PathLike | Any | None = None,
) -> None:
    """Save an array to a file.

    Supported output formats include:
    `.npy`, `.npz`, `.complex`, `.txt` (PERSEUS), image files, `.dcm`, `.nrrd`,
    and `.raw` (uint16 little-endian).
    """
    data = np.asarray(arr)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext == ".npy":
        np.save(p, data)
    elif ext == ".npz":
        np.savez_compressed(p, data=data)
    elif ext == ".complex":
        save_dipha_complex(data, p)
    elif ext == ".txt":
        save_perseus(data, p)
    elif ext == ".nrrd":
        nrrd = _require_nrrd()
        nrrd.write(str(p), data, index_order="C")
    elif ext == ".dcm":
        _save_dicom(data, p, dicom_reference=dicom_reference)
    elif ext == ".raw":
        np.asarray(data, dtype=np.uint16).ravel().tofile(str(p))
    elif ext in _IMAGE_EXTENSIONS or ext == ".pgm":
        Image = _require_pil()
        if data.ndim == 3:
            _save_image_slices(data, p)
        elif data.ndim == 2:
            Image.fromarray(_to_uint8(data)).save(str(p))
        else:
            raise ValueError("Image saving expects a 2D image or 3D series.")
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def detect_format(path: PathLike) -> str:
    """Return a logical format label from extension."""
    ext = _normalise_ext(path)
    if ext in {".npy", ".npz"}:
        return "numpy"
    if ext == ".complex":
        return "dipha"
    if ext == ".txt":
        return "perseus"
    if ext == ".dcm":
        return "dicom"
    if ext == ".nrrd":
        return "nrrd"
    if ext in _IMAGE_EXTENSIONS or ext in {".pgm", ".raw", ".csv"}:
        return "image"
    return "unknown"


__all__ = [
    "detect_format",
    "load_dipha_complex",
    "save_dipha_complex",
    "load_perseus",
    "save_perseus",
    "load_image",
    "load_series",
    "save_image",
]
