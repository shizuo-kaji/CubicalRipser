"""Python package for CubicalRipser bindings and utilities.

This package exposes the pybind implementation from ``. _cripser`` as a
stable import path ``cripser.computePH`` and provides helpers in
``cripser.utils``.
"""

from .utils import *
from ._cripser import computePH, __version__  # type: ignore
from .image_loader import (
    detect_format,
    load_dipha_complex,
    save_dipha_complex,
    load_perseus,
    save_perseus,
    load_image,
    load_series,
    save_image,
)
from .transform import SUPPORTED_TRANSFORMS, binarize, apply_transform, preprocess_image
from .vectorization import create_PH_histogram_volume, persistence_image
try:
    from tcripser import computePH as computePH_T
except ImportError:
    ValueError(
        "tcripser is not installed. Please install it to use the T-construction."
    )

try:
    from .torch_utils import compute_ph_torch, finite_lifetimes
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    def _torch_required(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for `compute_ph_torch` and `finite_lifetimes`."
        )

    compute_ph_torch = _torch_required
    finite_lifetimes = _torch_required

try:
    from .plots import plot_diagrams
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise

    def _matplotlib_required(*args, **kwargs):
        raise ImportError("matplotlib is required for `plot_diagrams`.")

    plot_diagrams = _matplotlib_required

__all__ = ["computePH", "computePH_T",
    "__version__", "compute_ph",
    "dual_embedding",
    "to_gudhi_diagrams",
    "to_gudhi_persistence",
    "group_by_dim",
    "detect_format",
    "load_dipha_complex",
    "save_dipha_complex",
    "load_perseus",
    "save_perseus",
    "load_image",
    "load_series",
    "save_image",
    "SUPPORTED_TRANSFORMS",
    "binarize",
    "apply_transform",
    "preprocess_image",
    "create_PH_histogram_volume",
    "persistence_image",
    "compute_ph_torch",
    "finite_lifetimes",
    "plot_diagrams"]
