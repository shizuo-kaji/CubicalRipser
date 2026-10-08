"""Scalar arrays for examples, tests and benchmarks.

Synthetic arrays are generated on the fly; real 3D and 4D volumes are
downloaded on first use by :func:`fetch`.  Every function returns a
C-contiguous ``float64`` array, which can be passed to :func:`compute_ph` or
saved with ``np.save`` as input for the command-line programs.  Coordinates
(``center``, ``points``) are array indices in NumPy axis order, as in the
coordinate columns of :func:`compute_ph`.

The shape generators return the Euclidean distance to a shape, so the
sublevel set ``arr <= t`` is the shape thickened by ``t``.  With ``t`` about
one grid unit it has the homology of the shape; e.g.
``np.where(sphere(shape) <= 0.85, 0.0, 1.0)`` is a binary hollow sphere for
both constructions.  A thinner shell can have gaps on the grid.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "uniform_noise",
    "gaussian_random_field",
    "sphere",
    "torus",
    "distance_to_points",
    "Volume",
    "VOLUMES",
    "fetch",
]


def _shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(n) for n in shape)


def _center(shape: tuple[int, ...], center: Sequence[float] | None) -> np.ndarray:
    if center is None:
        return (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    c = np.asarray(center, dtype=np.float64)
    if c.shape != (len(shape),):
        raise ValueError(f"center must have length {len(shape)}, got shape {c.shape}")
    return c


def _offsets(shape: tuple[int, ...], center: np.ndarray) -> list[np.ndarray]:
    """Per-axis ``index - center``, shaped to broadcast against ``shape``."""
    return [g - c for g, c in zip(np.ogrid[tuple(slice(0, n) for n in shape)], center)]


def uniform_noise(
    shape: int | Sequence[int],
    *,
    levels: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Independent uniform values at every grid point.

    Parameters
    - shape: array shape.
    - levels: if given, values are integers ``0, ..., levels - 1`` (stored as
      ``float64``), so many values are tied.  Otherwise values are uniform in
      ``[0, 1)`` and almost surely distinct.
    - seed: seed or ``numpy.random.Generator``.

    Returns
    - np.ndarray of ``float64``.
    """
    rng = np.random.default_rng(seed)
    shape = _shape(shape)
    if levels is None:
        return rng.random(shape)
    return rng.integers(0, levels, size=shape).astype(np.float64)


def gaussian_random_field(
    shape: int | Sequence[int],
    *,
    sigma: float = 2.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Smooth random field: white noise convolved with a Gaussian kernel.

    Local extrema and the features of the sublevel sets have a typical size
    of about ``sigma`` grid units.  The convolution is periodic (computed by
    FFT), so the field wraps around at the array boundary.

    Parameters
    - shape: array shape.
    - sigma: standard deviation of the Gaussian kernel in grid units;
      ``0`` gives white noise.
    - seed: seed or ``numpy.random.Generator``.

    Returns
    - np.ndarray of ``float64``, standardized to mean 0 and standard
      deviation 1.
    """
    rng = np.random.default_rng(seed)
    shape = _shape(shape)
    field = rng.standard_normal(shape)
    if sigma > 0:
        spectrum = np.fft.rfftn(field)
        last = len(shape) - 1
        for axis, n in enumerate(shape):
            freq = np.fft.rfftfreq(n) if axis == last else np.fft.fftfreq(n)
            gain = np.exp(-2.0 * (np.pi * sigma * freq) ** 2)
            spectrum *= gain.reshape((-1,) + (1,) * (last - axis))
        field = np.fft.irfftn(spectrum, s=shape, axes=tuple(range(len(shape))))
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def sphere(
    shape: int | Sequence[int],
    *,
    radius: float | None = None,
    center: Sequence[float] | None = None,
) -> np.ndarray:
    """Distance to a sphere of dimension ``len(shape) - 1``.

    A circle in 2D, a 2-sphere in 3D and a 3-sphere in 4D.  The persistence
    diagram has one long bar in dimension ``len(shape) - 1``: it is born when
    the thickened shell closes, below one grid unit, and dies when the
    interior fills, at about ``radius``.

    Parameters
    - shape: array shape.
    - radius: defaults to ``(min(shape) - 1) / 2``, the largest sphere
      around the default center that fits in the array.
    - center: defaults to the center of the array, ``(shape - 1) / 2``.

    Returns
    - np.ndarray of ``float64``: ``| ||x - center|| - radius |``.
    """
    shape = _shape(shape)
    c = _center(shape, center)
    if radius is None:
        radius = (min(shape) - 1) / 2.0
    dist = np.sqrt(sum(d * d for d in _offsets(shape, c)))
    return np.abs(dist - radius)


def torus(
    shape: Sequence[int],
    *,
    major_radius: float | None = None,
    minor_radius: float | None = None,
    center: Sequence[float] | None = None,
) -> np.ndarray:
    """Distance to a torus surface in a 3D array.

    The axis of revolution is parallel to the last array axis.  The
    thickened surface ``arr <= t`` is a hollow torus, with Betti numbers 1, 2
    and 1 in dimensions 0, 1 and 2, from about one grid unit until ``t``
    approaches ``minor_radius`` or ``major_radius - minor_radius``.

    Parameters
    - shape: 3D array shape.
    - major_radius: distance from the axis to the center of the tube.
    - minor_radius: radius of the tube.
      With ``a = (min(shape[0], shape[1]) - 1) / 2``, the defaults are
      ``minor_radius = min(a / 3, (shape[2] - 1) / 2)`` and
      ``major_radius = a - minor_radius``, so the torus fits in the array.
    - center: defaults to the center of the array, ``(shape - 1) / 2``.

    Returns
    - np.ndarray of ``float64``.
    """
    shape = _shape(shape)
    if len(shape) != 3:
        raise ValueError(f"torus needs a 3D shape, got {shape}")
    a = (min(shape[0], shape[1]) - 1) / 2.0
    if minor_radius is None:
        minor_radius = min(a / 3.0, (shape[2] - 1) / 2.0)
    if major_radius is None:
        major_radius = a - minor_radius
    dx, dy, dz = _offsets(shape, _center(shape, center))
    ring = np.sqrt(dx * dx + dy * dy) - major_radius
    return np.abs(np.sqrt(ring * ring + dz * dz) - minor_radius)


def distance_to_points(
    shape: int | Sequence[int],
    points: int | np.ndarray = 10,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Distance to the nearest of a set of points.

    The sublevel set ``arr <= t`` is the union of balls of radius ``t``
    centered at the points, so the persistence diagram describes clusters
    (H₀) and the loops and voids enclosed by the balls.

    Parameters
    - shape: array shape.
    - points: number of points drawn uniformly from the array's bounding box
      ``[0, n - 1]`` on each axis, or an array of shape ``(n_points,
      len(shape))`` with their coordinates.
    - seed: seed or ``numpy.random.Generator`` for drawing the points.

    Returns
    - np.ndarray of ``float64``.
    """
    shape = _shape(shape)
    if isinstance(points, (int, np.integer)):
        rng = np.random.default_rng(seed)
        points = rng.random((int(points), len(shape))) * (np.asarray(shape) - 1)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(shape) or len(points) == 0:
        raise ValueError(
            f"points must have shape (n_points, {len(shape)}) with n_points >= 1, "
            f"got {points.shape}"
        )
    out = np.full(shape, np.inf)
    for p in points:
        np.minimum(out, sum(d * d for d in _offsets(shape, p)), out=out)
    return np.sqrt(out, out=out)


@dataclass(frozen=True)
class Volume:
    """A volume that :func:`fetch` downloads.

    Attributes
    - shape: shape of the array returned by :func:`fetch`.
    - dtype: element type of the raw file (``None`` for NIfTI).
    - description: what the volume shows.
    - credit: the source to acknowledge when publishing results.
    - url: download location.
    - sha512: SHA-512 digest of the downloaded file.
    """

    shape: tuple[int, ...]
    dtype: str | None
    description: str
    credit: str
    url: str
    sha512: str


# The Open SciVis Datasets collection (https://klacansky.com/open-scivis-datasets/)
# is served over plain HTTP; the SHA-512 digests below, published with the
# collection, guarantee the content.  Raw files store x fastest, so they are
# read as C arrays indexed [z, y, x].
_SCIVIS = "http://klacansky.com/open-scivis-datasets/"
_SCIVIS_CREDIT = " (via the Open SciVis Datasets collection, klacansky.com)"

VOLUMES: dict[str, Volume] = {
    "bonsai": Volume(
        (256, 256, 256), "uint8", "CT scan of a bonsai tree.",
        "volvis.org and S. Roettger, VIS, University of Stuttgart" + _SCIVIS_CREDIT,
        _SCIVIS + "bonsai/bonsai_256x256x256_uint8.raw",
        "b34156a0ffc80ffaf84d069f3d05a40fdd999a35f05492829a2b0c13403a3147"
        "e73712b1d10c2cc34da66a59540a1632dae6adc96f3ebf3efa5d4d6c10598997",
    ),
    "foot": Volume(
        (256, 256, 256), "uint8", "Rotational C-arm X-ray scan of a human foot (tissue and bone).",
        "volvis.org and Philips Research, Hamburg, Germany" + _SCIVIS_CREDIT,
        _SCIVIS + "foot/foot_256x256x256_uint8.raw",
        "56a73bd1f694a09809688f5b0bbf0dfff0ef3e853a34c60c4d51bef02dd96fbf"
        "e5191f47c7762bdf7caab6f9a22f1d0e830652779f87c70cc519f345221edd0e",
    ),
    "skull": Volume(
        (256, 256, 256), "uint8", "Rotational C-arm X-ray scan of a phantom of a human skull.",
        "volvis.org and Siemens Medical Solutions, Forchheim, Germany" + _SCIVIS_CREDIT,
        _SCIVIS + "skull/skull_256x256x256_uint8.raw",
        "6b31b8c3c056e4fca2c908c8feffe1ad4872e1201feafe136ed22178d2593f54"
        "9c4a8328b2f6ec7d7855271f28f7a4981ffae8ced9ff21df11652abe7c0766bc",
    ),
    "aneurism": Volume(
        (256, 256, 256), "uint8",
        "Rotational C-arm X-ray scan of the arteries of the right half of a human head, "
        "with contrast agent and an aneurism.",
        "volvis.org and Philips Research, Hamburg, Germany" + _SCIVIS_CREDIT,
        _SCIVIS + "aneurism/aneurism_256x256x256_uint8.raw",
        "31f1232cbf75b0182f172375e46cc57fe013c0edaae292f2358e2477052ca974"
        "faae92efd803a2d229172b851b5ddaa461129872b89bd6ce45d43bfe595eed43",
    ),
    "engine": Volume(
        (128, 256, 256), "uint8", "CT scan of two cylinders of an engine block.",
        "volvis.org and General Electric" + _SCIVIS_CREDIT,
        _SCIVIS + "engine/engine_256x256x128_uint8.raw",
        "f17228483b1b5edb90146eac90ba979531629a2880d9e7a0493a98290da8af6a"
        "c2cf80f2c567a3d08695dc7dd854916c861dad8856b2d5721501954e913de13f",
    ),
    "mri_ventricles": Volume(
        (124, 256, 256), "uint8",
        "1.5T MRI (3D CISS) of a human head, highlighting the cavities filled with "
        "cerebrospinal fluid.",
        "volvis.org and Dirk Bartz, VCM, University of Tübingen, Germany" + _SCIVIS_CREDIT,
        _SCIVIS + "mri_ventricles/mri_ventricles_256x256x124_uint8.raw",
        "fd522a0c616d3367ec5c8f63efe918825f032882bf2e50c10d4e155c4db6a14d"
        "139a5b6a0e7cedd0cda69aaf1c89d3aad60a984a72b57fc6e2eca48f34ef54e0",
    ),
    "fuel": Volume(
        (64, 64, 64), "uint8", "Simulation of fuel injection into a combustion chamber.",
        "volvis.org and SFB 382 of the German Research Council (DFG)" + _SCIVIS_CREDIT,
        _SCIVIS + "fuel/fuel_64x64x64_uint8.raw",
        "77fdd7c657da1946bafc84e88c6b8a03ae104a79a5bdec3c7db9257480ef4bf7"
        "2551a08d22fd237c8e387dd2571b575f1a1a11f5f32b1fa4d4ef385d9fe1d613",
    ),
    "hydrogen_atom": Volume(
        (128, 128, 128), "uint8",
        "Simulated probability distribution of the electron of a hydrogen atom in a "
        "strong magnetic field.",
        "volvis.org and SFB 382 of the German Research Council (DFG)" + _SCIVIS_CREDIT,
        _SCIVIS + "hydrogen_atom/hydrogen_atom_128x128x128_uint8.raw",
        "bc80b55ffc983f41b3981433707b59f6c8b3f16cc9cd3ea18087cb9e734b702e"
        "b1ad0410f36f38881b2e2fa85617dc0858bb2d9fbd3188abb39af43ea84e3521",
    ),
    "tacc_turbulence": Volume(
        (256, 256, 256), "float32",
        "Enstrophy in one time step of an isotropic turbulence simulation.",
        "Gregory D. Abram and Gregory P. Johnson, Texas Advanced Computing Center, "
        "The University of Texas at Austin; simulation by Diego A. Donzis" + _SCIVIS_CREDIT,
        _SCIVIS + "tacc_turbulence/tacc_turbulence_256x256x256_float32.raw",
        "2206fc1368064e62a752cea90b3932289a26606a3b1552b945745f3c1c57ded8"
        "67ec8b28daa1d05bfd20438c8c3fa95460a43c37f086aaffea1f5b5742b72580",
    ),
    # OpenNeuro ds000105 (PDDL), pinned to an S3 object version.  NIfTI stores x
    # fastest; the array is indexed [x, y, z, t] as in nibabel.
    "fmri_haxby": Volume(
        (40, 64, 64, 121), None,
        "BOLD fMRI time series (x, y, z, t) of subject 1, run 1, of a visual object "
        "recognition experiment.",
        "Haxby et al., Science 293:2425 (2001); OpenNeuro ds000105, OpenfMRI project "
        "(NSF grant OCI-1131441)",
        "https://s3.amazonaws.com/openneuro.org/ds000105/sub-1/func/"
        "sub-1_task-objectviewing_run-01_bold.nii.gz?versionId=PrLtn8mz.k7O9Biw2w_KbzkkDC3GlrSl",
        "c86e303de89862331dc84d823b966b09dbf3c18d634b72ec371ba67b4055e86d"
        "79b6c6f3779171c8b3527afb567ffebbb126e8b1c60d7b7de7480270499d2f19",
    ),
}


def _data_dir(data_home: str | os.PathLike | None) -> Path:
    if data_home is None:
        data_home = os.environ.get("CRIPSER_DATA_DIR")
    if data_home is None:
        cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        data_home = Path(cache) / "cripser"
    return Path(data_home).expanduser()


def _download(url: str, path: Path, sha512: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    try:
        digest = hashlib.sha512()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha512:
            raise OSError(f"{url} does not match its SHA-512 digest; the download was discarded")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# NIfTI-1 datatype codes of the types used in this registry.
_NIFTI_DTYPES = {2: "u1", 4: "<i2", 16: "<f4", 64: "<f8", 512: "<u2"}


def _read_nifti_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    ndim, *dims = struct.unpack_from("<8h", buf, 40)
    (code,) = struct.unpack_from("<h", buf, 70)
    (offset,) = struct.unpack_from("<f", buf, 108)
    slope, inter = struct.unpack_from("<2f", buf, 112)
    shape = tuple(dims[:ndim])
    data = np.frombuffer(
        buf, dtype=_NIFTI_DTYPES[code], count=int(np.prod(shape)), offset=int(offset)
    )
    out = np.ascontiguousarray(data.reshape(shape[::-1]).T, dtype=np.float64)
    if slope != 0 and (slope, inter) != (1, 0):
        out *= slope
        out += inter
    return out


def fetch(name: str, *, data_home: str | os.PathLike | None = None) -> np.ndarray:
    """Load a real volume, downloading it on first use.

    The file is saved under ``data_home`` and checked against its SHA-512
    digest; later calls read the saved file.  See :data:`VOLUMES` for the
    available names, shapes and sources.

    Parameters
    - name: key of :data:`VOLUMES`, e.g. ``"bonsai"`` or ``"fmri_haxby"``.
    - data_home: directory for downloaded files.  Defaults to the
      ``CRIPSER_DATA_DIR`` environment variable, or else
      ``$XDG_CACHE_HOME/cripser`` (``~/.cache/cripser``).

    Returns
    - np.ndarray of ``float64`` with shape ``VOLUMES[name].shape``.  The 3D
      volumes are indexed ``[z, y, x]``; ``"fmri_haxby"`` is ``[x, y, z, t]``.
    """
    try:
        volume = VOLUMES[name]
    except KeyError:
        raise ValueError(f"Unknown volume {name!r}; available: {', '.join(VOLUMES)}") from None
    filename = urllib.parse.urlsplit(volume.url).path.rsplit("/", 1)[-1]
    path = _data_dir(data_home) / filename
    if not path.exists():
        print(f"Downloading {name} from {volume.url} to {path}", file=sys.stderr)
        _download(volume.url, path, volume.sha512)
    if volume.dtype is None:
        return _read_nifti_gz(path)
    data = np.fromfile(path, dtype=np.dtype(volume.dtype).newbyteorder("<"))
    return data.reshape(volume.shape).astype(np.float64)
