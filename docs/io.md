# Input and output

[Manual](README.md) · Previous: [CLI](cli.md) · Next: [Constructions](concepts.md)

## CLI input formats

| Extension | Interpretation |
| --- | --- |
| `.npy` | 1D–4D NumPy array; save as `float64` for the C++ CLI. Both C and Fortran order are supported. |
| `.csv` | Rectangular 2D numeric grid, with no header. The CLI maps columns to `x` and rows to `y`. |
| `.txt` | Perseus-style dimension and shape header followed by scalar values. |
| `.complex` | DIPHA image/cubical-complex binary. |

For consistent axis interpretation between Python and the CLI, prefer `.npy`.
Python `np.loadtxt(..., delimiter=",")` puts CSV rows on axis 0; the CLI CSV
loader uses the opposite axis naming. A persistence CSV is an **output table**,
not an input scalar image, even though both use the same extension.

```python
import numpy as np

image = np.arange(20, dtype=np.float64).reshape(4, 5)
np.save("input.npy", image)
```

See [CLI output files](cli.md#output-files) for persistence-table formats.

## Perseus and DIPHA

Perseus text starts with the number of dimensions, then one axis length per
line, then one value per line with the first axis varying fastest (Fortran
order). Helpers are `cripser.load_perseus` and `cripser.save_perseus`.

**The sentinel `-1` has special meaning:** the CLI replaces it with its
filtration threshold, while `load_perseus` replaces it with 0 by default
(configurable through `replace_minus_one_with`). Use `.npy` when `-1` is an
ordinary filtration value or when exact round trips are required.

DIPHA helpers are `cripser.load_dipha_complex` and
`cripser.save_dipha_complex`. Inspect their transpose options when exchanging
arrays with another tool; check shapes and axis order as well as barcode values.

Format references: [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/) and
[DIPHA](https://github.com/DIPHA/dipha#file-formats).

## Images and slice stacks

`load_image` reads `.npy`, `.npz`, `.csv`, Perseus `.txt`, DIPHA `.complex`,
DICOM `.dcm`, NRRD `.nrrd`, and ordinary images. Ordinary images are converted
to grayscale. Install [format-specific dependencies](installation.md#optional-dependencies).

```python
import cripser

image = cripser.load_image("image.png")
ph = cripser.compute_ph(image, maxdim=1)
```

For a volume assembled from slices:

```python
import cripser

volume = cripser.load_series(
    "slices/", input_extension="png", numeric_sort=True, squeeze=False,
)
ph = cripser.compute_ph(volume, maxdim=2)
```

`load_series` stacks along axis 0. `sort=True` sorts by filename;
`numeric_sort=True` uses numeric filename ordering. These options do not infer
anatomical order from DICOM metadata: ensure that filenames or an explicitly
ordered file list match the intended slice order. The default `squeeze=True`
removes singleton axes; disable it when preserving dimensionality matters.

`load_image(..., return_metadata=True)` returns `(array, metadata)`;
`load_series` can similarly return per-file metadata. `save_image` dispatches
by output extension; specialized helpers are available for Perseus and DIPHA.

## Preprocessing

`binarize`, `apply_transform`, and `preprocess_image` expose the demo pipeline
as library functions. Available transforms are listed in
`cripser.SUPPORTED_TRANSFORMS`: binarisation, distance/signed-distance and their
inverse forms, radial/geodesic and their inverse forms, upward, and downward.

```python
import numpy as np
import cripser

image = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
field = cripser.preprocess_image(
    image, transform="distance", threshold=0.5, dtype="float64",
)
ph = cripser.compute_ph(field, maxdim=1)
```

Preprocessing changes the filtration whose topology you measure. The pipeline
applies transpose, slice range, rescaling, tiling, transform, value shift, and
dtype conversion in that order. Distance transforms need SciPy; automatic
thresholding and rescaling need scikit-image; geodesic transforms need scikit-fmm.

## Conversion script

[`demo/img2npy.py`](../demo/img2npy.py) converts images and arrays without
computing persistence:

```bash
python demo/img2npy.py -h
python demo/img2npy.py image.jpg output.npy
python demo/img2npy.py input*.jpg volume.npy
python demo/img2npy.py input00.dcm input01.dcm input02.dcm volume.npy
python demo/img2npy.py dicom/*.dcm output.npy
python demo/img2npy.py img.npy img.complex
python demo/img2npy.py img.complex img.npy
python demo/img2npy.py result.output result.npy
```

The last form reads DIPHA persistence output (`.output` or `.diagram`), rather
than a scalar image. Globs in these commands are expanded by the shell; verify
slice ordering when filenames do not sort in acquisition order.
