# CubicalRipser: Persistent Homology for 1D Time Series, 2D Images, and 3D/4D Volumes

Authors: Takeki Sudo, Kazushi Ahara (Meiji University), Shizuo Kaji (Kyushu University / Kyoto University)

CubicalRipser is an adaptation of [Ripser](http://ripser.org) by Ulrich Bauer, specialized in fast computation of persistent homology for cubical complexes.

## Overview

### Key Features
- High performance for cubical complexes up to 4D
- C++ command-line binaries and Python modules
- PyTorch integration for differentiable workflows
- Utility helpers for loading, plotting, and vectorization
- Flexible filtrations with both V- and T-constructions
- Binary coefficients (field F2)
- Optional creator/destroyer locations in outputs

## Citation
If you use this software in research, please cite:

```bibtex
@misc{2005.12692,
  author = {Shizuo Kaji and Takeki Sudo and Kazushi Ahara},
  title = {Cubical Ripser: Software for computing persistent homology of image and volume data},
  year = {2020},
  eprint = {arXiv:2005.12692}
}
```

## Contents
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Python Usage](#python-usage)
- [Command-Line Usage](#command-line-usage)
- [Input Formats](#input-formats)
- [V and T Constructions](#v-and-t-constructions)
- [Creator and Destroyer Cells](#creator-and-destroyer-cells)
- [Deep Learning Integration](#deep-learning-integration)
- [Timing Comparisons](#timing-comparisons)
- [Testing and Regression Checks](#testing-and-regression-checks)
- [Other Software for Cubical Complex PH](#other-software-for-cubical-complex-ph)
- [Release Notes](#release-notes)
- [License](#license)

## Getting Started

### Try Online
- **Google Colab Demo**: [CubicalRipser in Action](https://colab.research.google.com/github/shizuo-kaji/CubicalRipser/blob/main/demo/cubicalripser.ipynb)
- **Topological Data Analysis Tutorial**: [Hands-On Guide](https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/master/TopologicalDataAnalysisWithPython.ipynb)
- **Applications in Deep Learning**:
  - [Example 1: Homology-enhanced CNNs](https://github.com/shizuo-kaji/HomologyCNN)
  - [Example 2: Pretraining CNNs without Data](https://github.com/shizuo-kaji/PretrainCNNwithNoData)

### Quickstart (Python)
```bash
pip install -U cripser
```

```python
import numpy as np
import cripser

arr = np.load("sample/rand2d.npy")
ph = cripser.compute_ph(arr, filtration="V", maxdim=2)
print(ph[:5])
```

### Quickstart (CLI)
Build binaries first (see [Installation](#installation)), then:

```bash
./build/cubicalripser --maxdim 2 --output out.csv sample/3dimsample.txt
./build/tcubicalripser --maxdim 2 --output out_t.csv sample/3dimsample.txt
```

## Installation

### Using `pip` (recommended)
```bash
pip install -U cripser
```

If wheel compatibility is an issue on your platform:

```bash
pip uninstall -y cripser
pip install --no-binary cripser cripser
```

### Building from source
Requirements:
- Python >= 3.8
- CMake >= 3.15
- C++14+ compiler (GCC, Clang, MSVC; `src/Makefile` defaults to C++20)

Clone and initialize submodules:

```bash
git clone https://github.com/shizuo-kaji/CubicalRipser.git
cd CubicalRipser
git submodule update --init --recursive
```

Build CLI binaries:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Outputs:
- `build/cubicalripser` (V-construction)
- `build/tcubicalripser` (T-construction)

Install the Python package from source:

```bash
pip install .
```

Legacy alternative (from `src/`):

```bash
cd src
make all
```

## Python Usage

CubicalRipser works on 1D/2D/3D/4D NumPy arrays (dtype convertible to `float64`).

### Core APIs
- `cripser.computePH(...)`: low-level binding
- `cripser.compute_ph(...)`: convenience wrapper (converts essential deaths to `np.inf`)

Both support:
- `filtration="V"` or `filtration="T"`
- `maxdim`, `top_dim`, `embedded`, `location`

Example (V-construction):

```python
import numpy as np
import cripser

arr = np.load("sample/rand4d.npy")
ph = cripser.compute_ph(arr, filtration="V", maxdim=3)
```

### Output format
For 1D-3D input, each row is typically:

```text
dim, birth, death, x1, y1, z1, x2, y2, z2
```

For 4D input:

```text
dim, birth, death, x1, y1, z1, w1, x2, y2, z2, w2
```

See [Creator and Destroyer Cells](#creator-and-destroyer-cells) for interpretation of coordinates.

Notes:
- `computePH(...)` and CLI may represent essential deaths as `DBL_MAX`.
- `compute_ph(...)` converts essential deaths to `np.inf`.

### GUDHI conversion helpers
```python
dgms = cripser.to_gudhi_diagrams(ph)
persistence = cripser.to_gudhi_persistence(ph)
```

GUDHI plotting example:

```python
import gudhi as gd
gd.plot_persistence_diagram(diagrams=dgms)
```

### Differentiable PyTorch wrapper
```python
import torch
import cripser

x = torch.rand(32, 32, requires_grad=True)
ph = cripser.compute_ph_torch(x, maxdim=1, filtration="V")
loss = cripser.finite_lifetimes(ph, dim=0).sum()
loss.backward()
```

The gradient is propagated through birth/death values to creator/destroyer voxel locations. Pairing changes are discrete, so the gradient is piecewise-defined.

### Additional utilities
- plotting: `cripser.plot_diagrams(...)` (requires `matplotlib`)
- vectorization: `cripser.persistence_image(...)`, `cripser.create_PH_histogram_volume(...)`
- OT distance: `cripser.wasserstein_distance(...)` (requires `torch` and `POT`/`ot`)

### Helper Python script (`demo/cr.py`)
A convenience wrapper for quick experiments without writing Python code.

Typical capabilities:
- Accepts a single file (`.npy`, image, DICOM, etc.) or a directory of slices
- Builds 1D-4D arrays from files
- Chooses V- or T-construction
- Computes PH up to a chosen max dimension
- Writes CSV or `.npy` outputs
- Optional sorting for DICOM/sequence inputs

Help:

```bash
python demo/cr.py -h
```

Examples:

```bash
# single NumPy array
python demo/cr.py sample/rand2d.npy -o ph.csv

# increase max dimension, use T-construction
python demo/cr.py sample/rand128_3d.npy -o ph.csv --maxdim 3 --filtration T

# directory of DICOM files (sorted)
python demo/cr.py dicom/ --sort -it dcm -o ph.csv

# directory of PNG slices
python demo/cr.py slices/ -it png -o ph.csv

# save PH as NumPy for reuse
python demo/cr.py sample/rand128_3d.npy -o ph.npy

# invert intensity sign
python demo/cr.py sample/rand128_3d.npy -o ph.csv --negative
```

Selected options (see `-h` for full list):
- `--maxdim k`
- `--filtration V|T`
- `--sort`
- `-it EXT`
- `-o FILE`
- `--embedded`
- `--negative`
- `--transform ...`, `--threshold ...`, `--threshold_upper_limit ...`

## Command-Line Usage

### Basic examples
Perseus-style text input:

```bash
./build/cubicalripser --print --maxdim 2 --output out.csv sample/3dimsample.txt
```

NumPy input (1D-4D):

```bash
./build/cubicalripser --maxdim 3 --output result.csv sample/rand128_3d.npy
```

T-construction:

```bash
./build/tcubicalripser --maxdim 3 --output volume_ph.csv sample/rand128_3d.npy
```



### Common options (`cubicalripser --help`)
- `--maxdim, -m <k>`: compute up to dimension `k` (default `3`)
- `--threshold, -t <value>`: threshold for births
- `--print, -p`: print pairs to stdout
- `--embedded, -e`: Alexander dual interpretation
- `--top_dim`: top-dimensional computation using Alexander duality
- `--location, -l yes|none`: include or omit creator/destroyer coordinates
- `--algorithm, -a link_find|compute_pairs`: 0-dimensional PH method
- `--cache_size, -c <n>`: cache limit
- `--min_recursion_to_cache, -mc <n>`: recursion threshold for caching
- `--output, -o <FILE>`: write `.csv`, `.npy`, or DIPHA-style persistence binary
- `--verbose, -v`

Notes:
- CLI does not use `--filtration`; V/T are separate binaries.
- Use `--output none` to suppress output file creation.

## Input Formats

### Supported input formats (CLI)
- **NumPy (`.npy`)**
- **Perseus text (`.txt`)**: [Specification](http://people.maths.ox.ac.uk/nanda/perseus/)
- **CSV (`.csv`)**
- **DIPHA complex (`.complex`)**: [Specification](https://github.com/DIPHA/dipha#file-formats)

### Image-to-array conversion (`demo/img2npy.py`)
A helper utility converts images/volumes between multiple formats.

```bash
# image -> .npy
python demo/img2npy.py demo/img.jpg output.npy

# image series glob -> volume .npy (shell expansion)
python demo/img2npy.py input*.jpg volume.npy

# explicit files -> volume .npy
python demo/img2npy.py input00.dcm input01.dcm input02.dcm volume.npy
```

DICOM volume conversion:

```bash
python demo/img2npy.py dicom/*.dcm output.npy
```

Direct DICOM PH computation:

```bash
python demo/cr.py dicom/ --sort -it dcm -o output.csv
```

DIPHA conversions:

```bash
# NumPy -> DIPHA complex
python demo/img2npy.py img.npy img.complex

# DIPHA complex -> NumPy
python demo/img2npy.py img.complex img.npy

# DIPHA persistence output (.output/.diagram) -> NumPy
python demo/img2npy.py result.output result.npy
```

### 1D time series
A scalar time series can be treated as a 1D image, so CubicalRipser can compute its persistent homology.

For this special case, other software may be more efficient.

A related frequency-regression example is demonstrated in the [TutorialTopologicalDataAnalysis repository](https://github.com/shizuo-kaji/TutorialTopologicalDataAnalysis).

## V and T Constructions

- **V-construction**: pixels/voxels represent 0-cells (4-neighborhood in 2D)
- **T-construction**: pixels/voxels represent top-cells (8-neighborhood in 2D)

Use:
- Python: `filtration="V"` or `filtration="T"`
- CLI: `cubicalripser` (V), `tcubicalripser` (T)

By Alexander duality, the following are closely related:

```bash
./build/cubicalripser input.npy
./build/tcubicalripser --embedded input.npy
```

The difference is in the sign of filtration and treatment of permanent cycles. Here, `--embedded` converts input `I` to `-I^infty` in the paper's notation.

For details, see [Duality in Persistent Homology of Images](https://arxiv.org/abs/2005.04597) by Adelie Garin et al.

## Creator and Destroyer Cells

The creator of a cycle is the cell that gives birth to the cycle. For example, in 0-dimensional homology, the voxel with lower filtration in a component creates that class, and a connecting voxel can destroy the class with higher birth time.

Creator and destroyer cells are not unique, but they are useful for localizing cycles.

For finite lifetime in the default convention:

```text
arr[x2,y2,z2] - arr[x1,y1,z1] = death - birth = lifetime
```

where `(x1,y1,z1)` is creator and `(x2,y2,z2)` is destroyer.

With `--embedded`, creator and destroyer roles are swapped:

```text
arr[x1,y1,z1] - arr[x2,y2,z2] = death - birth = lifetime
```

Thanks to Nicholas Byrne for suggesting this convention and providing test code.

## Deep Learning Integration

The original project examples include lifetime-enhanced and histogram-style topological channels for CNNs.

Historical note: older documentation referenced `demo/stackPH.py`; equivalent functionality is now available in the Python vectorization APIs.

Example using current APIs:

```python
import numpy as np
import cripser

arr = np.load("sample/rand2d.npy")
ph = cripser.compute_ph(arr, maxdim=1)

# Persistence image (channels = selected homology dimensions)
pi = cripser.persistence_image(
    ph,
    homology_dims=(0, 1),
    n_birth_bins=32,
    n_life_bins=32,
)

# PH histogram volume (channels encode dim x life-bin x birth-bin)
hist = cripser.create_PH_histogram_volume(
    ph,
    image_shape=arr.shape,
    homology_dims=(0, 1),
    n_birth_bins=4,
    n_life_bins=4,
)
```

For practical CNN examples, see [HomologyCNN](https://github.com/shizuo-kaji/HomologyCNN).

## Timing Comparisons

Timing scripts are under `demo/check/`.

CLI timing example:

```bash
python3 demo/check/timing_cr.py sample/bonsai128.npy \
  --mode cli \
  --cubicalripser-bin build/cubicalripser \
  --tcubicalripser-bin build/tcubicalripser \
  --runs 5 --warmup 1 \
  -o timing_bonsai128.csv
```

This writes:
- `run` rows: one per timed iteration (`elapsed_seconds`)
- `summary` rows: aggregate metrics per `(binary, dataset)`
- statistics: `mean_seconds`, `std_seconds`, `min_seconds`, `max_seconds`
- metadata: `timestamp_utc`, `git_commit`, `maxdim`, `binary_path`, `input_path`

Reference comparison example:

```bash
python3 demo/check/timing_cr.py \
  --mode cli \
  --cubicalripser-bin build/cubicalripser \
  --tcubicalripser-bin build/tcubicalripser \
  --reference-csv demo/check/performance/reference_timing.csv \
  --max-slowdown 1.10 \
  --fail-on-regression \
  -o timing_current_vs_reference.csv
```

## Testing and Regression Checks

Run Python tests:

```bash
pytest
```

Reference-based checks:

```bash
# computation regression checks (CLI + Python)
python3 demo/check/check_computation.py --mode all \
  --cubicalripser-bin build/cubicalripser \
  --tcubicalripser-bin build/tcubicalripser

# benchmark and compare against reference timing
python3 demo/check/timing_cr.py \
  --mode cli \
  --cubicalripser-bin build/cubicalripser \
  --tcubicalripser-bin build/tcubicalripser \
  --reference-csv demo/check/performance/reference_timing.csv \
  --runs 3 --warmup 1 \
  -o timing_current_vs_reference.csv
```

More details: `demo/check/README.md`.

## Other Software for Cubical Complex PH
The following notes are based on limited understanding and tests and may be incomplete.

- [Cubicle](https://bitbucket.org/hubwag/cubicle/src/master/) by Hubert Wagner
  - V-construction
  - parallelized algorithms can be faster on multicore machines
  - chunked input handling can reduce memory usage

- [HomcCube](https://i-obayashi.info/software.html) by Ippei Obayashi
  - V-construction
  - integrated into HomCloud

- [DIPHA](https://github.com/DIPHA/dipha) by Ulrich Bauer and Michael Kerber
  - V-construction
  - MPI-parallelized for cluster use
  - commonly used; memory footprint can be relatively large

- [GUDHI](http://gudhi.gforge.inria.fr/) (INRIA)
  - V- and T-construction in arbitrary dimensions
  - strong documentation and usability
  - generally emphasizes usability over raw performance

- [diamorse](https://github.com/AppliedMathematicsANU/diamorse)
  - V-construction

- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/) by Vidit Nanda
  - V-construction

## Release Notes
- **v0.0.24**: Repository renamed from `CubicalRipser_3dim` to `CubicalRipser`.
  - update old remote if needed:
    ```bash
    git remote set-url origin https://github.com/shizuo-kaji/CubicalRipser.git
    ```
- **v0.0.23**: Added torch integration
- **v0.0.22**: Changed birth coordinates for T-construction to better match GUDHI for permanent cycles
- **v0.0.19**: Added support for 4D cubical complexes
- **v0.0.8**: Fixed memory leak in Python bindings (pointed out by Nicholas Byrne)
- **v0.0.7**: Speed improvements
- **v0.0.6**: Changed [birth/death location definition](#creator-and-destroyer-cells)
- **up to v0.0.5**, differences from the [original version](https://github.com/CubicalRipser/CubicalRipser_3dim):
  - optimized implementation (lower memory footprint and faster on some data)
  - improved Python usability
  - much larger practical input sizes
  - cache control
  - Alexander duality option for highest-degree PH
  - both V and T constructions
  - birth/death location output

## License
Distributed under GNU Lesser General Public License v3.0 or later. See `LICENSE`.
