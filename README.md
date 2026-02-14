# CubicalRipser: Persistent Homology for 1D Time Series, 2D Images, and 3D and 4D Volumes

Authors: Takeki Sudo, Kazushi Ahara (Meiji University), Shizuo Kaji (Kyushu University / Kyoto University)

---

## Overview
CubicalRipser is an adaptation of [Ripser](http://ripser.org) by Ulrich Bauer, specialised in fast computation of persistent homology for cubical complexes.

### Key Features
- High performance (among the fastest for cubical complexes up to 4D)
- Flexible filtrations: supports both V- and T-constructions (see: V and T Constructions)
- Binary coefficients (field F2)
- Python module and standalone CLI
- PyTorch integration for differentiable persistent homology
- Birth/death locations with creator/destroyer cells

If you use this software in research, please cite:

```bibtex
@misc{2005.12692,
Author = {Shizuo Kaji and Takeki Sudo and Kazushi Ahara},
Title = {Cubical Ripser: Software for computing persistent homology of image and volume data},
Year = {2020},
Eprint = {arXiv:2005.12692},
}
```

---

## License
Distributed under the GNU Lesser General Public License v3.0 or later.

## Getting Started

### Try Online
- **Google Colab Demo**: [CubicalRipser in Action](https://colab.research.google.com/github/shizuo-kaji/CubicalRipser/blob/main/demo/cubicalripser.ipynb)
- **Topological Data Analysis (TDA) Tutorial**: [Hands-On Guide](https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/main/TopologicalDataAnalysisWithPython.ipynb)
- **Applications in Deep Learning**:
  - [Example 1: Homology-enhanced CNNs](https://github.com/shizuo-kaji/HomologyCNN)
  - [Example 2: Pretraining CNNs without Data](https://github.com/shizuo-kaji/PretrainCNNwithNoData)

### Installation

#### Using `pip` (Recommended)
Install the Python module directly:
```bash
pip install -U cripser
```

If you encounter architecture compatibility issues, try:
```bash
pip uninstall cripser
pip install --no-binary cripser cripser
```

#### Building from Source
Requires a C++11-compatible compiler (e.g., GCC, Clang, MSVC).

1. Clone the repository:
   ```bash
   git clone https://github.com/shizuo-kaji/CubicalRipser.git
   cd CubicalRipser
   ```

2. Build the command-line executable:
   ```bash
   cd build
   cmake ..
   make
   ```
   The executable `cubicalripser` will be created.

3. Alternatively, without `cmake`:
   ```bash
   cd src
   make all
   ```
   Modify the `Makefile` if needed.

4. Install the Python module:
   ```bash
   pip install .
   ```

## Usage
### Python Module

CubicalRipser works on 1D / 2D / 3D / 4D NumPy arrays (dtype convertible to float64).

Basic example (V-construction, default):
```python
import numpy as np, cripser

arr = np.load("input.npy")
ph = cripser.compute_ph(arr, maxdim=3, filtration="V")   # alias: computePH(...)
```

Result:
- For 1D–3D input: ph has shape (n, 9)
  Columns: dim, birth, death, x1, y1, z1, x2, y2, z2
- For 4D input: shape (n, 11)
  Columns: dim, birth, death, x1, y1, z1, w1, x2, y2, z2, w2
- death is DBL_MAX for essential features

Creator (x1,...) gives birth; destroyer (x2,...) kills the class (see Creator and Destroyer cells).

T-construction (8-neighborhood in 2D, etc.):
```python
ph_T = cripser.compute_ph(arr, maxdim=3, filtration="T")
```

Convert to GUDHI-style structures (see section below):
```python
dgms = cripser.to_gudhi_diagrams(ph)
pers = cripser.to_gudhi_persistence(ph)
```

#### Differentiable PyTorch Wrapper
  ```python
  import torch
  import cripser

  x = torch.rand(32, 32, requires_grad=True)
  ph = cripser.compute_ph_torch(x, maxdim=1, filtration="V")
  loss = cripser.finite_lifetimes(ph, dim=0).sum()
  loss.backward()
  ```
  The gradient is propagated through birth/death values to the corresponding
  creator/destroyer voxel locations. Pairing changes are discrete, so the
  gradient is piecewise defined.

#### GUDHI Conversion Helpers
For convenience, a small utility package is included to convert
the raw output into GUDHI-compatible formats.

```python
import numpy as np
import cripser

arr = np.load("input.npy")
ph = cripser.compute_ph(arr)

# List of diagrams per dimension, each an array of shape (k, 2)
dgms = cripser.to_gudhi_diagrams(ph)

# Or, GUDHI SimplexTree-like list of (dim, (birth, death))
persistence = cripser.to_gudhi_persistence(ph)

# Example: plot using GUDHI
import gudhi as gd
gd.plot_persistence_diagram(diagrams=dgms)
```

Infinite deaths encoded internally as `DBL_MAX` are automatically converted to `np.inf`.

### Helper Python script (demo/cr.py)

A convenience wrapper around the core library for quick experiments without writing code.

Typical capabilities:
- Accepts a single file (e.g. .npy, image, DICOM) or a directory of sequential image / DICOM slices
- Builds a 1D–4D NumPy array
- Chooses V- or T-construction
- Computes persistent homology up to a chosen max dimension
- Writes raw PH pairs (CSV) or serialized NumPy results
- Optional sorting of input slice filenames (useful for DICOM)

Basic help:
```bash
python demo/cr.py -h
```

Examples

1. Single NumPy array (default: V-construction, maxdim=2):
```bash
python demo/cr.py input.npy -o ph.csv
```

2. Increase max dimension and use T-construction:
```bash
python demo/cr.py volume.npy -o ph.csv --maxdim 3 --filtration T
```

3. Directory of DICOM files (sorted), output CSV:
```bash
python demo/cr.py dicom/ --sort -it dcm -o ph.csv
```

4. Directory of PNG slices -> PH (auto-detect by extension):
```bash
python demo/cr.py slices/ -o ph.csv
```

5. Save raw PH as NumPy (to reuse in Python):
```bash
python demo/cr.py volume.npy -o ph.npy --format npy
```

6. Invert intensities (example flag; use only if present in -h):
```bash
python demo/cr.py volume.npy -o ph.csv --invert
```

Typical options (exact list: see -h):
- --maxdim k        maximum homology dimension
- --filtration V|T  choose construction
- --sort            lexicographically sort input filenames
- -it EXT           explicit input slice extension (e.g. dcm, png)
- -o FILE           output file (.csv or .npy)
- --format csv|npy  override format if extension is ambiguous
- --embedded        Alexander dual interpretation (matches CLI flag)
- --invert          intensity inversion (if implemented)



### Command-Line Usage

Basic example (text / Perseus-style input):
```bash
./cubicalripser --print --maxdim 2 --output out.csv demo/3dimsample.txt
```
Output file (CSV): each row
```
dim, birth, death, x1, y1, z1, x2, y2, z2
```
Meaning:
- dim: homology dimension
- birth, death: filtration values (death = DBL_MAX if essential)
- (x1,y1,z1): creator cell coordinates
- (x2,y2,z2): destroyer cell coordinates (omitted / meaningless if death is infinite)

Numpy array input (1D–4D):
```bash
./cubicalripser --output result.csv input.npy
```

Common options:
- --maxdim k        compute up to dimension k (default: 2)
- --print           also print pairs to stdout
- --embedded        use embedded (Alexander dual) interpretation
- --filtration V|T  choose construction (default: V); T alternative executable: tcubicalripser
- --output FILE     write CSV (omit to print only)

Example (T-construction on a 3D volume):
```bash
./tcubicalripser --maxdim 3 --output volume_ph.csv volume.npy
```

Note: For 4D input, two extra coordinates (w1, w2) are appended:
```
dim, birth, death, x1, y1, z1, w1, x2, y2, z2, w2
```

---

## Input Formats

### Supported Formats (command-line version)
- **NUMPY (.npy)**: Native format for both Python and CLI.
- **Perseus Text (.txt)**: [Specification](http://people.maths.ox.ac.uk/nanda/perseus/).
- **CSV (.csv)**: Simplified input for 2D images.
- **DIPHA (.complex)**: [Specification](https://github.com/DIPHA/dipha#file-formats).

### Image to Array Conversion
A small utility is included that converts images in various formats into NUMPY arrays.
- Convert images to `.npy`:
  ```bash
  python demo/img2npy.py input.jpg output.npy
  ```
  A series of image files such as JPEG and PNG files (as long as the Pillow library can handle them)
can also be made into a volume in a similar way:
  ```bash
    python demo/img2npy.py input*.jpg volume.npy
  ```
    Note that here we rely on the shell's path expansion. If your shell does not support it, you can manually specify file names as in the following:
  ```bash
    python demo/img2npy.py input00.dcm input01.dcm input02.dcm volume.npy
  ```

- Handle DICOM volumes:
Given a series of DICOM files named **input00.dcm**, **input01.dcm**, **input02.dcm**... under the directory **dicom**,
we can convert the DICOM files to a single 3D Numpy array **volume.npy**
that is compatible with Cubical Ripser by
  ```bash
  python demo/img2npy.py dicom/*.dcm output.npy
  ```
  Or, we can compute persistent homology directly by
  ```bash
    python demo/cr.py dicom  --sort -it dcm -o output.csv
  ```
  by reading .dcm files from the directry **dicom** in a sorted order.

- Handle the DIPHA format:
The filename should end with ".complex".
Look at [DIPHA binary format](https://github.com/DIPHA/dipha#file-formats) for specification.

We can convert input and output files between Cubical Ripser and DIPHA.
- to convert an Numpy array **img.npy** into DIPHA's format **img.complex**
  ```bash
    python demo/img2npy.py img.npy img.complex
  ```
- the other way around
  ```bash
    python demo/img2npy.py img.complex img.npy
  ```
- convert DIPHA's output **result.output** into an Numpy array **result.npy**
  ```bash
    python demo/img2npy.py result.output result.npy
  ```


### 1D time series
A scalar time-series can be considered as a 1D image,
so Cubical Ripser can compute its persistent homology.
Note that other software would be more efficient for this purpose.

An example of regressing the frequency of noisy sine curves
is demonstrated [here](https://github.com/shizuo-kaji/TutorialTopologicalDataAnalysis).

---

## V and T Constructions

- **V-Construction**: Pixels represent 0-cells (4-neighbor connectivity in 2D).
- **T-Construction**: Pixels represent top-cells (8-neighbor connectivity in 2D).

Use the appropriate executable for your needs:
- **V-construction**: `cubicalripser` (Python module: `cripser`).
- **T-construction**: `tcubicalripser` (Python module: `tcripser`).

By the Alexander duality, the following two give essentially the same results:

    ./cubicalripser input.npy
    ./tcubicalripser --embedded input.npy

The difference is in the sign of the filtration and the permanent cycle.
Here, (--embedded) converts the input I to -I^\infty described in the paper below.

For more details, see
*[Duality in Persistent Homology of Images](https://arxiv.org/abs/2005.04597)* by Adélie Garin et al.

---

## Creator and Destroyer cells
The creator of a cycle is the cell which gives birth to the cycle.
For example, the voxel in a connected component with the lowest filtration value creates a 0-dimensional cycle,
and the voxel which connects two separate connected components destroys the component with a higher birth time.
The creator and the destroyer cells are not uniquely determined, but they provide useful information to localise the cycle.
Cubical Ripser adopts the following convention on the location of these cells:
when the lifetime of a cycle is finte,

    arr[x2,y2,z2] - arr[x1,y1,z1] = death - birth = lifetime

where arr is the image, (x1,y1,z1) is the location of the creator cell, and (x2,y2,z2) is the location of the destroyer cell.
Note that when computed with the (--embedded) option, the roles of creator and destroyer are switched:

    arr[x1,y1,z1] - arr[x2,y2,z2] = death - birth = lifetime


The authors thank Nicholas Byrne for suggesting the convention and providing a test code.


---

## Deep Learning Integration

- **Lifetime Enhanced Image**: Adds topological features as additional channels for CNNs.
  ```bash
  ./cubicalripser --output result.npy input.npy
  python demo/stackPH.py result.npy -o lifetime_image.npy -i input.npy
  ```
  In **lifetime_image.npy**, persistent homology is encoded as the extra channels so that it can be used as input for CNNs.

  Please look at the example section of [our paper](https://arxiv.org/abs/2005.12692).

- **Persistent Histogram Image**:
  Similarly, the *persistent histogram image* can be obtained by
  ```bash
  python demo/stackPH.py result.npy -o hist_image.npy -t hist -i input.npy
  ```

  For practical examples, see [HomologyCNN](https://github.com/shizuo-kaji/HomologyCNN).

---

## Timing Comparisons

The following are some timing comparisons.

```bash
python demo/timing_cr.py sample/bonsai128.npy --runs 5 --warmup 1 -o timing_bonsai128.csv
```

This writes `timing_bonsai128.csv` with:

- `run` rows: one row per timed iteration (`elapsed_seconds`)
- one `summary` row per `(software, filtration)` pair:
  `mean_seconds`, `std_seconds`, `min_seconds`, `max_seconds`
- metadata for comparisons across code changes:
  `timestamp_utc`, `git_commit`, `software`, `filtration`, `maxdim`, `input_shape`
- default combinations are:
  `cubicalripser + V`, `cubicalripser + T`, `gudhi + V`, `gudhi + T`
- you can restrict combinations with e.g.:
  `--software cubicalripser` or `-f V`




## Other software for persistent homology of cubical complexes
We give a referece to various software for persistent homology of images.
The comments are based on our limited understanding and tests, and hence, could be wrong.

- [Cubicle](https://bitbucket.org/hubwag/cubicle/src/master/) by Hubert Wagner

It computes for the V-construction of the image.
Its parallelised algorithm offers faster computation on multi-core machines.
Also, it reads the input image in small chunks so that it requires much less memory footprint.

- [HomcCube](https://i-obayashi.info/software.html) By Ippei Obayashi.

It computes for the V-construction of the image.
It is integrated into Homcloud developed by the same author.

- [DIPHA](https://github.com/DIPHA/dipha) by Ulrich Bauer and Michael Kerber

It computes for the V-construction of the image.
It is parallelised with MPI so it works on a cluster.
The software has been used in various projects.
The memory footprint is relatively large.

- [GUDHI](http://gudhi.gforge.inria.fr/) developed at INRIA

It computes for the V- and T-construction of an array of any dimension.
It is well-documented and offers a well-organised and easy to use interface.
It focuses more on usability than performance.

- [diamorse](https://github.com/AppliedMathematicsANU/diamorse) developed at The Australian National University.

It computes for the V-construction of the image.

- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/) by Vidit Nanda

It computes for the V-construction of the image.

## Release Notes
- (v0.0.24) Repository renamed: This project moved from CubicalRipser_3dim to CubicalRipser.
Old GitHub links should redirect automatically. If you have a local clone, update your remote:
git remote set-url origin https://github.com/shizuo-kaji/CubicalRipser.git
- (v0.0.23) added support for torch integration
- (v0.0.22) changed birth coordinates for T-construction to much with GUDHI for permanent cycles
- (v0.0.19) added support for 4D cubical complexes
- (v0.0.8) fixed memory leak in Python bindings (pointed out by Nicholas Byrne)
- (v0.0.7) slight speed up
- (v0.0.6) changes in the [definition of birth/death location](#Creator-and-Destroyer-cells) (suggested by Nicholas Byrne)
- (up to v0.0.5, difference from the [original version](https://github.com/CubicalRipser/CubicalRipser_3dim)
    - optimised codes (much less memory footprint, much faster for certain data; sometimes more than 100 times.)
    - Python friendly: see the Jupyter Notebook example found under the demo directory.
    - virtually infinite input size (compared to 510x510x510)
    - cache control
    - option to use the Alexander duality for the highest degree persistent homology
    - V and T construction for building cubical complexes from an image
    - output birth/death location
