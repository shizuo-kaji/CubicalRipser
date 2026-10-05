# Installation

[Manual](README.md) · Next: [Python API](python-api.md)

## Python package

Use Python 3.9 or later:

```bash
python -m pip install -U numpy cripser
```

Check the installation:

```bash
python -c "import cripser; print(cripser.__version__)"
```

If a compatible wheel is unavailable, build from source using the requirements
below:

```bash
python -m pip install --no-binary cripser cripser
```

Python 3.8 is no longer supported. Wheels built for CPython 3.12 use the stable
ABI and can serve newer compatible CPython versions; Python 3.9–3.11 use
version-specific wheels.

## Optional dependencies

Install the dependencies for the features you use:

| Feature | Packages to install with `python -m pip install ...` |
| --- | --- |
| Plotting diagrams and cycles | `matplotlib` |
| Ordinary image files | `Pillow` |
| DICOM / NRRD | `pydicom` / `pynrrd` |
| Distance transforms | `scipy` |
| Automatic thresholding and rescaling | `scikit-image` |
| Geodesic transforms | `scikit-fmm` |
| Differentiable PH | `torch` |
| Wasserstein distance | `torch POT` (POT is imported as `ot`) |
| GUDHI plotting and comparisons | `gudhi` |
| Test runner | `pytest` |

For the legacy `demo/cr.py` script, install `matplotlib Pillow scipy scikit-image`
even for NumPy input: these are imported at startup. DICOM and other specialized
formats need their additional packages. The package-level I/O helpers load
optional dependencies when needed.

## Build from source

Requirements are a C++17 compiler (GCC, Clang, or MSVC), Python development
headers, CMake, and nanobind. The Python build declares CMake ≥ 3.21,
Ninja ≥ 1.11, and nanobind ≥ 2.0 as build dependencies.

```bash
git clone https://github.com/shizuo-kaji/CubicalRipser.git
cd CubicalRipser
python -m pip install .
```

`pip` installs build dependencies in an isolated build environment. For editable
development without build isolation, install them into the active environment:

```bash
python -m pip install 'setuptools>=69' wheel 'cmake>=3.21' 'ninja>=1.11' 'nanobind>=2.0' numpy
python -m pip install -e . --no-build-isolation
```

## Build the command-line programs

Run from the repository root. The CMake project also configures the Python
bindings, so nanobind and Python development headers are needed for this build.

```bash
python -m pip install 'nanobind>=2.0'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

The executables are `build/cubicalripser` (V-construction) and
`build/tcubicalripser` (T-construction). Multi-configuration generators such as
Visual Studio normally put the `.exe` files in `build/Release/`.
If CMake selects a different Python environment, pass
`-DPython_EXECUTABLE=/path/to/python` when configuring.

Always use a Release build for performance measurements.
A legacy alternative is `make all` from `src/`; CMake is the canonical build.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `import cripser` fails after source edits | Rebuild the extension with the editable-install command above. |
| CMake cannot locate nanobind | Install nanobind into the interpreter selected by CMake. |
| A helper reports a missing module | Install the corresponding optional dependency from the table. |
| A repository example cannot find `sample/...` | Clone the repository and run from its root, or supply your own array. |

Continue with the [Python quickstart](python-api.md#first-computation) or
[CLI examples](cli.md#basic-usage).
