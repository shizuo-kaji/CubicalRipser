# CubicalRipser

Fast persistent homology for 1D time series, 2D images, and 3D/4D volumes.

Authors: Takeki Sudo, Kazushi Ahara (Meiji University), Shizuo Kaji (Kyoto University)

CubicalRipser adapts [Ripser](http://ripser.org) by Ulrich Bauer to cubical
complexes. It provides C++ command-line programs and a Python package, with
V- and T-constructions, coefficients in F₂, creator/destroyer locations,
optional representative cycles, zigzag persistence of time-varying binary
masks, and PyTorch integration.

**[Read the manual](docs/README.md)** ·
[Try in Google Colab](https://colab.research.google.com/github/shizuo-kaji/CubicalRipser/blob/main/demo/cubicalripser.ipynb) ·
[Release notes](docs/release-notes.md)

## Quickstart

Python 3.9 or later:

```bash
python -m pip install -U numpy cripser
```

```python
import numpy as np
import cripser

# A low-valued ring surrounding a high-valued center.
image = np.ones((7, 7), dtype=np.float64)
image[1:6, 1:6] = 0.0
image[2:5, 2:5] = 1.0

ph = cripser.compute_ph(image, filtration="V", maxdim=1)
print(ph[:, :3])  # homology dimension, birth, death
```

Use `filtration="T"` for T-construction. Results also include creator/destroyer
coordinates; essential classes have `death=np.inf`. See the
[Python API](docs/python-api.md) and [output semantics](docs/concepts.md).

For the CLI, [build the binaries](docs/installation.md#build-the-command-line-programs), then run from the repository root.
The input below is a hollow sphere from [`cripser.datasets`](docs/python-api.md#synthetic-arrays):

```bash
python -c "import numpy as np, cripser; np.save('sphere.npy', cripser.datasets.sphere((32, 32, 32)))"
./build/cubicalripser --maxdim 2 --output out.csv sphere.npy
./build/tcubicalripser --maxdim 2 --output out_t.csv sphere.npy
```

## Documentation

| Task | Guide |
| --- | --- |
| Install Python or build from source | [Installation](docs/installation.md) |
| Compute PH in Python or from the terminal | [Python API](docs/python-api.md) · [CLI](docs/cli.md) |
| Load images, volumes, and slice stacks | [Input and output](docs/io.md) |
| Understand V/T constructions and coordinates | [Constructions and output semantics](docs/concepts.md) |
| Extract and visualize cycles | [Representative cycles](docs/cycles.md) |
| Track features through videos and time-lapse masks | [Zigzag persistence](docs/zigzag.md) |
| Plot diagrams or create feature arrays | [Plotting and vectorization](docs/analysis.md) |
| Use topology in differentiable workflows | [PyTorch and distances](docs/torch.md) |
| Test changes and check performance | [Development and validation](docs/development.md) |
| Compare other implementations | [Related software](docs/related-software.md) |

More tutorials and application examples are listed in the
[manual](docs/README.md#tutorials-and-applications).

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

## License

Distributed under the GNU Lesser General Public License v3.0 or later.
See [LICENSE](LICENSE).
