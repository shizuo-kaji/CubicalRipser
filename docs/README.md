# CubicalRipser manual

CubicalRipser computes persistent homology of scalar arrays in one to four
spatial dimensions, with coefficients in F₂. This manual describes the Python
package, command-line programs, and tools shipped in this repository.

## Start here

1. [Install CubicalRipser](installation.md).
2. [Compute your first persistence diagram](python-api.md#first-computation).
3. [Interpret birth, death, and coordinates](concepts.md).
4. [Plot or turn the result into features](analysis.md).

The examples use synthetic arrays, built inline or with
[`cripser.datasets`](python-api.md#synthetic-arrays). Commands using `demo/` or
`build/` assume the repository root as the working directory; these
directories are not provided by a normal wheel installation.

## Chapters

| Chapter | What you will find |
| --- | --- |
| [Installation](installation.md) | Python requirements, optional dependencies, source and CLI builds |
| [Python API](python-api.md) | Computation parameters, output tables, array layouts, threading, example data |
| [Command-line usage](cli.md) | V/T binaries, options, output files, `demo/cr.py` |
| [Input and output](io.md) | NumPy, CSV, Perseus, DIPHA, images, slice stacks, preprocessing |
| [Constructions and output semantics](concepts.md) | V/T connectivity, sublevel filtrations, duality, creator/destroyer locations |
| [Representative cycles](cycles.md) | Cycle encoding, selection, plotting, computational cost |
| [Zigzag persistence](zigzag.md) | Time-varying masks, interval ends, relation to ordinary persistence |
| [Plotting and vectorization](analysis.md) | Persistence diagrams, GUDHI conversion, persistence images, spatial histograms |
| [PyTorch and diagram distances](torch.md) | Gradients, losses, Wasserstein distance, limitations |
| [Development and validation](development.md) | Builds, tests, performance checks, repository layout |
| [Related software](related-software.md) | Other cubical persistent homology implementations |
| [Release notes](release-notes.md) | Changes and compatibility history |

## Common tasks

- **Analyze dark or bright structures:** read [filtration direction](concepts.md#filtration-direction).
- **Choose pixel connectivity:** compare [V and T constructions](concepts.md#v-and-t-constructions).
- **Find a feature in the original image:** inspect [coordinates](concepts.md#creator-and-destroyer-cells), or obtain a [representative cycle](cycles.md).
- **Process many images:** use [threaded batch computation](python-api.md#parallelism).
- **Load a medical volume or a stack of slices:** see [image loading](io.md#images-and-slice-stacks).
- **Track features through a video or time-lapse masks:** use [zigzag persistence](zigzag.md).
- **Train with a topology-based loss:** start with [PyTorch](torch.md).

## Tutorials and applications

- [Main notebook](../demo/cubicalripser.ipynb), also available in [Google Colab](https://colab.research.google.com/github/shizuo-kaji/CubicalRipser/blob/main/demo/cubicalripser.ipynb).
- [Hands-on TDA tutorial](https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/master/TopologicalDataAnalysisWithPython.ipynb), including a time-series frequency-regression example.
- Deep learning examples: [Homology-enhanced CNNs](https://github.com/shizuo-kaji/HomologyCNN) and [Pretraining CNNs without Data](https://github.com/shizuo-kaji/PretrainCNNwithNoData).

[Project overview, citation, and license](../README.md)
