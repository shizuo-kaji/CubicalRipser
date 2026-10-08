# Command-line usage

[Manual](README.md) · Previous: [Python API](python-api.md) · Next: [Input and output](io.md)

## Basic usage

[Build the executables](installation.md#build-the-command-line-programs) and
create example inputs with [`cripser.datasets`](python-api.md#synthetic-arrays):

```python
import numpy as np
from cripser import datasets

np.save("sphere.npy", datasets.sphere((32, 32, 32)))        # one long H₂ bar
np.save("sphere4d.npy", datasets.sphere((12, 12, 12, 12)))  # one long H₃ bar
np.save("field.npy", datasets.gaussian_random_field((128, 128), seed=0))
```

Then run these commands from the repository root:

```bash
./build/cubicalripser --maxdim 2 --output out.csv sphere.npy
./build/tcubicalripser --maxdim 2 --output out_t.csv sphere.npy
./build/cubicalripser --maxdim 3 --output ph4d.npy sphere4d.npy
```

The binary selects the construction: there is no `--filtration` option in the
C++ CLI. Input file extensions select the loader; see [formats](io.md#cli-input-formats).

## Options

| Option | Meaning / default |
| --- | --- |
| `--help`, `-h` | Show supported options. |
| `--maxdim k`, `-m k` | Compute through homology dimension `k`; default 3, bounded by input dimension minus one. |
| `--threshold t`, `-t t` | Include cells with birth below the threshold; default `DBL_MAX`. Classes still alive at `t` are reported with death `t` and no destroyer. |
| `--print`, `-p` | Print persistence pairs to stdout. |
| `--output FILE`, `-o FILE` | Output filename; default `output.csv`. Use `none` to suppress file creation. |
| `--location yes\|none`, `-l yes\|none` | Include or omit coordinates in CSV output; default `yes`. |
| `--embedded`, `-e` | Use the [Alexander-dual embedded convention](concepts.md#alexander-duality-and-embedding). |
| `--top_dim` | Use the [top-dimensional shortcut](concepts.md#top-dimensional-shortcut). |
| `--threads n` | Workers for grid scans and sorts: 1 (default), 0 (auto), or a positive count. |
| `--verbose`, `-v` | Additional computation details. |

Advanced reduction settings:

| Option | Meaning / default |
| --- | --- |
| `--algorithm link_find`, `-a ...` | H₀ method; `link_find` (union-find) is the only one. |
| `--cache_size n`, `-c n` | Maximum cached reduced columns; default `2^31`. |
| `--min_recursion_to_cache n`, `-mc n` | Minimum recursion count for caching; default 0. |
| `--explicit-clearing` / `--no-explicit-clearing` | Enable / disable pivot-table compression before clearing; enabled by default. |
| `--vector-working-column` | Experimental sorted-vector working columns for 4D H₁; disabled by default. |

Measure workload-specific time and memory before changing advanced settings.
`--threads 0` respects `CRIPSER_NUM_THREADS`, as in the [Python API](python-api.md#parallelism).

## Output files

| Filename | Contents |
| --- | --- |
| `*.csv` | Headerless rows of dimension, birth, death, and coordinates (unless `--location none`). |
| `*.npy` | A full 9- or 11-column table, including coordinates even with `--location none`. |
| Other extension, e.g. `*.diagram` | DIPHA-style persistence binary with dimension and endpoints; no coordinates. |
| `none` | No output file; console messages may still be printed. |

Raw CLI output uses `DBL_MAX` for essential deaths, or the threshold `t` when
one is given. See [output semantics](concepts.md) before filtering or plotting
the result.

```bash
./build/cubicalripser --print --location none --output pairs.csv field.npy
./build/cubicalripser --output none sphere.npy
```

## Python helper script

[`demo/cr.py`](../demo/cr.py) combines loading, preprocessing, and computation.
It accepts a file or a directory of slices and selects V/T using `--filtration`.
It requires the [demo dependencies](installation.md#optional-dependencies).

```bash
python demo/cr.py -h
python demo/cr.py field.npy -o ph.csv
python demo/cr.py sphere.npy --maxdim 2 --filtration T -o ph.npy
python demo/cr.py dicom/ --sort -it dcm -o ph.csv
python demo/cr.py slices/ -it png -o ph.csv
python demo/cr.py sphere.npy --negative -o ph.csv
```

Selected options are `--maxdim`, `--filtration V|T`, `--embedded`, `--top_dim`,
`--sort`, `-it EXT`, `-o FILE`, `--negative`, `--transform`, `--threshold`, and
`--threshold_upper_limit`. The helper's thresholds control preprocessing;
they are distinct from the C++ CLI's filtration cutoff. Consult `-h` for the
full script interface.

For conversion without computing PH, use [demo/img2npy.py](io.md#conversion-script).
