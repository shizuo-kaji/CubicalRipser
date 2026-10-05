# Related software for cubical persistent homology

[Manual](README.md)

The following notes are based on our limited understanding and tests and may be incomplete.

- [Cubicle](https://bitbucket.org/hubwag/cubicle/src/master/) by Hubert Wagner
  - T-construction
  - slices the volume, simplifies each slice in parallel with discrete Morse
    theory, then reduces a single global boundary matrix
  - streams slices through external memory, so volumes larger than RAM can be
    processed
  - the choice for very large volumes, and whenever memory is the binding constraint
  - input is a raw binary file, 8-bit by default (other element types need a recompile)

- [HomcCube](https://i-obayashi.info/software.html) by Ippei Obayashi
  - V-construction
  - integrated into HomCloud, which provides a full TDA workflow around it

- [DIPHA](https://github.com/DIPHA/dipha) by Ulrich Bauer and Michael Kerber
  - V-construction
  - MPI-parallelized; the choice when a compute cluster is available

- [GUDHI](https://gudhi.inria.fr/) (INRIA)
  - V- and T-construction in arbitrary dimensions
  - extensive documentation, and a broad TDA library beyond cubical complexes
  - the choice for dimensions above 4, or when the surrounding toolkit is useful

- [diamorse](https://github.com/AppliedMathematicsANU/diamorse)
  - V-construction

- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/) by Vidit Nanda
  - V-construction

When comparing outputs, align V/T construction, filtration direction, homology
dimensions, and essential-class conventions. Measure on the data and hardware
you intend to use; these notes are not a universal performance ranking.
