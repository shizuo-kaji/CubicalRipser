# PyTorch and diagram distances

[Manual](README.md) · Previous: [Plotting and vectorization](analysis.md) · Next: [Development](development.md)

## Differentiable persistent homology

Install `torch`, then compute on a floating-point tensor:

```python
import torch
import cripser

x = torch.rand(32, 32, requires_grad=True)
ph = cripser.compute_ph_torch(x, maxdim=1, filtration="V")
loss = cripser.finite_lifetimes(ph, dim=0).sum()
loss.backward()
print(x.grad.shape)  # torch.Size([32, 32])
```

The function accepts `filtration="V"`, `maxdim=3`, `top_dim=False`,
`embedded=False`, and `location="yes"` as keyword options. The result is a PH
table with the [same column layout](python-api.md#output-table) as the NumPy
API. `finite_lifetimes(ph, dim=None)` selects finite-death intervals, optionally
restricted to one homology dimension, and returns `death - birth`.

## How gradients work

The forward pass transfers input to CPU `float64` and calls the NumPy backend.
The output table is returned to the input device. This is CPU persistence
computation even if the input tensor is on a GPU.

The backward pass routes birth/death gradients to creator/destroyer voxel
locations, treating the pairing as fixed. Coordinates and dimension columns
are not differentiable. Pairings change discretely, so gradients are
piecewise-defined rather than smooth everywhere. Nonfinite endpoints and
out-of-bounds locations do not receive gradient contributions.

For training examples, start with the default `embedded=False, top_dim=False`
convention. The wrapper's backward pass adds endpoint gradients directly at
the returned locations; it has no separate sign correction for embedded or
duality-based conventions. Do not assume its gradients in those modes follow
the ordinary sublevel-set interpretation.

Every tensor axis is treated as a spatial direction. Process batch or channel
axes separately when they represent independent samples. The wrapper does
not expose the NumPy API's `representatives`, `n_threads`, or `inf_cutoff`
arguments.

## Differentiable persistence images

Using `ph` from the first example, a smooth feature tensor can be constructed
before calling `backward` on a chosen loss:

```python
features = cripser.persistence_image(
    ph,
    homology_dims=(0, 1),
    birth_range=(0.0, 1.0),
    life_range=(0.0, 1.0),
    n_birth_bins=16,
    n_life_bins=16,
)
```

Keep feature ranges fixed across samples. See [persistence images](analysis.md#persistence-images)
for the remaining parameters. When experimenting with multiple backward
passes, recompute `ph` for each forward/backward cycle as usual in PyTorch.

## Wasserstein distance

Install `POT` in addition to PyTorch. `wasserstein_distance` accepts either
full PH tables or `(n, 2)` birth/death tensors:

```python
import torch
import cripser

a = torch.tensor([[0.0, 1.0], [0.2, 0.7]], requires_grad=True)
b = torch.tensor([[0.1, 0.9]])
distance = cripser.wasserstein_distance(a, b, p=2.0, q=2.0)
distance.backward()
```

`p` controls the Wasserstein order and `q` the ground Minkowski metric.
Matching to the diagonal is included, and nonfinite pairs are dropped.
With full PH tables, use `dim=k` to compare a specific homology dimension;
otherwise dimensions are pooled. Inputs must be on the same device.
`return_pth_power=True` returns the transport cost before taking the p-th root.

Applications: [Homology-enhanced CNNs](https://github.com/shizuo-kaji/HomologyCNN)
and [Pretraining CNNs without Data](https://github.com/shizuo-kaji/PretrainCNNwithNoData).
