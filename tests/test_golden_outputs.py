"""Fixed outputs of compute_ph on small deterministic inputs.

A changed digest means changed persistence pairs, birth/death values,
coordinates or cycles.  Update a digest only for an intended change, and say
so in the commit message.
"""

import hashlib

import numpy as np
import pytest

import cripser


def _inputs():
    rng = np.random.default_rng(20261004)
    return {
        "1d": rng.random(64),
        "2d": rng.random((14, 11)),
        "2d_ties": rng.integers(0, 4, (12, 10)).astype(np.float64),
        "3d": rng.random((7, 6, 5)),
        "3d_ties": rng.integers(0, 4, (7, 6, 5)).astype(np.float64),
        "3d_fortran": np.asfortranarray(rng.random((6, 5, 7))),
        "4d": rng.random((4, 4, 3, 3)),
        "4d_ties": rng.integers(0, 3, (4, 3, 4, 3)).astype(np.float64),
    }


INPUTS = _inputs()


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _run(name, filtration, option):
    arr = INPUTS[name]
    if option == "plain":
        return _digest(cripser.compute_ph(arr, filtration=filtration))
    if option == "embedded":
        return _digest(cripser.compute_ph(arr, filtration=filtration, embedded=True))
    if option == "top_dim":
        return _digest(cripser.compute_ph(arr, filtration=filtration, top_dim=True))
    pairs, cycles = cripser.compute_ph(arr, filtration=filtration, representatives=True)
    return _digest(pairs, *[np.asarray(c, dtype=np.float64) for c in cycles])


GOLDEN = {
    "1d-T-embedded": "8f7858db3590804a",
    "1d-T-plain": "8bb2a7f5cb5d34b0",
    "1d-V-embedded": "8f7858db3590804a",
    "1d-V-plain": "8bb2a7f5cb5d34b0",
    "2d-T-embedded": "011d0c04835f3e06",
    "2d-T-plain": "19bc567cb1cb06d7",
    "2d-T-reps": "e1ce51882bccef56",
    "2d-T-top_dim": "3a42a45e25470f09",
    "2d-V-embedded": "fa742ff48e0baf54",
    "2d-V-plain": "7aaa90a0f1e602c4",
    "2d-V-reps": "21f8fdc9d21f2b90",
    "2d-V-top_dim": "2db71542a7681303",
    "2d_ties-T-embedded": "44091b1d669558f3",
    "2d_ties-T-plain": "b22fe17e600c8f64",
    "2d_ties-T-reps": "ea4657768ea2eb2d",
    "2d_ties-T-top_dim": "7e0395857daf70bd",
    "2d_ties-V-embedded": "550835069d75c9dd",
    "2d_ties-V-plain": "80902fdccf37aeb2",
    "2d_ties-V-reps": "e352b4f13b8356c7",
    "2d_ties-V-top_dim": "587142e6f3a92e4d",
    "3d-T-embedded": "7c2474255d2953db",
    "3d-T-plain": "456aa8f70d052482",
    "3d-T-reps": "d0b172752aead72a",
    "3d-T-top_dim": "4bc2f024866d103e",
    "3d-V-embedded": "2223f4b4dba46870",
    "3d-V-plain": "854e0875db08cb7b",
    "3d-V-reps": "77be235be36babd6",
    "3d-V-top_dim": "5221699f2c600c72",
    "3d_fortran-T-embedded": "05d952a082151c82",
    "3d_fortran-T-plain": "04f29bcedd6e7fe0",
    "3d_fortran-T-reps": "cbe4c70cc2af7b02",
    "3d_fortran-T-top_dim": "170cd01cfa021c9d",
    "3d_fortran-V-embedded": "8d4946b40ca84a89",
    "3d_fortran-V-plain": "9b7b9230bfc42bbe",
    "3d_fortran-V-reps": "b85ae5c72bd9f253",
    "3d_fortran-V-top_dim": "cf7606892d515527",
    "3d_ties-T-embedded": "74f2be82feb9a8c4",
    "3d_ties-T-plain": "096e2f2e3f045695",
    "3d_ties-T-reps": "ef151b78398d653c",
    "3d_ties-T-top_dim": "733975257e97803a",
    "3d_ties-V-embedded": "eb5b9217f67c87f2",
    "3d_ties-V-plain": "1b1a945b73581f34",
    "3d_ties-V-reps": "2f31c27e188424b5",
    "3d_ties-V-top_dim": "a4ec9dcd84a7525e",
    "4d-T-embedded": "b37ad44c9bf1966b",
    "4d-T-plain": "e99239cc009f5863",
    "4d-T-top_dim": "bd408deb22cdac2c",
    "4d-V-embedded": "e00b85c7692cbfd4",
    "4d-V-plain": "2033d5fdc4681f6e",
    "4d-V-top_dim": "2c46dcdbb98556da",
    "4d_ties-T-embedded": "7e18ea5e5ae3352e",
    "4d_ties-T-plain": "6b3f01b22520ef5c",
    "4d_ties-T-top_dim": "2c46dcdbb98556da",
    "4d_ties-V-embedded": "b0a17150a0ca8f00",
    "4d_ties-V-plain": "c084375dda95c1fc",
    "4d_ties-V-top_dim": "2c46dcdbb98556da",
}

CASES = sorted(GOLDEN)


@pytest.mark.parametrize("case", CASES)
def test_golden_output(case):
    name, filtration, option = case.split("-")
    assert _run(name, filtration, option) == GOLDEN[case]
