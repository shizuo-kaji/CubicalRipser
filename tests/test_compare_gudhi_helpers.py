import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_compare_gudhi_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "demo" / "compare_gudhi.py"
    spec = importlib.util.spec_from_file_location("compare_gudhi", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_dataset_path_prefers_sample_stem():
    compare_gudhi = _load_compare_gudhi_module()
    repo_root = Path(__file__).resolve().parent.parent
    sample_dir = repo_root / "sample"

    resolved = compare_gudhi._resolve_dataset_path("bonsai128", sample_dir)

    assert resolved == (sample_dir / "bonsai128.npy").resolve()


def test_collect_sample_dataset_paths_expands_directory(tmp_path):
    compare_gudhi = _load_compare_gudhi_module()
    repo_root = tmp_path / "repo"
    sample_dir = repo_root / "sample"
    nested_dir = sample_dir / "nested"
    nested_dir.mkdir(parents=True)
    np.save(sample_dir / "alpha.npy", np.arange(4, dtype=np.uint8))
    np.save(nested_dir / "beta.npy", np.arange(6, dtype=np.uint8))

    rows = compare_gudhi._collect_sample_dataset_paths(["sample"], sample_dir, repo_root)

    assert [(name, path.relative_to(repo_root).as_posix()) for name, path in rows] == [
        ("alpha", "sample/alpha.npy"),
        ("sample/nested/beta", "sample/nested/beta.npy"),
    ]
