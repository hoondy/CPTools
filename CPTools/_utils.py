from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


def to_dense_matrix(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Return a dense float32 matrix from dense/sparse inputs."""
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def infer_batch_from_path(path: str | Path) -> str:
    """Infer batch (plate id) from a schema filename."""
    name = Path(path).name
    first = name.split("_", 1)[0]
    if first:
        return first

    match = re.match(r"([A-Za-z0-9]+)", Path(path).stem)
    if not match:
        raise ValueError(f"Unable to infer batch id from path: {path}")
    return match.group(1)


def clean_feature_name(name: str) -> str:
    """Convert Harmony feature labels into compact Cell Painting names."""
    out = name.replace("Nuclei Selected - ", "")
    out = out.replace(" - Mean per Well", "")
    out = out.replace("PhenoVue Hoechst 33342", "Nuclei")
    out = out.replace("PhenoVue Fluor 488", "ER")
    out = out.replace("PhenoVue Fluor 555", "AGP_NR")
    out = out.replace("PhenoVue 641 Mito Stain", "Mito")
    return out


def make_unique(values: Iterable[str]) -> list[str]:
    """Make strings unique by appending .1, .2, ... for duplicates."""
    seen: dict[str, int] = {}
    out: list[str] = []

    for value in values:
        count = seen.get(value, 0)
        if count == 0:
            out.append(value)
        else:
            out.append(f"{value}.{count}")
        seen[value] = count + 1
    return out
