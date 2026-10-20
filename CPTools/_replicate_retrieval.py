"""Replicate retrieval: copairs-style mean average precision with label-shuffle nulls.

Kalinin et al. (2024) score a perturbation by how well its wells retrieve each
other. A query well ranks candidate wells by cosine similarity; average
precision (AP) is the mean, over the query's positives, of the precision at
each positive's rank, and mAP averages AP over queries.

Profiles are centered per batch on the mean of the wells being compared,
which does not depend on the treatment/control labels. Centering on the
controls alone would give a treatment's wells a shared offset (the noise of
the control mean), so inert replicates would retrieve each other; the effect
grows with the number of features. A per-batch location shift already in the
layer (e.g. DMSO centering) cancels.
"""
from __future__ import annotations

from itertools import combinations, product
from math import comb, prod
from typing import Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from ._reference_normalization import _batches, _source
from ._utils import bh_fdr

# Elements per (labelings x wells x wells) block; bounds memory of the vectorized nulls.
_BLOCK = 2_000_000
_TOL = 1e-12


def _setup(adata, treatment_key, control_value, batch_key, treatments, layer, features, min_wells):
    if treatment_key not in adata.obs:
        raise KeyError(f"Missing {treatment_key!r} in obs.")
    batch = _batches(adata, batch_key)
    if adata.obs[treatment_key].isna().any():
        raise ValueError(f"Treatment labels in {treatment_key!r} must not be missing.")
    label = adata.obs[treatment_key].astype(str).to_numpy()
    control = label == str(control_value)
    controls = {b: np.flatnonzero(control & (batch == b)) for b in pd.unique(batch[control])}
    controls = {b: idx for b, idx in controls.items() if len(idx) >= 2}
    if not controls:
        raise ValueError(f"No batch has at least 2 wells with {treatment_key} == {control_value!r}.")
    if treatments is None:
        names, requested = [t for t in pd.unique(label) if t != str(control_value)], False
    else:
        names = list(dict.fromkeys([treatments] if isinstance(treatments, str) else map(str, treatments)))
        present = set(label)
        missing = [t for t in names if t not in present]
        if not names or missing or str(control_value) in names:
            raise ValueError(f"treatments must be nonempty, exclude the control and exist; missing: {missing}")
        requested = True

    wells, skipped, partial = {}, [], []
    for name in names:
        idx = np.flatnonzero(label == name)
        keep = idx[np.isin(batch[idx], list(controls))]
        partial += [name] if 0 < len(keep) < len(idx) else []
        if len(keep) < min_wells:
            skipped.append(name)
        else:
            wells[name] = keep
    if skipped:
        message = (f"Treatments with fewer than {min_wells} wells in batches with >= 2 control wells: "
                   f"{skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
        if requested:
            raise ValueError(message)
        warnings.warn(f"Skipped {len(skipped)} {message}", stacklevel=3)
    if partial:
        warnings.warn(f"Ignored wells of {len(partial)} treatments on batches with < 2 control wells, "
                      f"e.g. {partial[:5]}.", stacklevel=3)
    if not wells:
        raise ValueError("No treatment can be scored.")

    data = adata if features is None else adata[:, list(features)]
    if data.n_vars == 0:
        raise ValueError("Select at least one feature.")
    X = _source(data, layer)
    used_batches = {b for idx in wells.values() for b in batch[idx]}
    used = np.concatenate(list(wells.values()) + [controls[b] for b in used_batches])
    if not np.isfinite(X[used]).all():
        raise ValueError("Selected profiles contain nonfinite values; select a common finite feature set first.")
    # Label-free plate centers: mean of all analyzed wells on the plate.
    means = {b: X[used[batch[used] == b]].mean(axis=0) for b in used_batches}
    return X, batch, controls, means, wells


def _labelings(sizes, n_perm, rng):
    """Within-block labelings of k of p pool wells: all of them if <= n_perm, else n_perm random."""
    offsets = np.cumsum([0] + [p for p, _ in sizes])[:-1]
    width = sum(p for p, _ in sizes)
    total = prod(comb(p, k) for p, k in sizes)
    if total <= n_perm:
        labels = np.zeros((total, width), dtype=bool)
        blocks = [list(combinations(range(p), k)) for p, k in sizes]
        for row, choice in enumerate(product(*blocks)):
            for offset, combo in zip(offsets, choice):
                labels[row, offset + np.asarray(combo)] = True
        return labels, True
    labels = np.zeros((n_perm, width), dtype=bool)
    rows = np.arange(n_perm)[:, None]
    for offset, (p, k) in zip(offsets, sizes):
        labels[rows, offset + np.argsort(rng.random((n_perm, p)), axis=1)[:, :k]] = True
    return labels, False


def _activity_map(cosine, block, labels):
    """mAP for each labeling (row of ``labels``) of the pool wells.

    Labeled wells are queries and positives; unlabeled wells are controls and
    are negatives only for queries in their own block.
    """
    n, P = labels.shape
    k = int(labels[0].sum())
    order = np.argsort(-cosine, axis=1, kind="stable")    # each query's ranking
    not_self = np.take_along_axis(~np.eye(P, dtype=bool), order, axis=1)
    same = np.take_along_axis(block[:, None] == block[None, :], order, axis=1) & not_self
    out = np.empty(n)
    step = max(1, _BLOCK // (P * P))
    for start in range(0, n, step):
        L = labels[start:start + step]
        hit = L[:, order] & not_self                     # (labeling, query, rank)
        seen = np.cumsum(hit | same, axis=2)
        precision = np.where(hit, np.cumsum(hit, axis=2) / np.maximum(seen, 1), 0.0)
        ap = precision.sum(axis=2) / (k - 1)
        out[start:start + len(L)] = (ap * L).sum(axis=1) / k
    return out


def _activity(X, batch, controls, wells, n_perm, rng):
    rows, cache = [], {}
    for name, idx in wells.items():
        blocks = list(dict.fromkeys(batch[idx]))
        pool, block, sizes = [], [], []
        for j, b in enumerate(blocks):
            treated = idx[batch[idx] == b]
            pool += [treated, controls[b]]
            block.append(np.full(len(treated) + len(controls[b]), j))
            sizes.append((len(treated) + len(controls[b]), len(treated)))
        pool, block = np.concatenate(pool), np.concatenate(block)
        # Center each block on its pool mean (treated + control wells), which
        # every labeling of the block shares.
        profiles = X[pool].copy()
        for j in range(len(blocks)):
            profiles[block == j] -= profiles[block == j].mean(axis=0)
        norm = np.linalg.norm(profiles, axis=1, keepdims=True)
        unit = np.divide(profiles, norm, out=np.zeros_like(profiles), where=norm > 0)
        cosine = unit @ unit.T
        observed_labels = np.concatenate([np.arange(p) < k for p, k in sizes])
        if tuple(sizes) not in cache:
            cache[tuple(sizes)] = _labelings(sizes, n_perm, rng)
        labels, exact = cache[tuple(sizes)]
        observed = float(_activity_map(cosine, block, observed_labels[None])[0])
        null = _activity_map(cosine, block, labels)
        exceed = int(np.sum(null >= observed - _TOL))
        rows.append({
            "treatment": name, "n_wells": len(idx), "n_batches": len(blocks),
            "n_controls": int(sum(len(controls[b]) for b in blocks)),
            "mAP": observed, "null_mean": float(null.mean()),
            # The observed labeling is one of the enumerated ones when exact.
            "p_value": exceed / len(null) if exact else (1 + exceed) / (1 + len(null)),
            "p_min": 1 / len(null) if exact else 1 / (1 + len(null)),
            "n_perm": len(null), "exact": exact,
        })
    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("p_value") + 1, "q_value", bh_fdr(out["p_value"].to_numpy()))
    return out


def _group_map(ranks, groups):
    """mAP of each group (row of ``groups``) when its members retrieve each other."""
    n, m = groups.shape
    out = np.empty(n)
    step = max(1, _BLOCK // (m * m))
    for start in range(0, n, step):
        g = groups[start:start + step]
        sub = ranks[g[:, :, None], g[:, None, :]].astype(float)
        sub[:, np.arange(m), np.arange(m)] = np.inf
        sub.sort(axis=2)
        out[start:start + len(g)] = (np.arange(1, m) / sub[:, :, :m - 1]).mean(axis=(1, 2))
    return out


def _consistency(adata, X, batch, means, wells, annotation_key, n_perm, rng):
    if annotation_key not in adata.obs:
        raise KeyError(f"Missing {annotation_key!r} in obs.")
    annotation = adata.obs[annotation_key]
    names, profiles, labels = [], [], []
    for name, idx in wells.items():
        values = pd.unique(annotation.iloc[idx].dropna().astype(str))
        if len(values) > 1:
            raise ValueError(f"Treatment {name!r} has several {annotation_key!r} values: {list(values)}.")
        if len(values) == 1:
            names.append(name)
            labels.append(values[0])
            profiles.append((X[idx] - np.stack([means[b] for b in batch[idx]])).mean(axis=0))
    labels = np.asarray(labels, dtype=object)
    groups = {v: np.flatnonzero(labels == v) for v in pd.unique(labels)}
    scored = {v: g for v, g in groups.items() if len(g) >= 2}
    if len(names) < 3 or not scored:
        raise ValueError(f"Need >= 3 annotated treatments and a {annotation_key!r} value shared by >= 2.")
    profiles = np.stack(profiles)
    norm = np.linalg.norm(profiles, axis=1, keepdims=True)
    unit = np.divide(profiles, norm, out=np.zeros_like(profiles), where=norm > 0)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    N = len(names)
    # ranks[i, j]: rank of treatment j in treatment i's retrieval list (self last).
    ranks = np.empty((N, N), dtype=np.int32)
    ranks[np.arange(N)[:, None], np.argsort(-similarity, axis=1, kind="stable")] = np.arange(1, N + 1)
    # The null of a random group depends only on its size.
    nulls = {
        m: _group_map(ranks, np.argpartition(rng.random((n_perm, N)), m - 1, axis=1)[:, :m])
        for m in sorted({len(g) for g in scored.values()})
    }
    rows = []
    for value, members in scored.items():
        observed = float(_group_map(ranks, members[None])[0])
        null = nulls[len(members)]
        rows.append({
            annotation_key: value, "n_treatments": len(members), "mAP": observed,
            "null_mean": float(null.mean()),
            "p_value": (1 + int(np.sum(null >= observed - _TOL))) / (1 + n_perm),
            "n_perm": n_perm,
        })
    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("p_value") + 1, "q_value", bh_fdr(out["p_value"].to_numpy()))
    return out


def replicate_map(
    adata: ad.AnnData,
    *,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    batch_key: str = "Batch",
    annotation_key: str | None = None,
    treatments: Sequence[str] | None = None,
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    n_perm: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Mean average precision of replicate retrieval (copairs, Kalinin et al. 2024).

    Phenotypic activity (default): every well of a treatment queries the
    treatment's other wells (positives, any batch) against the control wells of
    its own batch (negatives), ranked by cosine similarity. Each batch's wells
    are centered on the mean of that batch's treated and control wells. One row
    per treatment: ``mAP``, ``null_mean`` (expected mAP under
    the null), ``p_value``, BH ``q_value`` over the returned treatments,
    ``p_min`` (smallest attainable p-value), ``n_perm`` and ``exact``.

    The null shuffles treatment/control labels within each batch, keeping the
    per-batch counts. When the number of distinct labelings is <= ``n_perm`` they are all
    enumerated and the p-value is exact; e.g. 3 wells and 8 controls on one
    plate give C(11, 3) = 165 labelings and ``p_min`` = 1/165. Under no effect
    the test is valid if treatment and control wells of a batch are
    exchangeable; controls in fixed (e.g. edge) positions break that.

    Phenotypic consistency (``annotation_key``, e.g. ``"MOA"``): consensus
    profiles (mean well, centered on the mean of all analyzed wells of its
    batch) of treatments sharing an annotation
    retrieve each other against all other annotated treatments. One row per
    annotation value shared by >= 2 treatments; treatments without an
    annotation are left out. The null draws random treatment groups of the same
    size (label shuffle). Pass the phenotypically active treatments with
    ``treatments=``.

    Cosine similarity depends on the feature scaling: use a normalized layer and
    a common finite ``features`` set (e.g. ``highly_variable``). Per-batch
    centering already in the layer cancels; per-batch scales estimated from the
    controls do not. Wells are technical replicates on the same plates.
    """
    if not isinstance(n_perm, (int, np.integer)) or n_perm < 1:
        raise ValueError("n_perm must be an integer >= 1.")
    consistency = annotation_key is not None
    X, batch, controls, means, wells = _setup(
        adata, treatment_key, control_value, batch_key, treatments, layer, features,
        min_wells=1 if consistency else 2,
    )
    rng = np.random.default_rng(random_state)
    if consistency:
        return _consistency(adata, X, batch, means, wells, annotation_key, int(n_perm), rng)
    return _activity(X, batch, controls, wells, int(n_perm), rng)
