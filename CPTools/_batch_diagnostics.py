"""Batch-effect diagnostics for profiles from multi-plate screens.

When compounds are assigned to plates at random, nothing biological makes two
compounds from one plate more alike than two compounds from different plates.
Any excess similarity within plates (or imaging days, or cell batches) is a
batch component of the profiles, for example the noise of the plate center
they were normalized to. Matched profiles of one compound measured on
different plates (salt forms, repeated library entries) show whether biology
survives the correction.
"""
from __future__ import annotations

from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._reference_normalization import _batches, _query_mask, _source

# Similarity rows computed at once; bounds memory to about _CHUNK x n_profiles.
_CHUNK = 1024


def _unit_profiles(adata, layer, use_rep, features, query):
    data = adata if query is None else adata[_query_mask(adata.obs, query)]
    if use_rep is not None:
        if layer is not None or features is not None:
            raise ValueError("Pass either use_rep or layer/features, not both.")
        if use_rep not in data.obsm:
            raise KeyError(f"Missing {use_rep!r} in obsm.")
        X = np.asarray(data.obsm[use_rep], dtype=float)
    else:
        X = _source(data if features is None else data[:, list(features)], layer)
    if len(X) < 3 or X.shape[1] == 0:
        raise ValueError("Need at least 3 profiles and 1 feature.")
    if not np.isfinite(X).all():
        raise ValueError("Profiles contain nonfinite values; select a common finite feature set first.")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Some profiles are all zero; cosine similarity is undefined.")
    return X / norm, data.obs


def _codes(obs, key):
    if key not in obs:
        raise KeyError(f"Missing {key!r} in obs.")
    if obs[key].isna().any():
        raise ValueError(f"{key!r} must not be missing.")
    return pd.factorize(obs[key].astype(str))[0]


def batch_similarity(
    adata: ad.AnnData,
    *,
    batch_keys: str | Sequence[str] = "Batch",
    layer: str | None = None,
    use_rep: str | None = None,
    features: Sequence[str] | None = None,
    query: str | None = None,
    exclude_key: str | None = None,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """How much more a profile resembles profiles from its own batch.

    Typical input is one consensus profile per compound (e.g. from
    ``sc.get.aggregate``) in a normalized layer, or an embedding via
    ``use_rep``. For each key in ``batch_keys`` (e.g. plate, imaging day, cell
    batch) the result has one row with:

    - ``cosine_same`` / ``cosine_cross``: mean cosine similarity of profile
      pairs from the same / different batches; ``cosine_excess`` is their
      difference. With compounds assigned to batches at random, an excess above
      0 is a component the profiles share with their batch.
    - ``knn_same``: fraction of each profile's ``n_neighbors`` most similar
      profiles from its own batch; ``knn_expected``: the fraction expected if
      neighbors were unrelated to batch (the same-batch share of the other
      profiles); ``knn_enrichment`` = ``knn_same / knn_expected``.

    ``exclude_key`` ignores pairs sharing its value, e.g. ``"Treatment"`` for
    well-level profiles whose replicates share a plate. Profiles are used as
    given, without centering, because the origin of a normalization is part of
    what is evaluated. Compounds that truly share a phenotype and sit on one
    plate add a small legitimate excess.
    """
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be an integer >= 1.")
    keys = [batch_keys] if isinstance(batch_keys, str) else list(batch_keys)
    if not keys:
        raise ValueError("Supply at least one batch key.")
    U, obs = _unit_profiles(adata, layer, use_rep, features, query)
    n = len(U)
    codes = {key: _codes(obs, key) for key in keys}
    exclude = _codes(obs, exclude_key) if exclude_key is not None else np.arange(n)
    if n - 1 - (np.bincount(exclude)[exclude] - 1).max() < n_neighbors:
        raise ValueError("Too few comparable profiles for n_neighbors.")

    sums = {key: np.zeros(2) for key in keys}       # same, cross
    counts = {key: np.zeros(2) for key in keys}
    knn_same = {key: np.zeros(n) for key in keys}
    expected = {key: np.zeros(n) for key in keys}
    for start in range(0, n, _CHUNK):
        rows = np.arange(start, min(start + _CHUNK, n))
        S = U[rows] @ U.T
        valid = exclude[rows, None] != exclude[None, :]
        valid[np.arange(len(rows)), rows] = False
        nearest = np.argpartition(np.where(valid, -S, np.inf), n_neighbors - 1, axis=1)[:, :n_neighbors]
        for key in keys:
            same = codes[key][rows, None] == codes[key][None, :]
            for k, mask in enumerate((same & valid, ~same & valid)):
                sums[key][k] += S[mask].sum()
                counts[key][k] += mask.sum()
            knn_same[key][rows] = np.take_along_axis(same, nearest, axis=1).mean(axis=1)
            expected[key][rows] = (same & valid).sum(axis=1) / valid.sum(axis=1)

    records = []
    for key in keys:
        same_mean, cross_mean = np.divide(sums[key], counts[key], out=np.full(2, np.nan),
                                          where=counts[key] > 0)
        observed, chance = knn_same[key].mean(), expected[key].mean()
        records.append({
            "batch_key": key, "n_profiles": n, "n_batches": int(codes[key].max() + 1),
            "cosine_same": same_mean, "cosine_cross": cross_mean, "cosine_excess": same_mean - cross_mean,
            "knn_same": observed, "knn_expected": chance,
            "knn_enrichment": observed / chance if chance > 0 else np.nan,
        })
    return pd.DataFrame(records).set_index("batch_key")


def group_retrieval(
    adata: ad.AnnData,
    *,
    group_key: str,
    relation_keys: Sequence[str] | None = None,
    layer: str | None = None,
    use_rep: str | None = None,
    features: Sequence[str] | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    """Rank of matched profiles among all profiles, e.g. salt forms of one compound.

    Every profile with a ``group_key`` value queries all other profiles by
    cosine similarity. For each other member of its group (a positive), the
    result reports ``rank``: 1 + the number of non-members more similar to the
    query than the positive (other members never push a positive down), and
    ``rank_fraction`` = ``rank / n_candidates`` with ``n_candidates`` the
    non-members plus that positive. Profiles without a group value are
    candidates only. A positive in the top 1% has ``rank_fraction <= 0.01``.

    ``relation_keys``, finest first (e.g. ``["Batch", "day", "cell_batch"]``),
    label each pair by the first key on which query and positive agree
    (``"same Batch"``, ``"same day"``, ...) or ``"different <last key>"``.
    One row per ordered (query, positive) pair, so each pair appears twice.
    """
    U, obs = _unit_profiles(adata, layer, use_rep, features, query)
    if group_key not in obs:
        raise KeyError(f"Missing {group_key!r} in obs.")
    labels = obs[group_key]
    names = obs.index.to_numpy()
    relation_keys = list(relation_keys or [])
    relation_codes = [_codes(obs, key) for key in relation_keys]
    groups = {g: np.flatnonzero((labels == g).to_numpy()) for g in pd.unique(labels.dropna())}
    groups = {g: idx for g, idx in groups.items() if len(idx) >= 2}
    if not groups:
        raise ValueError(f"No {group_key!r} value has at least 2 profiles.")

    records = []
    for group, members in groups.items():
        S = U[members] @ U.T
        outsider = np.ones(len(U), dtype=bool)
        outsider[members] = False
        for a, i in enumerate(members):
            negatives = S[a, outsider]
            for b, j in enumerate(members):
                if i == j:
                    continue
                relation = (f"different {relation_keys[-1]}" if relation_keys else "")
                for key, code in zip(relation_keys, relation_codes):
                    if code[i] == code[j]:
                        relation = f"same {key}"
                        break
                rank = 1 + int(np.sum(negatives > S[a, j]))
                records.append({
                    "query": names[i], "positive": names[j], "group": group, "relation": relation,
                    "cosine": float(S[a, j]), "rank": rank, "n_candidates": len(negatives) + 1,
                    "rank_fraction": rank / (len(negatives) + 1),
                })
    out = pd.DataFrame(records)
    if not relation_keys:
        out = out.drop(columns="relation")
    return out


def scrambled_treatments(
    adata: ad.AnnData,
    *,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    batch_key: str = "Batch",
    group_size: int | None = None,
    prefix: str = "scrambled_",
    random_state: int = 0,
) -> pd.Series:
    """Relabel treated wells into random same-batch groups of different treatments.

    Each scrambled group takes ``group_size`` treated wells of one batch, every
    one from a different treatment. Scored like real treatments (e.g. with
    ``replicate_map``), the groups estimate how often unrelated treated wells
    look active. Unlike control-well pseudo-treatments (``pseudo_treatment_null``),
    they keep whatever treated wells share and controls lack: compound handling,
    solvent, the wells the layout reserves for compounds, and an origin that
    differs from the controls. A method whose scrambled groups score well above
    the nominal rate inflates activity calls.

    Returns a Series indexed like ``adata.obs``: ``control_value`` for controls,
    ``"<prefix><batch>_<k>"`` for assigned wells, and NaN for treated wells left
    over when a batch's count is not a multiple of ``group_size`` (subset to
    ``labels.notna()`` before scoring). ``group_size`` defaults to the most
    common number of wells per treatment within a batch.
    """
    if treatment_key not in adata.obs:
        raise KeyError(f"Missing {treatment_key!r} in obs.")
    if adata.obs[treatment_key].isna().any():
        raise ValueError(f"Treatment labels in {treatment_key!r} must not be missing.")
    batch = _batches(adata, batch_key)
    label = adata.obs[treatment_key].astype(str).to_numpy()
    treated = label != str(control_value)
    if not treated.any():
        raise ValueError("No treated wells.")
    sizes = pd.Series(1, index=pd.MultiIndex.from_arrays([batch[treated], label[treated]])).groupby(level=[0, 1]).size()
    if group_size is None:
        group_size = int(sizes.mode().iloc[0])
    if not isinstance(group_size, (int, np.integer)) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2.")

    rng = np.random.default_rng(random_state)
    out = np.full(adata.n_obs, np.nan, dtype=object)
    out[~treated] = str(control_value)
    for b in pd.unique(batch[treated]):
        wells = np.flatnonzero(treated & (batch == b))
        n_groups = len(wells) // group_size
        per_treatment = sizes.loc[b]
        if n_groups < 1 or per_treatment.max() > n_groups:
            raise ValueError(f"Batch {b!r}: too few treated wells for groups of {group_size} different treatments.")
        # Contiguous treatment blocks in random order, dealt round-robin into
        # groups: a treatment with <= n_groups wells never repeats in a group.
        order = rng.permutation(len(wells))
        blocks = rng.permutation(pd.unique(label[wells]))
        rank = {t: r for r, t in enumerate(blocks)}
        ordered = wells[order][np.argsort([rank[t] for t in label[wells][order]], kind="stable")]
        assigned = ordered[: n_groups * group_size]
        group = np.arange(len(assigned)) % n_groups
        out[assigned] = [f"{prefix}{b}_{g}" for g in group]
    clash = set(out[treated & pd.notna(out)]) & set(label)
    if clash:
        raise ValueError(f"Treatment labels already use scrambled names, e.g. {sorted(clash)[:3]}.")
    return pd.Series(out, index=adata.obs_names, name=f"{prefix.rstrip('_')}_{treatment_key}")
