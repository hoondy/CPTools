"""Pseudo-treatments drawn from control wells: an empirical null for any score."""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from ._reference_normalization import _batches, _query_mask
from ._utils import bh_fdr

PSEUDO_PREFIX = "pseudo_"


def _distinct_groups(n, k, count, rng):
    """``count`` distinct k-subsets of range(n), or all of them if fewer exist."""
    total = comb(n, k)
    if total <= 100_000:
        combos = np.asarray(list(combinations(range(n), k)))
        return combos if count >= total else combos[rng.choice(total, size=count, replace=False)]
    seen = {}
    while len(seen) < count:
        seen.setdefault(tuple(np.sort(rng.choice(n, size=k, replace=False))), None)
    return np.asarray(list(seen))


def pseudo_treatment_null(
    adata: ad.AnnData,
    *,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    batch_key: str = "Batch",
    n_per_group: int = 3,
    n_draws: int = 1000,
    groupby: str | Sequence[str] | None = None,
    query: str | None = None,
    include_query: str | None = None,
    min_remaining: int = 2,
    random_state: int = 0,
) -> ad.AnnData:
    """Draw random same-batch groups of control wells as fake treatments.

    Each draw picks ``n_per_group`` control wells from one stratum (a batch, or
    a batch x ``groupby`` cell such as batch x genotype) and becomes its own
    pseudo-batch ``"<batch>|pseudo_<d>"``: copies of *every* control well of
    the source batch (plus ``include_query`` wells), with the drawn wells
    relabeled ``"pseudo_<d>"`` in ``treatment_key``. Scores that compare a
    treatment with same-batch controls (``treatment_vectors``,
    ``rescue_scores``, ...) can then run unchanged on the result; each
    pseudo-treatment is compared with the remaining controls, never with
    itself. Use the scores as an empirical null with ``empirical_pvalues``.

    ``query`` restricts which control wells may be drawn (e.g. drug-screen
    plates only, or mutant DMSO wells for ``rescue_scores``).
    ``include_query`` selects further wells copied unchanged into every
    pseudo-batch of their plate, e.g. WT references labeled differently from
    the drawn vehicle wells. Strata need ``n_per_group + min_remaining``
    drawable wells. Draws are spread evenly over eligible strata and are
    distinct within a stratum; if a stratum has fewer distinct groups than its
    share, all of them are used and a warning gives the smaller total.

    Added obs columns: ``pseudo_draw``, ``pseudo_role`` (``"pseudo_treatment"``
    or ``"control"``), ``pseudo_source_batch`` and ``pseudo_source_well``. X,
    layers, obsm, var and varm are copied; obsp and uns are not. Parameters are
    stored in ``uns["pseudo_treatment_null"]``; the result writes to H5AD.

    Caveats: a pseudo-treatment's controls are ``n_per_group`` fewer than a
    real treatment's, so its centroid difference is slightly noisier (a
    conservative null for magnitude scores). Groups from one plate share wells,
    so null scores are dependent and a plate with c controls offers only
    C(c, n_per_group) groups. Control wells in fixed positions do not carry the
    position or solvent effects of treatment wells; scores driven by those
    effects will look significant against this null, which is the point.
    """
    for name, value in (("n_per_group", n_per_group), ("n_draws", n_draws), ("min_remaining", min_remaining)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1.")
    keys = [groupby] if isinstance(groupby, str) else list(groupby or [])
    for key in [treatment_key, *keys]:
        if key not in adata.obs:
            raise KeyError(f"Missing {key!r} in obs.")
    batch = _batches(adata, batch_key)
    if adata.obs[treatment_key].isna().any():
        raise ValueError(f"Treatment labels in {treatment_key!r} must not be missing.")
    label = adata.obs[treatment_key].astype(str).to_numpy()
    control = label == str(control_value)
    drawable = control & (_query_mask(adata.obs, query) if query is not None else True)

    frame = adata.obs.loc[drawable, keys].copy()
    frame.insert(0, "__batch", batch[drawable])
    frame["__position"] = np.flatnonzero(drawable)
    strata = [
        (values[0], group["__position"].to_numpy())
        for values, group in frame.groupby(["__batch", *keys], sort=True, dropna=False, observed=True)
        if len(group) >= n_per_group + min_remaining
    ]
    if not strata:
        raise ValueError(
            f"No batch{' x ' + str(keys) if keys else ''} stratum has {n_per_group + min_remaining} drawable "
            f"control wells (n_per_group + min_remaining).")

    rng = np.random.default_rng(random_state)
    shares = np.full(len(strata), n_draws // len(strata))
    shares[rng.choice(len(strata), size=n_draws % len(strata), replace=False)] += 1
    draws = []
    for (source, positions), share in zip(strata, shares):
        if share:
            draws += [(source, positions[g]) for g in _distinct_groups(len(positions), n_per_group, share, rng)]
    if len(draws) < n_draws:
        warnings.warn(f"Only {len(draws)} distinct pseudo-treatments exist (requested {n_draws}).", stacklevel=2)

    width = max(4, len(str(len(draws) - 1)))
    names = np.asarray([f"{PSEUDO_PREFIX}{d:0{width}d}" for d in range(len(draws))], dtype=object)
    clash = set(names) & set(label)
    if clash:
        raise ValueError(f"Treatment labels already use pseudo-treatment names, e.g. {sorted(clash)[:3]}.")
    copied = control | (_query_mask(adata.obs, include_query) if include_query is not None else False)
    members = {b: np.flatnonzero(copied & (batch == b)) for b in {source for source, _ in draws}}
    rows = np.concatenate([members[source] for source, _ in draws])
    draw = np.concatenate([np.full(len(members[source]), d) for d, (source, _) in enumerate(draws)])
    drawn = np.concatenate([np.isin(members[source], chosen) for source, chosen in draws])

    obs = adata.obs.iloc[rows].copy()
    source_wells = adata.obs_names.to_numpy().astype(str)[rows]
    obs.index = pd.Index([f"{well}|{name}" for well, name in zip(source_wells, names[draw])])
    obs[treatment_key] = pd.Categorical(np.where(drawn, names[draw], label[rows]))
    obs[batch_key] = pd.Categorical([f"{b}|{name}" for b, name in zip(batch[rows], names[draw])])
    obs["pseudo_draw"] = draw.astype(np.int64)
    obs["pseudo_role"] = pd.Categorical(
        np.where(drawn, "pseudo_treatment", "control"), categories=["pseudo_treatment", "control"])
    obs["pseudo_source_batch"] = pd.Categorical(batch[rows])
    obs["pseudo_source_well"] = pd.Categorical(source_wells)

    def take(matrix):
        if isinstance(matrix, pd.DataFrame):
            return matrix.iloc[rows].set_axis(obs.index)
        return matrix[rows]

    out = ad.AnnData(
        X=None if adata.X is None else take(adata.X),
        obs=obs, var=adata.var.copy(),
        # anndata >= 0.13 also lists X as layers[None].
        layers={key: take(value) for key, value in adata.layers.items() if key is not None},
        obsm={key: take(value) for key, value in adata.obsm.items()},
        varm={key: value for key, value in adata.varm.items()},
    )
    out.uns["pseudo_treatment_null"] = {
        "treatment_key": treatment_key, "control_value": str(control_value), "batch_key": batch_key,
        "n_per_group": int(n_per_group), "n_draws": len(draws), "groupby": [str(k) for k in keys],
        "query": query or "", "include_query": include_query or "", "min_remaining": int(min_remaining),
        **({"random_state": int(random_state)} if random_state is not None else {}),
    }
    return out


def empirical_pvalues(
    observed: pd.Series | Sequence[float],
    null: Sequence[float],
    *,
    alternative: str = "greater",
) -> pd.DataFrame:
    """Empirical p-values of observed scores against null scores.

    ``null`` is typically the same score computed on pseudo-treatments from
    ``pseudo_treatment_null``. With n null scores, ``"greater"`` gives
    p = (1 + #{null >= score}) / (1 + n), ``"less"`` uses null <= score, and
    ``"two-sided"`` is min(1, 2 x the smaller one-sided p). The smallest
    attainable p is 1 / (1 + n). Returns a table indexed like ``observed`` with
    ``score``, ``null_z`` ((score - null mean) / null SD), ``p_value`` and BH
    ``q_value`` over the finite scores; missing scores get NaN.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'.")
    scores = observed if isinstance(observed, pd.Series) else pd.Series(np.asarray(observed, dtype=float))
    values = pd.to_numeric(scores).to_numpy(dtype=float)
    reference = np.sort(np.asarray(null, dtype=float).ravel())
    if reference.size == 0:
        raise ValueError("null is empty.")
    if not np.isfinite(reference).all():
        raise ValueError("null contains nonfinite scores; remove them explicitly before testing.")
    n = reference.size
    greater = (1 + n - np.searchsorted(reference, values, side="left")) / (1 + n)
    less = (1 + np.searchsorted(reference, values, side="right")) / (1 + n)
    p = {"greater": greater, "less": less, "two-sided": np.minimum(1.0, 2 * np.minimum(greater, less))}[alternative]
    finite = np.isfinite(values)
    p = np.where(finite, p, np.nan)
    q = np.full(len(p), np.nan)
    q[finite] = bh_fdr(p[finite])
    sd = reference.std(ddof=1) if n > 1 else 0.0
    z = (values - reference.mean()) / sd if sd > 0 else np.full(len(values), np.nan)
    return pd.DataFrame({"score": values, "null_z": z, "p_value": p, "q_value": q}, index=scores.index)
