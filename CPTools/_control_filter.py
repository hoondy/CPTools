"""Feature QC from repeated copies of the same control condition."""
from __future__ import annotations

from typing import Mapping
import warnings

import anndata as ad
import numpy as np
from scipy import stats

from ._reference_normalization import _query_mask, _source
from ._utils import bh_fdr


def control_agreement_filter(
    adata: ad.AnnData,
    *,
    copies: Mapping[str, tuple[str, str]],
    batch_key: str | None = None,
    source_layer: str | None = None,
    alpha: float = 0.05,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """Flag features on which two copies of the same condition disagree beyond well noise.

    ``copies`` maps a name to two ``obs.query`` selections of the *same*
    biological condition in different well sets, for example dedicated and
    scattered WT+DMSO wells: ``{"DMSO": ("Drug_code == 'c_DM'",
    "Drug_code == 'c_DM_rand' and line == 'WT'")}``. Every pair is compared
    within each batch (``batch_key``) where both copies are present, so plate
    offsets cancel. With ``batch_key=None`` the queries themselves must
    restrict the comparison (e.g. ``Batch == 'WT_CTRL' and ...``).

    Per feature and comparison k, z_k = (mean_b - mean_a) / (s * sqrt(1/n_a +
    1/n_b)), with s the feature's pooled within-copy SD over all compared
    wells. The statistic is mean_k(z_k^2): about 1 when the copies differ only
    by well noise. Under that null, it follows F(K, df) with K comparisons and
    df pooled within-copy degrees of freedom. Features with BH q < ``alpha``
    fail; features without enough finite data also fail.

    Only control wells are used, so no genotype or treatment effect enters the
    selection. The statistic is invariant to per-feature scaling and, within a
    batch, to per-plate centering, so raw or normalized input give the same
    result.

    Writes ``var["control_agreement_stat"]``, ``var["control_agreement_q"]`` and
    ``var["pass_control_agreement"]``; ``uns["cptools"]["control_agreement_filter"]``
    holds a summary, including each comparison's median |z| relative to its
    pure-noise value (0.6745): near 1 means the copies agree in typical
    features; well above 1 means a broad disagreement that dropping a few
    features will not remove. By default (``subset=False``) features are only
    flagged.
    """
    if not copies:
        raise ValueError("Provide at least one {name: (query_a, query_b)} pair.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    target = adata if inplace else adata.copy()
    X = _source(target, source_layer)
    F = X.shape[1]
    if batch_key is None:
        batches, batch = [None], None
    else:
        if batch_key not in target.obs:
            raise KeyError(f"Missing batch column {batch_key!r}.")
        batch = target.obs[batch_key].astype(str).to_numpy()
        batches = list(dict.fromkeys(batch))

    sse, dof = np.zeros(F), np.zeros(F)
    differences, inverse_n, labels = [], [], []
    for name, pair in copies.items():
        if len(pair) != 2:
            raise ValueError(f"Copies {name!r} must be two queries.")
        mask_a, mask_b = (_query_mask(target.obs, q) for q in pair)
        if np.any(mask_a & mask_b):
            raise ValueError(f"The two copies of {name!r} share wells.")
        for plate in batches:
            in_plate = np.ones(target.n_obs, dtype=bool) if plate is None else batch == plate
            xa, xb = X[mask_a & in_plate], X[mask_b & in_plate]
            if len(xa) == 0 or len(xb) == 0:
                continue
            na, nb = np.isfinite(xa).sum(axis=0), np.isfinite(xb).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_a, mean_b = np.nanmean(xa, axis=0), np.nanmean(xb, axis=0)
            sse += np.nansum((xa - mean_a) ** 2, axis=0) + np.nansum((xb - mean_b) ** 2, axis=0)
            dof += np.maximum(na - 1, 0) + np.maximum(nb - 1, 0)
            differences.append(mean_b - mean_a)
            with np.errstate(divide="ignore"):
                inverse_n.append(np.where((na > 0) & (nb > 0), 1 / np.maximum(na, 1) + 1 / np.maximum(nb, 1), np.nan))
            labels.append(name if plate is None else f"{name} | {plate}")
    if not differences:
        raise ValueError("No comparison has both copies present; check the queries and batch_key.")

    D, inv = np.vstack(differences), np.vstack(inverse_n)
    variance = np.divide(sse, dof, out=np.full(F, np.nan), where=dof > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = D / np.sqrt(variance * inv)
        z = np.where((D == 0) & (variance == 0), 0.0, z)      # constant and equal: no evidence
    finite = np.isfinite(z)
    k = finite.sum(axis=0)
    with np.errstate(invalid="ignore"):
        statistic = np.where(k > 0, np.nansum(np.where(finite, z, 0) ** 2, axis=0) / np.maximum(k, 1), np.nan)
    p = np.ones(F)
    testable = (k > 0) & (dof > 0) & np.isfinite(statistic)
    p[testable] = stats.f.sf(statistic[testable], k[testable], dof[testable])
    q = np.full(F, np.nan)
    q[testable] = bh_fdr(p[testable])
    passed = testable & (q >= alpha)

    target.var["control_agreement_stat"] = statistic.astype(np.float32)
    target.var["control_agreement_q"] = q.astype(np.float32)
    target.var["pass_control_agreement"] = passed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = {label: float(np.nanmedian(np.abs(row)) / 0.6745) for label, row in zip(labels, z)}
    target.uns.setdefault("cptools", {})["control_agreement_filter"] = {
        "copies": {name: list(pair) for name, pair in copies.items()},
        "batch_key": batch_key or "", "source_layer": source_layer or "", "alpha": alpha,
        "comparisons": labels, "median_abs_z_ratio": typical,
        "median_dof": float(np.nanmedian(dof)), "n_flagged": int((~passed).sum()),
        "n_untestable": int((~testable).sum()), "n_features": int(F),
    }
    if (~testable).any():
        warnings.warn(f"{int((~testable).sum())} features lack finite data for the comparison and are flagged.",
                      stacklevel=2)
    broad = {label: ratio for label, ratio in typical.items() if ratio > 1.5}
    if broad:
        warnings.warn(
            "Copies disagree in typical features, not just a few (median |z| / noise > 1.5 for "
            f"{sorted(broad)}). Dropping flagged features will not remove a disagreement this broad; "
            "consider correcting it instead.",
            stacklevel=2,
        )
    if subset:
        target._inplace_subset_var(passed)
    return target
