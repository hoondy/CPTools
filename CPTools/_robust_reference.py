"""Multi-anchor, shrinkage-stabilized reference normalization.

Extends the shared-reference normalizer for designs with only a few reference
wells per plate. Plate offsets are estimated from every anchor condition that is
repeated identically on every plate, shrunk toward zero when they are not larger
than their own estimation noise, and the feature scale can be estimated from
replicate residuals of all replicated groups instead of the few reference wells.
The returned model is consumed by ``apply_reference_normalization``.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from ._reference_normalization import _batches, _query_mask, _source


_LOCATION_VARIANCE_FACTOR = {"mean": 1.0, "median": np.pi / 2}


def _condition_summaries(X, mask, batch, plates, location):
    """Per-plate location, finite counts and SSE about the plate mean."""
    P, F = len(plates), X.shape[1]
    loc, sse = np.full((P, F), np.nan), np.zeros((P, F))
    n, wells = np.zeros((P, F), dtype=int), np.zeros(P, dtype=int)
    for p, plate in enumerate(plates):
        selected = mask & (batch == plate)
        wells[p] = selected.sum()
        values = X[selected]
        finite = np.isfinite(values)
        n[p] = finite.sum(axis=0)
        means = np.divide(np.where(finite, values, 0).sum(axis=0), n[p],
                          out=np.full(F, np.nan), where=n[p] > 0)
        if location == "mean":
            loc[p] = means
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                loc[p] = np.nanmedian(np.where(finite, values, np.nan), axis=0)
        sse[p] = (np.where(finite, values - means, 0) ** 2).sum(axis=0)
    return loc, n, sse, wells


def _additive_plate_fit(loc, n, max_iter=200, tol=1e-10):
    """Weighted fit of loc[c, p] = mu[c] + delta[p] with mean_p(delta) = 0.

    Weights are finite well counts. One condition reproduces its plate means
    exactly; with several, delta is the offset shared by all conditions.
    """
    w = np.where(n > 0, n, 0).astype(float)
    y = np.where(n > 0, loc, 0.0)
    delta = np.zeros(loc.shape[1:])
    for _ in range(max_iter):
        mu = np.divide((w * (y - delta[None])).sum(axis=1), w.sum(axis=1),
                       out=np.full((loc.shape[0], loc.shape[2]), np.nan), where=w.sum(axis=1) > 0)
        updated = np.divide((w * (y - np.nan_to_num(mu)[:, None])).sum(axis=0), w.sum(axis=0),
                            out=np.zeros(delta.shape), where=w.sum(axis=0) > 0)
        updated -= updated.mean(axis=0)
        converged = np.max(np.abs(updated - delta)) < tol
        delta = updated
        if converged:
            break
    mu = np.divide((w * (y - delta[None])).sum(axis=1), w.sum(axis=1),
                   out=np.full((loc.shape[0], loc.shape[2]), np.nan), where=w.sum(axis=1) > 0)
    return mu, delta


def _replicate_scale(X, obs, fit, batch_key, groupby, scale_query, estimator):
    """Pooled (or median) within-group variance from replicated groups."""
    selected = fit.copy()
    if scale_query:
        selected &= _query_mask(obs, scale_query)
    keys = [batch_key, *[key for key in groupby if key != batch_key]]
    missing = [key for key in keys if key not in obs]
    if missing:
        raise KeyError(f"Missing scale_groupby columns: {missing}")
    labels = obs.loc[selected, keys].astype(str)
    groups = labels.groupby(keys, observed=True, sort=False).ngroup().to_numpy()
    frame = pd.DataFrame(X[selected])
    counts = frame.notna().groupby(groups).sum().to_numpy()
    variances = frame.groupby(groups).var(ddof=1).to_numpy()
    dof = np.where(counts >= 2, counts - 1, 0)
    replicated = int((np.bincount(groups) >= 2).sum())
    if estimator == "pooled":
        sse = np.where(dof > 0, variances * dof, 0).sum(axis=0)
        total = dof.sum(axis=0)
        variance = np.divide(sse, total, out=np.full(X.shape[1], np.nan), where=total > 0)
    else:
        # Median of chi-square-calibrated group variances; robust to outlying groups.
        calibrated = np.full(variances.shape, np.nan)
        for df in np.unique(dof[dof > 0]):
            hit = dof == df
            calibrated[hit] = variances[hit] * df / stats.chi2.median(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.nanmedian(calibrated, axis=0)
        total = dof.sum(axis=0)
    return np.sqrt(variance), total, replicated


def fit_robust_reference_normalization(
    adata: ad.AnnData,
    *,
    reference_query: str,
    anchor_queries: Mapping[str, str] | None = None,
    batch_key: str = "Batch",
    source_layer: str | None = "raw",
    fit_query: str | None = None,
    location: str = "mean",
    plate_shrinkage: bool = True,
    scale: str = "reference",
    scale_groupby: Sequence[str] | None = None,
    scale_query: str | None = None,
    scale_estimator: str = "pooled",
    min_controls: int = 2,
    min_scale: float = 1e-8,
) -> dict:
    """Fit plate centers from several shared anchors with shrunk plate offsets.

    ``reference_query`` defines the origin (normalized value 0), e.g. WT+DMSO.
    ``anchor_queries`` optionally adds conditions repeated identically on every
    plate (same genotype, compound, dose, time), such as WT positive controls.
    Anchors only inform plate offsets; they never define the origin. Per feature:

        location[c, p] = mu[c] + delta[p] + error      (weighted additive fit)
        center[p]      = mu[reference] + B * delta[p]

    ``B`` is the positive-part Efron-Morris (James-Stein) factor
    ``max(0, 1 - (P - 3) * v / sum_p delta[p]**2)``, where ``v`` is the average
    variance of an estimated plate offset: well sampling noise of the anchors
    and, with two or more anchor conditions, the plate x condition interaction
    variance estimated from additive-fit residuals (plate shifts that do not
    transfer between conditions). When shared plate offsets are no larger than
    this noise, B -> 0 and every plate is centered on the across-plate reference
    mean instead of on two or three noisy or locally biased wells. With
    ``plate_shrinkage=False`` and no anchors, centers equal
    ``fit_reference_normalization``. Shrinkage needs at least four plates.

    ``scale`` chooses the per-feature denominator:

    - ``"reference"``: pooled within-plate SD of the reference wells.
    - ``"anchors"``: pooled within-plate SD across reference and anchor wells.
    - ``"replicate"``: within-group SD of replicate groups defined by
      ``batch_key`` + ``scale_groupby`` (e.g. ``["line", "Treatment"]``) among
      ``scale_query`` wells. Only residuals around each group mean are used, so
      no disease or treatment effect enters the scale. Dispersed screening
      triplicates usually describe plate-wide well-to-well noise better than
      adjacent reference wells. ``scale_estimator="median"`` resists toxic or
      failing groups with inflated variance.

    Features lacking ``min_controls`` finite reference wells on any plate, or
    with scale <= ``min_scale``, are invalid and become NaN when applied.
    Validate choices with held-out controls (``reference_control_cv`` with
    ``robust_options``), never with downstream rescue rankings.
    """
    if location not in _LOCATION_VARIANCE_FACTOR:
        raise ValueError("location must be 'mean' or 'median'.")
    if scale not in {"reference", "anchors", "replicate"}:
        raise ValueError("scale must be 'reference', 'anchors' or 'replicate'.")
    if scale_estimator not in {"pooled", "median"}:
        raise ValueError("scale_estimator must be 'pooled' or 'median'.")
    if scale == "replicate" and not scale_groupby:
        raise ValueError("scale='replicate' requires scale_groupby, e.g. ['line', 'Treatment'].")
    if scale != "replicate" and (scale_groupby or scale_query):
        raise ValueError("scale_groupby/scale_query are used only with scale='replicate'.")
    if not isinstance(min_controls, (int, np.integer)) or min_controls < 2:
        raise ValueError("min_controls must be an integer >= 2.")
    if not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("min_scale must be finite and > 0.")
    if not adata.var_names.is_unique or adata.n_vars == 0:
        raise ValueError("Features must be nonempty and have unique names.")
    anchors = dict(anchor_queries or {})
    if "reference" in anchors:
        raise ValueError("Anchor names must differ from 'reference'.")

    X = _source(adata, source_layer)
    batch = _batches(adata, batch_key)
    plates = np.unique(batch)
    fit = np.ones(adata.n_obs, dtype=bool) if fit_query is None else _query_mask(adata.obs, fit_query)
    names = ["reference", *anchors]
    masks = [_query_mask(adata.obs, reference_query)] + [_query_mask(adata.obs, q) for q in anchors.values()]
    occupied = np.zeros(adata.n_obs, dtype=bool)
    for name, mask in zip(names, masks):
        if np.any(occupied & mask):
            raise ValueError(f"Condition {name!r} overlaps another reference or anchor condition.")
        occupied |= mask

    P, C, F = len(plates), len(names), adata.n_vars
    loc, n, sse = np.full((C, P, F), np.nan), np.zeros((C, P, F), dtype=int), np.zeros((C, P, F))
    wells = np.zeros((P, C), dtype=int)
    for c, (name, mask) in enumerate(zip(names, masks)):
        loc[c], n[c], sse[c], wells[:, c] = _condition_summaries(X, mask & fit, batch, plates, location)
        short = np.flatnonzero(wells[:, c] < min_controls)
        if short.size:
            plate = plates[short[0]]
            raise ValueError(f"Plate {plate!r}: {name!r} has {wells[short[0], c]} fitting wells; need {min_controls}.")

    dof = np.maximum(n - 1, 0).sum(axis=1)                       # (C, F)
    within_var = np.divide(sse.sum(axis=1), dof, out=np.full((C, F), np.nan), where=dof > 0)
    mu, delta = _additive_plate_fit(loc, n)

    # Variance of each anchor's plate location around mu[c] + delta[p]: well
    # sampling noise plus, with >= 2 anchors, plate x condition interaction
    # (offsets that do not transfer between conditions, e.g. local effects).
    kappa = _LOCATION_VARIANCE_FACTOR[location]
    weight = n.astype(float)
    sampling = kappa * np.divide(np.nan_to_num(within_var)[:, None], weight,
                                 out=np.zeros((C, P, F)), where=weight > 0)
    interaction = np.zeros(F)
    if C >= 2 and P >= 2:
        observed = n > 0
        residual = np.where(observed, loc - mu[:, None] - delta[None], 0.0)
        cells = observed.sum(axis=(0, 1))
        dof_interaction = np.maximum(cells - C - P + 1, 1)
        mean_sampling = np.divide(np.where(observed, sampling, 0).sum(axis=(0, 1)), cells,
                                  out=np.zeros(F), where=cells > 0)
        interaction = np.maximum((residual ** 2).sum(axis=(0, 1)) / dof_interaction - mean_sampling, 0)
    offset_var = np.divide((weight ** 2 * (sampling + interaction)).sum(axis=0), weight.sum(axis=0) ** 2,
                           out=np.full((P, F), np.nan), where=weight.sum(axis=0) > 0)
    noise = np.nanmean(offset_var, axis=0)
    shrink = np.ones(F)
    shrinkage_method = "none"
    if plate_shrinkage:
        if P < 4:
            warnings.warn("Plate-offset shrinkage needs at least four plates; using unshrunk offsets.",
                          stacklevel=2)
        else:
            shrinkage_method = "james-stein"
            spread = (delta ** 2).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                shrink = np.clip(1 - (P - 3) * noise / spread, 0, 1)
            shrink = np.where(np.isfinite(shrink), shrink, 0.0)
    centers = mu[0][None] + shrink[None] * delta

    if scale == "replicate":
        sd, scale_dof, n_groups = _replicate_scale(
            X, adata.obs, fit, batch_key, list(scale_groupby), scale_query, scale_estimator)
    else:
        used = slice(0, 1) if scale == "reference" else slice(None)
        scale_dof = dof[used].sum(axis=0)
        if scale_estimator == "pooled":
            sd = np.sqrt(np.divide(sse[used].sum(axis=(0, 1)), scale_dof,
                                   out=np.full(F, np.nan), where=scale_dof > 0))
        else:
            per_condition = np.divide(sse[used].sum(axis=1), dof[used],
                                      out=np.full((dof[used].shape[0], F), np.nan), where=dof[used] > 0)
            sd = np.sqrt(np.nanmedian(per_condition, axis=0))
        n_groups = int(wells[:, used].size)

    valid = ((n[0] >= min_controls).all(axis=0) & np.isfinite(centers).all(axis=0)
             & np.isfinite(sd) & (sd > min_scale))
    if not valid.any():
        raise ValueError("No features have sufficient finite reference data and nonzero scale.")
    if not valid.all():
        warnings.warn(f"{int((~valid).sum())} features cannot be normalized; they will be marked invalid and set to NaN.",
                      stacklevel=2)
    return {
        "method": "reference_robust", "batch_key": batch_key,
        "reference_query": reference_query, "anchor_queries": anchors,
        "fit_query": fit_query or "", "source_layer": source_layer if source_layer is not None else "",
        "location": location, "plate_shrinkage": shrinkage_method,
        "scale": scale, "scale_estimator": scale_estimator,
        "scale_groupby": np.asarray(list(scale_groupby or []), dtype=str),
        "scale_query": scale_query or "", "scale_dof": scale_dof, "scale_n_groups": int(n_groups),
        "min_controls": min_controls, "min_scale": min_scale,
        "features": adata.var_names.to_numpy(dtype=str), "batches": plates,
        "centers": centers, "gains": np.ones((P, F)), "pooled_sd": sd, "valid_features": valid,
        "condition_names": np.asarray(names, dtype=str), "condition_means": mu,
        "condition_counts": wells, "condition_within_sd": np.sqrt(within_var),
        "plate_offsets": delta, "plate_offset_noise_var": noise, "shrinkage_factor": shrink,
        "plate_condition_interaction_var": interaction,
        "reference_counts": wells[:, 0], "finite_reference_counts": n[0],
        "fit_reference_obs_names": adata.obs_names[fit & occupied].to_numpy(dtype=str),
    }


def robust_reference_normalize(
    adata: ad.AnnData,
    *,
    reference_query: str,
    batch_key: str = "Batch",
    source_layer: str | None = "raw",
    output_layer: str = "normalized",
    inplace: bool = True,
    **fit_options,
) -> ad.AnnData:
    """Fit and apply robust reference normalization; see fit_robust_reference_normalization."""
    from ._reference_normalization import apply_reference_normalization

    model = fit_robust_reference_normalization(
        adata, reference_query=reference_query, batch_key=batch_key,
        source_layer=source_layer, **fit_options,
    )
    return apply_reference_normalization(
        adata, model, source_layer=source_layer, output_layer=output_layer, inplace=inplace,
    )
