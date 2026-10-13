"""Control-only normalization for designs with shared references on every plate."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _query_mask(obs: pd.DataFrame, query: str) -> np.ndarray:
    if not obs.index.is_unique:
        raise ValueError("Observation names must be unique before selecting references.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Supply a nonempty pandas query selecting the intended controls.")
    return np.asarray(obs.index.isin(obs.query(query).index))


def _source(adata: ad.AnnData, layer: str | None) -> np.ndarray:
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Source layer {layer!r} is missing; no fallback is performed.")
    matrix = adata.X if layer is None else adata.layers[layer]
    return np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=float)


def _batches(adata: ad.AnnData, key: str) -> np.ndarray:
    if key not in adata.obs:
        raise KeyError(f"Missing batch column {key!r}.")
    if adata.obs[key].isna().any() or (adata.obs[key].astype(str).str.strip() == "").any():
        raise ValueError("Batch identifiers must not be missing or empty.")
    return adata.obs[key].astype(str).to_numpy()


def fit_reference_normalization(
    adata: ad.AnnData,
    *,
    reference_query: str,
    batch_key: str = "Batch",
    source_layer: str | None = "raw",
    fit_query: str | None = None,
    center: str = "mean",
    min_controls: int = 2,
    min_scale: float = 1e-8,
    positive_control_queries: Mapping[str, str] | None = None,
    gain_ridge: float = 10.0,
    gain_bounds: tuple[float, float] = (0.5, 2.0),
    min_positive_snr: float = 2.0,
    max_gain_disagreement: float = 2.0,
) -> dict:
    """Fit plate centers and pooled within-plate reference SD, without mutation.

    Queries are pandas ``obs.query`` expressions. ``fit_query`` excludes held-out
    wells from *all* fitting, including scale and positive-control consensus.
    At least ``min_controls`` reference wells are required on every plate.
    Variance uses within-plate means and n-1 degrees of freedom even when the
    location estimator is ``median``. This is pooled SD, not pooled MAD.

    Optional positive references must represent identical biology/dose/time on
    every plate. Their effects relative to the negative reference are fitted to
    the feature-wise median effect across plates. Ridge shrinks each gain to 1
    in pooled-SD units. Gains revert to 1 unless *every* supplied positive has
    sufficient signal, agrees in sign, and supports a consistent bounded gain.
    Positive calibration is experimental; validate with held-out controls.

    Features with insufficient finite references or pooled SD <= min_scale are
    marked invalid and become NaN on transform, never divided by an epsilon.
    The returned dictionary can be stored in AnnData.uns / H5AD.
    """
    if center not in {"mean", "median"}:
        raise ValueError("center must be 'mean' or 'median'.")
    if not isinstance(min_controls, (int, np.integer)) or min_controls < 2:
        raise ValueError("min_controls must be an integer >= 2.")
    if not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("min_scale must be finite and > 0.")
    if not np.isfinite(gain_ridge) or gain_ridge < 0:
        raise ValueError("gain_ridge must be finite and >= 0.")
    if len(gain_bounds) != 2 or not (0 < gain_bounds[0] <= 1 <= gain_bounds[1] < np.inf):
        raise ValueError("gain_bounds must be finite, positive, and enclose 1.")
    if not np.isfinite(min_positive_snr) or min_positive_snr <= 0:
        raise ValueError("min_positive_snr must be finite and > 0.")
    if not np.isfinite(max_gain_disagreement) or max_gain_disagreement < 1:
        raise ValueError("max_gain_disagreement must be finite and >= 1.")
    if not adata.var_names.is_unique or adata.n_vars == 0:
        raise ValueError("Features must be nonempty and have unique names.")

    X = _source(adata, source_layer)
    batch = _batches(adata, batch_key)
    plates = np.unique(batch)
    if len(plates) == 0:
        raise ValueError("No observations available for fitting.")
    fit = np.ones(adata.n_obs, dtype=bool) if fit_query is None else _query_mask(adata.obs, fit_query)
    reference = _query_mask(adata.obs, reference_query)
    positives = dict(positive_control_queries or {})
    positive_masks = {name: _query_mask(adata.obs, query) for name, query in positives.items()}
    occupied = reference.copy()
    for name, mask in positive_masks.items():
        if np.any(occupied & mask):
            raise ValueError(f"Positive reference {name!r} overlaps another reference condition.")
        occupied |= mask
    if positives and len(plates) < 3:
        raise ValueError("Positive-control gain fitting requires at least three plates.")

    shape = (len(plates), adata.n_vars)
    centers, sse, counts = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=int)
    reference_counts = np.zeros(len(plates), dtype=int)
    positive_counts = np.zeros((len(plates), len(positives)), dtype=int)
    effects = np.full((len(plates), len(positives), adata.n_vars), np.nan)
    positive_finite = np.ones(shape, dtype=bool)
    for p, plate in enumerate(plates):
        mask = (batch == plate) & fit & reference
        reference_counts[p] = mask.sum()
        if reference_counts[p] < min_controls:
            raise ValueError(f"Plate {plate!r}: {reference_counts[p]} fitting references; need {min_controls}.")
        values = X[mask]
        finite = np.isfinite(values)
        counts[p] = finite.sum(axis=0)
        means = np.divide(np.where(finite, values, 0).sum(axis=0), counts[p],
                          out=np.full(adata.n_vars, np.nan), where=counts[p] > 0)
        if center == "mean":
            centers[p] = means
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                centers[p] = np.nanmedian(np.where(finite, values, np.nan), axis=0)
        sse[p] = (np.where(finite, values - means, 0) ** 2).sum(axis=0)
        for c, (name, pos_mask) in enumerate(positive_masks.items()):
            selected = (batch == plate) & fit & pos_mask
            positive_counts[p, c] = selected.sum()
            if selected.sum() < min_controls:
                raise ValueError(f"Plate {plate!r}: positive {name!r} has fewer than {min_controls} fitting wells.")
            pos = X[selected]
            pos_finite = np.isfinite(pos)
            n = pos_finite.sum(axis=0)
            pos_mean = np.divide(np.where(pos_finite, pos, 0).sum(axis=0), n,
                                 out=np.full(adata.n_vars, np.nan), where=n > 0)
            effects[p, c] = pos_mean - centers[p]
            positive_finite[p] &= n >= min_controls

    dof = np.maximum(counts - 1, 0).sum(axis=0)
    raw_sd = np.sqrt(np.divide(sse.sum(axis=0), dof,
                              out=np.full(adata.n_vars, np.nan), where=dof > 0))
    valid = (counts >= min_controls).all(axis=0) & np.isfinite(raw_sd) & (raw_sd > min_scale)
    gains = np.ones(shape)
    gain_applied = np.zeros(shape, dtype=bool)
    consensus = np.empty((0, adata.n_vars))
    if positives:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            consensus = np.nanmedian(effects, axis=0)
        # Only valid features are divided, and all requested anchors must support a gain.
        safe_sd = np.where(valid, raw_sd, np.nan)
        u = consensus / safe_sd
        for p in range(len(plates)):
            v = effects[p] / safe_sd
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = v / u
                estimate = ((u * v).sum(axis=0) + gain_ridge) / ((u * u).sum(axis=0) + gain_ridge)
                agreement = ratios.max(axis=0) / ratios.min(axis=0)
            supported = (valid & positive_finite.all(axis=0)
                         & np.isfinite(u).all(axis=0) & np.isfinite(v).all(axis=0)
                         & (np.abs(u) >= min_positive_snr).all(axis=0)
                         & (np.abs(v) >= min_positive_snr).all(axis=0)
                         & (u * v > 0).all(axis=0)
                         & (agreement <= max_gain_disagreement)
                         & (estimate >= gain_bounds[0]) & (estimate <= gain_bounds[1]))
            gains[p, supported] = estimate[supported]
            gain_applied[p] = supported

    scale = np.sqrt(np.divide((sse / gains**2).sum(axis=0), dof,
                             out=np.full(adata.n_vars, np.nan), where=dof > 0))
    valid &= np.isfinite(scale) & (scale > min_scale)
    if not valid.any():
        raise ValueError("No features have sufficient finite reference data and nonzero pooled scale.")
    if not valid.all():
        warnings.warn(f"{int((~valid).sum())} features cannot be normalized; they will be marked invalid and set to NaN.",
                      stacklevel=2)
    return {
        "method": "reference_pooled_sd", "batch_key": batch_key,
        "reference_query": reference_query, "fit_query": fit_query or "",
        "source_layer": source_layer if source_layer is not None else "",
        "center": center, "min_controls": min_controls, "min_scale": min_scale,
        "features": adata.var_names.to_numpy(dtype=str), "batches": plates,
        "centers": centers, "gains": gains, "gain_applied": gain_applied,
        "pooled_sd": scale, "pooled_sd_before_gain": raw_sd, "valid_features": valid,
        "reference_counts": reference_counts, "finite_reference_counts": counts,
        "residual_dof": dof, "positive_control_queries": positives,
        "positive_counts": positive_counts, "positive_consensus": consensus,
        "gain_ridge": gain_ridge, "gain_bounds": np.asarray(gain_bounds, dtype=float),
        "min_positive_snr": min_positive_snr, "max_gain_disagreement": max_gain_disagreement,
        "fit_reference_obs_names": adata.obs_names[fit & occupied].to_numpy(dtype=str),
    }


def apply_reference_normalization(
    adata: ad.AnnData,
    model: dict,
    *,
    source_layer: str | None = "raw",
    output_layer: str = "normalized",
    inplace: bool = True,
) -> ad.AnnData:
    """Apply a fitted model to wells from known plates with the same ordered features.

    Sets X and output_layer; preserves the source. Invalid features become NaN
    and are flagged in var['pass_reference_normalization']. Parameters are saved
    in uns['cptools']['reference_normalize']. Subset features *after* applying.
    """
    if not output_layer or output_layer == "raw" or output_layer == source_layer:
        raise ValueError("output_layer must be nonempty and distinct from the source and 'raw'.")
    if not np.array_equal(adata.var_names.to_numpy(dtype=str), model["features"]):
        raise ValueError("Model and input must have identical ordered features.")
    batch = _batches(adata, model["batch_key"])
    plates = np.asarray(model["batches"])
    unknown = sorted(set(batch) - set(plates))
    if unknown:
        raise ValueError(f"Model has no reference parameters for plates: {unknown}")
    X = _source(adata, source_layer)
    output = np.full(X.shape, np.nan, dtype=np.float32)
    valid = np.asarray(model["valid_features"], dtype=bool)
    for p, plate in enumerate(plates):
        selected = batch == plate
        output[np.ix_(selected, valid)] = (
            (X[np.ix_(selected, valid)] - model["centers"][p, valid])
            / (model["gains"][p, valid] * model["pooled_sd"][valid])
        ).astype(np.float32)
    target = adata if inplace else adata.copy()
    if source_layer is None and "raw" not in target.layers:
        target.layers["raw"] = X.copy()
    target.X = output
    target.layers[output_layer] = output.copy()
    target.var["pass_reference_normalization"] = valid
    target.var["reference_pooled_sd"] = np.asarray(model["pooled_sd"])
    target.uns.setdefault("cptools", {})["reference_normalize"] = deepcopy(model)
    return target


def reference_normalize(
    adata: ad.AnnData,
    *,
    reference_query: str,
    batch_key: str = "Batch",
    source_layer: str | None = "raw",
    output_layer: str = "normalized",
    inplace: bool = True,
    **fit_options,
) -> ad.AnnData:
    """Fit and apply reference normalization; see fit_reference_normalization."""
    model = fit_reference_normalization(
        adata, reference_query=reference_query, batch_key=batch_key,
        source_layer=source_layer, **fit_options,
    )
    return apply_reference_normalization(
        adata, model, source_layer=source_layer, output_layer=output_layer, inplace=inplace,
    )
