from __future__ import annotations

import warnings
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from ._utils import to_dense_matrix


DEFAULT_BLOCKLIST_KEYWORDS = [
    "Manders",
    "RWC",
    "Location",
    "Granularity",
    "Execution",
    "Euler",
]


def _matrix_from_layer(adata: ad.AnnData, source_layer: str | None = None) -> np.ndarray:
    if source_layer is None:
        return to_dense_matrix(adata.X)
    if source_layer not in adata.layers:
        raise KeyError(f"Layer '{source_layer}' does not exist.")
    return to_dense_matrix(adata.layers[source_layer])


def _apply_var_mask(adata: ad.AnnData, mask: np.ndarray, inplace: bool) -> ad.AnnData:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if inplace:
        adata._inplace_subset_var(mask)
        return adata
    return adata[:, mask].copy()


def robust_zscore_norm(
    adata: ad.AnnData,
    batch_key: str = "Batch",
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    epsilon: float = 1e-6,
    source_layer: str | None = "raw",
    inplace: bool = True,
) -> ad.AnnData:
    """
    Robust Z-score normalization per batch using control wells.

    Formula:
        (X - median(controls)) / (MAD(controls) * 1.4826 + epsilon)
    """
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Missing '{batch_key}' in adata.obs")
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Missing '{treatment_key}' in adata.obs")

    target = adata if inplace else adata.copy()
    X = _matrix_from_layer(target, source_layer=source_layer)
    X_norm = np.array(X, dtype=np.float32, copy=True)

    batch_values = target.obs[batch_key].astype(str)
    treatment_values = target.obs[treatment_key].astype(str)

    for batch in pd.unique(batch_values):
        batch_mask = batch_values == batch
        control_mask = batch_mask & (treatment_values == control_value)

        if not np.any(control_mask):
            warnings.warn(
                f"Batch '{batch}' has no controls labeled '{control_value}'. Keeping raw values.",
                stacklevel=2,
            )
            continue

        controls = X[control_mask.to_numpy(), :]
        current = X[batch_mask.to_numpy(), :]

        med = np.nanmedian(controls, axis=0)
        mad = median_abs_deviation(controls, axis=0, scale="normal", nan_policy="omit")
        transformed = (current - med) / (mad + epsilon)
        X_norm[batch_mask.to_numpy(), :] = transformed.astype(np.float32, copy=False)

    if source_layer is None and "raw" not in target.layers:
        target.layers["raw"] = np.array(X, copy=True)

    target.X = X_norm
    target.uns.setdefault("cptools", {})
    target.uns["cptools"]["robust_zscore_norm"] = {
        "batch_key": batch_key,
        "treatment_key": treatment_key,
        "control_value": control_value,
        "epsilon": epsilon,
        "source_layer": source_layer,
    }
    return target


def blocklist_filter(
    adata: ad.AnnData,
    keywords: Sequence[str] = DEFAULT_BLOCKLIST_KEYWORDS,
    inplace: bool = True,
) -> ad.AnnData:
    """Drop features that contain known technical artifact keywords."""
    pattern = "|".join(map(str, keywords))
    keep_mask = ~adata.var_names.str.contains(pattern, regex=True)
    return _apply_var_mask(adata, np.asarray(keep_mask, dtype=bool), inplace=inplace)


def drop_nan_features(
    adata: ad.AnnData,
    source_layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData:
    """Drop features with non-finite values in any well."""
    X = _matrix_from_layer(adata, source_layer=source_layer)
    keep_mask = np.isfinite(X).all(axis=0)
    return _apply_var_mask(adata, keep_mask, inplace=inplace)


def variance_filter(
    adata: ad.AnnData,
    threshold: float = 1e-2,
    source_layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData:
    """Drop near-constant features."""
    X = _matrix_from_layer(adata, source_layer=source_layer)
    variances = np.nanvar(X, axis=0)
    keep_mask = variances > threshold
    return _apply_var_mask(adata, keep_mask, inplace=inplace)


def correlation_filter(
    adata: ad.AnnData,
    threshold: float = 0.9,
    source_layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Remove highly correlated features, keeping the highest variance feature per cluster.
    """
    X = _matrix_from_layer(adata, source_layer=source_layer)
    if X.shape[1] <= 1:
        return adata if inplace else adata.copy()

    variances = np.nanvar(X, axis=0)
    order = np.argsort(-variances)

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=1.0, neginf=1.0)

    keep_mask = np.zeros(X.shape[1], dtype=bool)
    dropped = np.zeros(X.shape[1], dtype=bool)

    for idx in order:
        if dropped[idx]:
            continue
        keep_mask[idx] = True
        correlated = corr[idx, :] > threshold
        correlated[idx] = False
        dropped = dropped | correlated

    return _apply_var_mask(adata, keep_mask, inplace=inplace)


def snr_feature_selection(
    adata: ad.AnnData,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    keep_top_fraction: float = 0.2,
    quantile_threshold: float | None = None,
    min_replicates: int = 2,
    source_layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Select features by replicate signal-to-noise ratio (SNR).

    Signal = variance of treatment means (excluding controls).
    Noise = mean within-treatment variance (including controls).
    """
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Missing '{treatment_key}' in adata.obs")
    if not (0 < keep_top_fraction <= 1):
        raise ValueError("keep_top_fraction must be in (0, 1].")

    X = _matrix_from_layer(adata, source_layer=source_layer)
    feature_names = adata.var_names.to_list()

    df = pd.DataFrame(X, columns=feature_names, index=adata.obs_names)
    df[treatment_key] = adata.obs[treatment_key].astype(str).values

    group_sizes = df.groupby(treatment_key).size()
    replicated_groups = group_sizes[group_sizes >= min_replicates].index
    if len(replicated_groups) == 0:
        raise ValueError(f"No treatment groups with at least {min_replicates} replicates.")

    df = df[df[treatment_key].isin(replicated_groups)].copy()
    non_controls = df[df[treatment_key] != control_value]
    if non_controls.empty:
        raise ValueError("No non-control groups available for SNR calculation.")

    signal = non_controls.groupby(treatment_key)[feature_names].mean().var(axis=0, ddof=1).fillna(0.0)
    noise = df.groupby(treatment_key)[feature_names].var(ddof=1).mean(axis=0).fillna(0.0)
    snr = signal / (noise + 1e-9)

    if quantile_threshold is not None:
        if not (0 <= quantile_threshold <= 1):
            raise ValueError("quantile_threshold must be in [0, 1].")
        cutoff = float(snr.quantile(quantile_threshold))
        selected = snr[snr > cutoff].index.tolist()
    else:
        n_keep = max(1, int(np.ceil(keep_top_fraction * len(snr))))
        selected = snr.sort_values(ascending=False).head(n_keep).index.tolist()

    if len(selected) == 0:
        selected = snr.sort_values(ascending=False).head(1).index.tolist()

    selected_set = set(selected)
    keep_mask = np.array([name in selected_set for name in adata.var_names], dtype=bool)
    target = adata if inplace else adata.copy()
    target.var["replicate_snr"] = snr.reindex(target.var_names).astype(np.float32)
    target.var["highly_variable"] = keep_mask
    target.var["highly_variable_rank"] = (
        target.var["replicate_snr"]
        .rank(method="first", ascending=False)
        .astype(np.float32)
    )
    target.uns.setdefault("cptools", {})
    target.uns["cptools"]["snr_feature_selection"] = {
        "treatment_key": treatment_key,
        "control_value": control_value,
        "keep_top_fraction": keep_top_fraction,
        "quantile_threshold": quantile_threshold,
        "min_replicates": min_replicates,
        "subset": subset,
    }

    if subset:
        return _apply_var_mask(target, keep_mask, inplace=True)
    return target


def zca_whiten(
    adata: ad.AnnData,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    epsilon: float = 1e-5,
    source_layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData:
    """Apply control-based regularized ZCA whitening."""
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Missing '{treatment_key}' in adata.obs")

    target = adata if inplace else adata.copy()
    X = _matrix_from_layer(target, source_layer=source_layer)
    treatment = target.obs[treatment_key].astype(str)
    control_mask = treatment == control_value

    n_controls = int(np.sum(control_mask))
    if n_controls < 2:
        raise ValueError("At least two control wells are required for ZCA whitening.")

    X_controls = X[control_mask.to_numpy(), :]
    control_mean = np.nanmean(X_controls, axis=0)
    X_centered = X - control_mean
    X_controls_centered = X_controls - control_mean

    sigma = (X_controls_centered.T @ X_controls_centered) / (n_controls - 1)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    inv_sqrt = 1.0 / np.sqrt(np.clip(eigvals, 0, None) + epsilon)
    whitening_matrix = (eigvecs * inv_sqrt) @ eigvecs.T

    target.X = (X_centered @ whitening_matrix).astype(np.float32, copy=False)
    target.uns.setdefault("cptools", {})
    target.uns["cptools"]["zca_whiten"] = {
        "treatment_key": treatment_key,
        "control_value": control_value,
        "epsilon": epsilon,
    }
    return target


def funnel(
    adata: ad.AnnData,
    batch_key: str = "Batch",
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    variance_threshold: float = 1e-2,
    corr_threshold: float = 0.9,
    snr_keep_top_fraction: float = 0.2,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Clean -> Normalize -> De-correlate -> Quality Control pipeline.
    """
    target = adata if inplace else adata.copy()
    robust_zscore_norm(
        target,
        batch_key=batch_key,
        treatment_key=treatment_key,
        control_value=control_value,
        source_layer="raw" if "raw" in target.layers else None,
        inplace=True,
    )
    blocklist_filter(target, inplace=True)
    drop_nan_features(target, inplace=True)
    variance_filter(target, threshold=variance_threshold, inplace=True)
    correlation_filter(target, threshold=corr_threshold, inplace=True)
    snr_feature_selection(
        target,
        treatment_key=treatment_key,
        control_value=control_value,
        keep_top_fraction=snr_keep_top_fraction,
        subset=False,
        inplace=True,
    )
    return target


# Aliases mirroring notebook naming.
robust_zscore_normalize = robust_zscore_norm
feature_selection_by_replicates = snr_feature_selection
