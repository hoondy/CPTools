from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import warnings
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from ._utils import clean_feature_name, to_dense_matrix


DEFAULT_BLOCKLIST_KEY = "default"


def _load_blocklist_yaml(
    path: str | Path | None = None,
    key: str = DEFAULT_BLOCKLIST_KEY,
) -> list[str]:
    """Load a simple YAML blocklist containing a top-level list under `key`."""
    if path is None:
        blocklist_path = files("CPTools").joinpath("default_blocklists.yaml")
        lines = blocklist_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()

    in_section = False
    values: list[str] = []
    section_header = f"{key}:"
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith((" ", "\t")):
            in_section = line == section_header
            continue
        if in_section and line.lstrip().startswith("- "):
            values.append(line.lstrip()[2:].strip().strip("\"'"))

    if not values:
        raise ValueError(f"No blocklist entries found under YAML key '{key}'.")
    return values


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


def _apply_obs_mask(adata: ad.AnnData, mask: np.ndarray, inplace: bool) -> ad.AnnData:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if inplace:
        adata._inplace_subset_obs(mask)
        return adata
    return adata[mask, :].copy()


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
    keywords: Sequence[str] | None = None,
    blocklist_path: str | Path | None = None,
    blocklist_key: str = DEFAULT_BLOCKLIST_KEY,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Mark/drop known noisy CellProfiler features.

    By default (``subset=False``), this only writes ``adata.var["pass_blocklist"]``.
    """
    target = adata if inplace else adata.copy()

    if keywords is not None:
        keyword_list = [str(keyword) for keyword in keywords]
        if len(keyword_list) == 0:
            raise ValueError("keywords cannot be empty.")
        blocklisted_mask = np.zeros(target.n_vars, dtype=bool)
        for keyword in keyword_list:
            blocklisted_mask |= np.asarray(
                target.var_names.str.contains(keyword, regex=False),
                dtype=bool,
            )
        keep_mask = ~blocklisted_mask
        blocklisted = target.var_names[~keep_mask].astype(str).tolist()
    else:
        blocklist = set(_load_blocklist_yaml(path=blocklist_path, key=blocklist_key))
        candidate_columns = [pd.Series(target.var_names.astype(str), index=target.var_names)]
        if "feature" in target.var.columns:
            raw_features = target.var["feature"].astype(str)
            candidate_columns.append(raw_features)
            candidate_columns.append(raw_features.map(clean_feature_name))

        blocklisted_mask = np.zeros(target.n_vars, dtype=bool)
        for candidates in candidate_columns:
            blocklisted_mask |= candidates.isin(blocklist).to_numpy(dtype=bool)

        keep_mask = ~blocklisted_mask
        blocklisted = target.var_names[blocklisted_mask].astype(str).tolist()

    keep_mask = np.asarray(keep_mask, dtype=bool)
    target.var["pass_blocklist"] = keep_mask
    target.var["blocklisted"] = ~keep_mask
    target.uns.setdefault("cptools", {})
    target.uns["cptools"]["blocklist_filter"] = {
        "source": "keywords" if keywords is not None else str(blocklist_path or "package:default_blocklists.yaml"),
        "blocklist_key": blocklist_key,
        "n_blocklisted": int((~keep_mask).sum()),
        "blocklisted_features": blocklisted,
    }
    if subset:
        return _apply_var_mask(target, keep_mask, inplace=True)
    return target


def nan_filter(
    adata: ad.AnnData,
    source_layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Mark/drop features with non-finite values in any well.

    By default (``subset=False``), this only writes ``adata.var["pass_non_nan"]``.
    """
    target = adata if inplace else adata.copy()
    X = _matrix_from_layer(target, source_layer=source_layer)
    keep_mask = np.isfinite(X).all(axis=0)
    target.var["pass_non_nan"] = keep_mask
    if subset:
        return _apply_var_mask(target, keep_mask, inplace=True)
    return target


def nan_obs_filter(
    adata: ad.AnnData,
    max_nan_fraction: float = 0.2,
    source_layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Mark/drop observations with too many non-finite feature values.

    By default (``subset=False``), this writes ``adata.obs["pass_non_nan"]``.
    """
    if not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be in [0, 1].")

    target = adata if inplace else adata.copy()
    X = _matrix_from_layer(target, source_layer=source_layer)
    nan_fraction = 1.0 - np.isfinite(X).mean(axis=1)
    keep_mask = nan_fraction <= max_nan_fraction
    target.obs["pass_non_nan"] = keep_mask
    target.obs["nan_fraction"] = nan_fraction.astype(np.float32, copy=False)
    if subset:
        return _apply_obs_mask(target, keep_mask, inplace=True)
    return target


def variance_filter(
    adata: ad.AnnData,
    threshold: float = 1e-2,
    source_layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Mark/drop near-constant features.

    By default (``subset=False``), this only writes ``adata.var["pass_variance"]``.
    """
    target = adata if inplace else adata.copy()
    X = _matrix_from_layer(target, source_layer=source_layer)
    variances = np.nanvar(X, axis=0)
    keep_mask = variances > threshold
    target.var["pass_variance"] = keep_mask
    if subset:
        return _apply_var_mask(target, keep_mask, inplace=True)
    return target


def correlation_filter(
    adata: ad.AnnData,
    threshold: float = 0.9,
    source_layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Mark/drop highly correlated features, keeping highest-variance representatives.

    By default (``subset=False``), this only writes ``adata.var["pass_correlation"]``.
    """
    target = adata if inplace else adata.copy()
    X = _matrix_from_layer(target, source_layer=source_layer)
    if X.shape[1] <= 1:
        target.var["pass_correlation"] = np.ones(target.n_vars, dtype=bool)
        return target

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

    target.var["pass_correlation"] = keep_mask
    if subset:
        return _apply_var_mask(target, keep_mask, inplace=True)
    return target


def snr_feature_selection(
    adata: ad.AnnData,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    keep_top_fraction: float = 0.2,
    quantile_threshold: float | None = None,
    min_replicates: int = 2,
    noise_aggregation: str = "pooled",
    source_layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Select features by replicate signal-to-noise ratio (SNR).
    Using a SNR approach, which is computationally efficient and effectively a univariate F-statistic,
    select features that are consistent within a treatment group (low noise),
    while vary between different treatments (high signal).

    Signal = variance of treatment means (excluding controls).
    Noise = within-treatment (replicates) variance (including controls),
    aggregated across groups via `noise_aggregation`:
      - ``"pooled"``: weighted by (n_group - 1), preferred for unequal replicate counts
      - ``"mean"``: simple mean of group variances
      - ``"median"``: robust median of group variances
    """
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Missing '{treatment_key}' in adata.obs")
    if not (0 < keep_top_fraction <= 1):
        raise ValueError("keep_top_fraction must be in (0, 1].")
    if noise_aggregation not in {"pooled", "mean", "median"}:
        raise ValueError("noise_aggregation must be one of: 'pooled', 'mean', 'median'.")

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
    n_non_control_groups = int(non_controls[treatment_key].nunique())
    if n_non_control_groups < 2:
        raise ValueError(
            "SNR requires at least 2 non-control treatment groups after replicate filtering "
            f"(min_replicates={min_replicates}). Found {n_non_control_groups}."
        )

    # Calculate Signal (Variance Between Treatment Means)
    # Group by Treatment -> Calculate Mean Vector for each Drug -> Calculate Variance of those Means
    signal = non_controls.groupby(treatment_key)[feature_names].mean().var(axis=0, ddof=1).fillna(0.0)
    # Calculate Noise (Within-group replicate variance)
    # Group by Treatment -> variance vector per group -> aggregate across groups.
    var_by_group = df.groupby(treatment_key)[feature_names].var(ddof=1)
    if noise_aggregation == "pooled":
        group_sizes = df.groupby(treatment_key).size().reindex(var_by_group.index)
        weights = (group_sizes - 1).clip(lower=1).astype(np.float64)
        weighted_sum = var_by_group.mul(weights, axis=0).sum(axis=0)
        noise = (weighted_sum / float(weights.sum())).fillna(0.0)
    elif noise_aggregation == "median":
        noise = var_by_group.median(axis=0).fillna(0.0)
    else:
        noise = var_by_group.mean(axis=0).fillna(0.0)
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
        "noise_aggregation": noise_aggregation,
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
    source_layer: str | None = None,
    obs_nan_threshold: float | None = 0.2,
    variance_threshold: float | None = 1e-2,
    corr_threshold: float | None = 0.9,
    snr_threshold: float | None = 0.8,
    snr_keep_top_fraction: float | None = None,
    subset: bool = False,
    verbose: bool = True,
    inplace: bool = True,
) -> ad.AnnData:
    """
    A series of filters and feature selections.

    Filtering is calculated from ``adata.X`` by default. Pass ``source_layer``
    to calculate filters from a specific layer without replacing ``adata.X``.

    By default, this function **does not remove features**. It marks selected
    features in ``adata.var["highly_variable"]``.

    Set ``subset=True`` to physically subset features to the final selected set.
    """
    if variance_threshold is not None and not (0 <= variance_threshold <= 1):
        raise ValueError("variance_threshold must be in [0, 1] and is interpreted as a quantile.")
    if corr_threshold is not None and not (-1 <= corr_threshold <= 1):
        raise ValueError("corr_threshold must be in [-1, 1].")
    if obs_nan_threshold is not None and not (0 <= obs_nan_threshold <= 1):
        raise ValueError("obs_nan_threshold must be in [0, 1].")

    if snr_keep_top_fraction is not None:
        if not (0 < snr_keep_top_fraction <= 1):
            raise ValueError("snr_keep_top_fraction must be in (0, 1].")
        if snr_threshold not in (None, 0.8):
            raise ValueError(
                "Pass only one of 'snr_threshold' or deprecated 'snr_keep_top_fraction'."
            )
        warnings.warn(
            "'snr_keep_top_fraction' is deprecated; use 'snr_threshold' "
            "(fraction to exclude from bottom).",
            DeprecationWarning,
            stacklevel=2,
        )
        snr_threshold = 1.0 - snr_keep_top_fraction

    if snr_threshold is not None and not (0 <= snr_threshold <= 1):
        raise ValueError("snr_threshold must be in [0, 1].")

    target = adata if inplace else adata.copy()
    source_matrix = _matrix_from_layer(target, source_layer=source_layer)

    # Run feature-reduction steps on a working copy so default behavior can
    # annotate HV features without mutating target feature space.
    working = target.copy()
    working.X = source_matrix.astype(np.float32, copy=True)

    n0 = working.n_vars
    obs_nan_fraction_by_obs = pd.Series(np.nan, index=working.obs_names, dtype=np.float32)
    if verbose:
        source_label = source_layer if source_layer is not None else "X"
        print(f"[funnel] starting with {working.n_obs} observations and {n0} features from {source_label}")

    before = working.n_vars
    blocklist_filter(working, subset=True, inplace=True)
    if verbose:
        filtered = before - working.n_vars
        print(f"[funnel] blocklist_filter: filtered {filtered}, remaining {working.n_vars}")

    if obs_nan_threshold is None:
        working.obs["pass_non_nan"] = np.ones(working.n_obs, dtype=bool)
        working.obs["nan_fraction"] = np.nan
        if verbose:
            print(
                "[funnel] nan_obs_filter: skipped (obs_nan_threshold=None); "
                f"remaining {working.n_obs} observations"
            )
    else:
        before_obs = working.n_obs
        nan_obs_filter(
            working,
            max_nan_fraction=obs_nan_threshold,
            subset=False,
            inplace=True,
        )
        obs_nan_fraction_by_obs = working.obs["nan_fraction"].copy()
        working._inplace_subset_obs(working.obs["pass_non_nan"].to_numpy(dtype=bool))
        if verbose:
            filtered_obs = before_obs - working.n_obs
            print(
                "[funnel] nan_obs_filter: filtered "
                f"{filtered_obs}, remaining {working.n_obs} observations"
            )

    before = working.n_vars
    nan_filter(working, subset=True, inplace=True)
    if verbose:
        filtered = before - working.n_vars
        print(f"[funnel] nan_filter: filtered {filtered}, remaining {working.n_vars}")

    variance_cutoff = None
    if variance_threshold is None:
        if verbose:
            print(
                "[funnel] variance_filter: skipped (variance_threshold=None); "
                f"remaining {working.n_vars}"
            )
    else:
        before = working.n_vars
        var_values = np.nanvar(_matrix_from_layer(working, source_layer=None), axis=0)
        variance_cutoff = float(np.nanquantile(var_values, variance_threshold))
        variance_filter(working, threshold=variance_cutoff, subset=True, inplace=True)
        if verbose:
            filtered = before - working.n_vars
            print(f"[funnel] variance_filter: filtered {filtered}, remaining {working.n_vars}")

    if corr_threshold is None:
        if verbose:
            print(
                "[funnel] correlation_filter: skipped (corr_threshold=None); "
                f"remaining {working.n_vars}"
            )
    else:
        before = working.n_vars
        correlation_filter(working, threshold=corr_threshold, subset=True, inplace=True)
        if verbose:
            filtered = before - working.n_vars
            print(f"[funnel] correlation_filter: filtered {filtered}, remaining {working.n_vars}")

    if snr_threshold is None:
        working.var["highly_variable"] = np.ones(working.n_vars, dtype=bool)
        working.var["replicate_snr"] = np.nan
        working.var["highly_variable_rank"] = np.nan
        if verbose:
            print(
                "[funnel] snr_feature_selection: skipped (snr_threshold=None); "
                f"selected {working.n_vars} highly_variable features from {working.n_vars}"
            )
    else:
        snr_feature_selection(
            working,
            treatment_key=treatment_key,
            control_value=control_value,
            quantile_threshold=snr_threshold,
            subset=False,
            inplace=True,
        )
        if verbose:
            hv_in_working = int(working.var["highly_variable"].sum())
            print(
                "[funnel] snr_feature_selection: selected "
                f"{hv_in_working} highly_variable features from {working.n_vars}"
            )

    hv_features = set(working.var_names[working.var["highly_variable"].astype(bool)])
    prefilter_features = set(working.var_names)
    prefilter_obs = set(working.obs_names)

    target.obs["pass_funnel_obs_prefilter"] = np.array(
        [name in prefilter_obs for name in target.obs_names],
        dtype=bool,
    )
    target.obs["funnel_nan_fraction"] = obs_nan_fraction_by_obs.reindex(target.obs_names).astype(
        np.float32
    )

    target.var["pass_funnel_prefilter"] = np.array(
        [name in prefilter_features for name in target.var_names],
        dtype=bool,
    )
    target.var["highly_variable"] = np.array(
        [name in hv_features for name in target.var_names],
        dtype=bool,
    )

    target.var["replicate_snr"] = np.nan
    if "replicate_snr" in working.var.columns:
        target.var.loc[working.var_names, "replicate_snr"] = working.var["replicate_snr"].astype(
            np.float32
        )

    target.var["highly_variable_rank"] = np.nan
    if "highly_variable_rank" in working.var.columns:
        target.var.loc[working.var_names, "highly_variable_rank"] = working.var[
            "highly_variable_rank"
        ].astype(np.float32)

    target.uns.setdefault("cptools", {})
    target.uns["cptools"]["funnel"] = {
        "batch_key": batch_key,
        "treatment_key": treatment_key,
        "control_value": control_value,
        "source_layer": source_layer,
        "obs_nan_threshold": obs_nan_threshold,
        "variance_threshold": variance_threshold,
        "variance_cutoff": variance_cutoff,
        "corr_threshold": corr_threshold,
        "snr_threshold": snr_threshold,
        "subset": subset,
        "verbose": verbose,
    }

    if verbose:
        obs_prefilter_n = int(target.obs["pass_funnel_obs_prefilter"].sum())
        prefilter_n = int(target.var["pass_funnel_prefilter"].sum())
        hv_n = int(target.var["highly_variable"].sum())
        print(
            "[funnel] annotation on original matrix: "
            f"pass_funnel_obs_prefilter={obs_prefilter_n}/{target.n_obs}, "
            f"pass_funnel_prefilter={prefilter_n}/{target.n_vars}, "
            f"highly_variable={hv_n}/{target.n_vars}"
        )

    if subset:
        if verbose:
            print(
                "[funnel] subset=True: subsetting to "
                f"{int(target.obs['pass_funnel_obs_prefilter'].sum())} observations and "
                f"{int(target.var['highly_variable'].sum())} features"
            )
        target._inplace_subset_obs(target.obs["pass_funnel_obs_prefilter"].to_numpy(dtype=bool))
        target._inplace_subset_var(target.var["highly_variable"].to_numpy(dtype=bool))
    return target


# Aliases mirroring notebook naming.
robust_zscore_normalize = robust_zscore_norm
feature_selection_by_replicates = snr_feature_selection
# Backward-compatible alias.
drop_nan_features = nan_filter
