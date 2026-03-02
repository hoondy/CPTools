from __future__ import annotations

from typing import Any, Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from ._utils import to_dense_matrix


def scatter(
    adata: ad.AnnData,
    color: str | Sequence[str] | None = None,
    use_rep: str = "X_umap",
    width: int = 800,
    height: int = 800,
    wspace: float = 0.1,
    legend: bool = True,
    title: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> go.Figure | None:
    """
    Plot a 2D embedding from ``adata.obsm`` using Plotly.

    Examples
    -------
    ``cpt.tl.scatter(adata, color="MOA", use_rep="X_umap")``
    ``cpt.tl.scatter(adata, color=["Batch", "DMSO"], use_rep="X_umap", wspace=0.4)``
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"Embedding '{use_rep}' not found in adata.obsm.")

    embedding = np.asarray(adata.obsm[use_rep])
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{use_rep}' must have shape (n_obs, >=2).")

    frame = pd.DataFrame(
        {
            f"{use_rep}_1": embedding[:, 0],
            f"{use_rep}_2": embedding[:, 1],
        },
        index=adata.obs_names,
    )

    if color is None:
        color_list: list[str | None] = [None]
    elif isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)
        if len(color_list) == 0:
            raise ValueError("If 'color' is a sequence, it cannot be empty.")

    for color_col in color_list:
        if color_col is not None and color_col not in adata.obs.columns:
            raise KeyError(f"Column '{color_col}' not found in adata.obs.")

    x_col = f"{use_rep}_1"
    y_col = f"{use_rep}_2"

    if len(color_list) == 1:
        color_col = color_list[0]
        single = frame.copy()
        if color_col is not None:
            single[color_col] = adata.obs[color_col].astype(str).values
        fig = px.scatter(
            single,
            x=x_col,
            y=y_col,
            color=color_col,
            width=width,
            height=height,
            title=title or f"{use_rep} scatter",
            **kwargs,
        )
        fig.update_layout(legend_title_text=color_col or "")
    else:
        ncols = len(color_list)
        max_spacing = 0.98 / max(1, ncols - 1)
        spacing = min(max(wspace, 0.0), max_spacing)
        subplot_titles = [c if c is not None else "All cells" for c in color_list]
        fig = make_subplots(
            rows=1,
            cols=ncols,
            subplot_titles=subplot_titles,
            horizontal_spacing=spacing,
        )

        for idx, color_col in enumerate(color_list, start=1):
            panel = frame.copy()
            if color_col is not None:
                panel[color_col] = adata.obs[color_col].astype(str).values

            panel_fig = px.scatter(
                panel,
                x=x_col,
                y=y_col,
                color=color_col,
                **kwargs,
            )
            for trace in panel_fig.data:
                if color_col is not None:
                    base_name = trace.name if trace.name else "value"
                    trace.name = f"{color_col}: {base_name}"
                    trace.legendgroup = trace.name
                fig.add_trace(trace, row=1, col=idx)
            fig.update_xaxes(title_text=x_col, row=1, col=idx)
            fig.update_yaxes(title_text=y_col, row=1, col=idx)

        fig.update_layout(
            width=width,
            height=height,
            title=title or f"{use_rep} scatter",
        )

    fig.update_layout(showlegend=legend)

    if show:
        fig.show()
        return None
    return fig


def _get_data_matrix(adata: ad.AnnData, layer: str | None) -> np.ndarray:
    if layer is None:
        return to_dense_matrix(adata.X)
    if layer not in adata.layers:
        warnings.warn(
            f"Layer '{layer}' not found. Falling back to adata.X.",
            stacklevel=2,
        )
        return to_dense_matrix(adata.X)
    return to_dense_matrix(adata.layers[layer])


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _normalize_treatments(treatments: str | Sequence[str] | None) -> list[str] | None:
    if treatments is None:
        return None
    if isinstance(treatments, str):
        parsed = [treatments]
    else:
        parsed = list(treatments)
    parsed = [str(t) for t in parsed]
    parsed = list(dict.fromkeys(parsed))
    if len(parsed) == 0:
        raise ValueError("treatments cannot be empty.")
    return parsed


def treatment_vectors(
    adata: ad.AnnData,
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    batch_key: str = "Batch",
    layer: str | None = "normalized",
    treatments: str | Sequence[str] | None = None,
    use_highly_variable: bool = False,
) -> pd.DataFrame:
    """
    Compute batch-matched control->treatment feature vectors.

    For each treatment and each batch where that treatment appears:
      vector(batch, treatment) = mean(treatment wells) - mean(control wells)
    Final treatment vector is a treatment-well-count weighted mean across batches.
    """
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Column '{treatment_key}' not found in adata.obs.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in adata.obs.")

    if use_highly_variable:
        if "highly_variable" not in adata.var.columns:
            raise KeyError("Column 'highly_variable' not found in adata.var.")
        hv_mask = adata.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
        if not np.any(hv_mask):
            raise ValueError("No features marked as highly_variable.")
        feature_names = adata.var_names[hv_mask].tolist()
    else:
        feature_names = adata.var_names.tolist()

    requested = _normalize_treatments(treatments)

    X = _get_data_matrix(adata, layer=layer)
    frame = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
    frame = frame.loc[:, feature_names].copy()
    frame[treatment_key] = adata.obs[treatment_key].astype(str).values
    frame[batch_key] = adata.obs[batch_key].astype(str).values

    all_treatments = pd.unique(frame[treatment_key]).tolist()
    if requested is None:
        target_treatments = [t for t in all_treatments if t != str(control_value)]
    else:
        missing = [t for t in requested if t not in all_treatments]
        if missing:
            raise ValueError(
                f"Requested treatment(s) not found in '{treatment_key}': {missing}"
            )
        target_treatments = [t for t in requested if t != str(control_value)]
        if len(target_treatments) == 0:
            raise ValueError("No non-control treatments requested.")

    vectors: list[np.ndarray] = []
    index: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for trt in target_treatments:
        trt_mask = frame[treatment_key] == trt
        trt_batches = pd.unique(frame.loc[trt_mask, batch_key]).tolist()

        batch_vectors: list[np.ndarray] = []
        weights: list[float] = []
        used_batches: list[str] = []

        for batch in trt_batches:
            batch_trt_mask = trt_mask & (frame[batch_key] == batch)
            batch_ctrl_mask = (frame[treatment_key] == str(control_value)) & (frame[batch_key] == batch)
            if not np.any(batch_ctrl_mask):
                continue

            trt_values = frame.loc[batch_trt_mask, feature_names]
            ctrl_values = frame.loc[batch_ctrl_mask, feature_names]
            delta = trt_values.mean(axis=0) - ctrl_values.mean(axis=0)
            batch_vectors.append(delta.to_numpy(dtype=np.float64))
            weights.append(float(trt_values.shape[0]))
            used_batches.append(str(batch))

        if len(batch_vectors) == 0:
            skipped.append(trt)
            continue

        weights_arr = np.asarray(weights, dtype=np.float64)
        trt_vector = np.average(np.vstack(batch_vectors), axis=0, weights=weights_arr)
        vectors.append(trt_vector.astype(np.float32, copy=False))
        index.append(str(trt))
        metadata_rows.append(
            {
                "treatment": str(trt),
                "n_batches_used": int(len(used_batches)),
                "batches_used": used_batches,
                "n_wells_used": int(weights_arr.sum()),
            }
        )

    if len(vectors) == 0:
        raise ValueError(
            "No treatment vectors could be computed. Check matched controls per batch."
        )

    if skipped:
        if requested is None:
            warnings.warn(
                "Skipped treatments with no matched control wells in their batches: "
                f"{sorted(skipped)}",
                stacklevel=2,
            )
        else:
            raise ValueError(
                "Matched control wells were not found for requested treatment(s): "
                f"{sorted(skipped)}"
            )

    out = pd.DataFrame(np.vstack(vectors), index=index, columns=feature_names)
    out.index.name = "treatment"
    metadata_df = pd.DataFrame(metadata_rows).set_index("treatment")
    # Keep attrs values simple/serializable; storing DataFrames in attrs can
    # break pandas repr/concat due to ambiguous DataFrame truth comparisons.
    out.attrs["metadata"] = metadata_df.to_dict(orient="index")
    out.attrs["params"] = {
        "treatment_key": treatment_key,
        "control_value": control_value,
        "batch_key": batch_key,
        "layer": layer,
        "use_highly_variable": use_highly_variable,
    }
    return out


def rank_treatment_correlations(
    vectors: pd.DataFrame,
    treatment: str,
    method: str = "spearman",
    top_n: int = 10,
    bottom_n: int = 0,
    legend: bool = True,
    show: bool = True,
) -> pd.DataFrame:
    """
    Rank correlations between one treatment vector and all other treatment vectors.

    Parameters
    ----------
    vectors
        Output table from ``cpt.tl.treatment_vectors`` (rows=treatments, cols=features).
    treatment
        Treatment row name to compare against all others.
    method
        Correlation method: ``"spearman"`` or ``"pearson"``.
    """
    if not isinstance(vectors, pd.DataFrame):
        raise TypeError("vectors must be a pandas DataFrame from cpt.tl.treatment_vectors.")
    if vectors.empty:
        raise ValueError("vectors is empty.")
    if top_n < 0:
        raise ValueError("top_n must be >= 0.")
    if bottom_n < 0:
        raise ValueError("bottom_n must be >= 0.")
    if top_n == 0 and bottom_n == 0:
        raise ValueError("At least one of top_n or bottom_n must be > 0.")
    method = method.lower()
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be one of: 'spearman', 'pearson'.")

    treatment = str(treatment)
    if treatment not in vectors.index:
        raise ValueError(
            f"Treatment '{treatment}' is not available in vectors index. "
            f"Available examples: {vectors.index[:10].tolist()}"
        )
    if vectors.shape[0] < 2:
        raise ValueError(
            "Need at least 2 treatment vectors to compute pairwise correlations."
        )
    numeric_vectors = vectors.apply(pd.to_numeric, errors="coerce")
    if numeric_vectors.isna().all(axis=1).any():
        bad_rows = numeric_vectors.index[numeric_vectors.isna().all(axis=1)].tolist()
        raise ValueError(
            "One or more treatment vectors are entirely non-numeric/NaN. "
            f"Example rows: {bad_rows[:5]}"
        )

    target = numeric_vectors.loc[treatment].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for other, row in numeric_vectors.iterrows():
        if other == treatment:
            continue
        row_arr = row.to_numpy(dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(row_arr)
        if valid.sum() < 2:
            corr = np.nan
        elif method == "spearman":
            corr = stats.spearmanr(target[valid], row_arr[valid], nan_policy="omit").statistic
        else:
            corr = stats.pearsonr(target[valid], row_arr[valid]).statistic
        if not np.isfinite(corr):
            corr = 0.0
        rows.append({"treatment": str(other), "correlation": float(corr)})

    ranked = pd.DataFrame(rows).sort_values("correlation", ascending=False, kind="stable")
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    ranked = ranked[["rank", "treatment", "correlation"]]

    display_parts: list[pd.DataFrame] = []
    if top_n > 0:
        top_df = ranked.head(top_n).copy()
        top_df["section"] = f"Top {top_n}"
        display_parts.append(top_df)
    if bottom_n > 0:
        bottom_df = ranked.tail(bottom_n).sort_values("correlation", ascending=True).copy()
        bottom_df["section"] = f"Bottom {bottom_n}"
        display_parts.append(bottom_df)
    display_df = pd.concat(display_parts, axis=0).drop_duplicates(subset=["treatment"], keep="first")
    if display_df.empty:
        raise ValueError("No correlation results available to plot.")
    display_df["label"] = display_df["treatment"]

    corr_label = "Spearman" if method == "spearman" else "Pearson"
    order = display_df.sort_values("correlation", ascending=True)["label"].tolist()
    fig = px.bar(
        display_df,
        x="correlation",
        y="label",
        orientation="h",
        color="correlation",
        color_continuous_scale="RdBu",
        range_color=(-1, 1),
        category_orders={"label": order},
        hover_data={"rank": True, "treatment": True, "section": True, "label": False},
        title=f"Treatment Vector {corr_label} Rank: {treatment}",
        width=950,
        height=max(450, 30 * len(display_df) + 180),
    )
    fig.update_layout(
        xaxis_title=f"{corr_label} correlation",
        yaxis_title="Treatment",
        showlegend=legend,
        coloraxis_showscale=legend,
    )

    if show:
        fig.show()
    ranked.attrs["displayed"] = display_df.loc[:, ["section", "rank", "treatment", "correlation"]]
    ranked.attrs["method"] = method
    ranked.attrs["query_treatment"] = treatment
    return ranked


def umap_treatment_arrows(
    adata: ad.AnnData,
    treatment: str | Sequence[str],
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    batch_key: str = "Batch",
    use_rep: str = "X_umap",
    legend: bool = True,
    width: int = 1000,
    height: int = 800,
    show: bool = True,
) -> go.Figure | None:
    """
    Visualize control->treatment arrows on a 2D embedding (e.g., UMAP).
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"Embedding '{use_rep}' not found in adata.obsm.")
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Column '{treatment_key}' not found in adata.obs.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in adata.obs.")

    embedding = np.asarray(adata.obsm[use_rep])
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{use_rep}' must have shape (n_obs, >=2).")

    selected = _normalize_treatments(treatment)
    assert selected is not None
    selected = [t for t in selected if t != str(control_value)]
    if len(selected) == 0:
        raise ValueError("Provide at least one non-control treatment to draw arrows.")

    x_col = f"{use_rep}_1"
    y_col = f"{use_rep}_2"
    frame = pd.DataFrame(
        {
            x_col: embedding[:, 0],
            y_col: embedding[:, 1],
            treatment_key: adata.obs[treatment_key].astype(str).values,
            batch_key: adata.obs[batch_key].astype(str).values,
        },
        index=adata.obs_names,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=frame[x_col],
            y=frame[y_col],
            mode="markers",
            name="All wells",
            marker={"size": 5, "color": "lightgray", "opacity": 0.3},
            hoverinfo="skip",
        )
    )

    ctrl_mask = frame[treatment_key] == str(control_value)
    if np.any(ctrl_mask):
        fig.add_trace(
            go.Scattergl(
                x=frame.loc[ctrl_mask, x_col],
                y=frame.loc[ctrl_mask, y_col],
                mode="markers",
                name=str(control_value),
                marker={"size": 5, "color": "#636EFA", "opacity": 0.4},
                hovertemplate=f"{treatment_key}={control_value}<br>{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
            )
        )

    available_treatments = set(pd.unique(frame[treatment_key]).tolist())
    missing_requested = [t for t in selected if t not in available_treatments]
    if missing_requested:
        raise ValueError(
            f"Requested treatment(s) not found in '{treatment_key}': {missing_requested}"
        )

    palette = px.colors.qualitative.Plotly
    for idx, trt in enumerate(selected):
        color = palette[idx % len(palette)]
        trt_mask = frame[treatment_key] == trt
        fig.add_trace(
            go.Scattergl(
                x=frame.loc[trt_mask, x_col],
                y=frame.loc[trt_mask, y_col],
                mode="markers",
                marker={"size": 6, "color": color, "opacity": 0.5},
                name=trt,
                legendgroup=trt,
                showlegend=False,
                hovertemplate=f"{treatment_key}={trt}<br>{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
            )
        )

        trt_batches = pd.unique(frame.loc[trt_mask, batch_key]).tolist()
        start_points: list[np.ndarray] = []
        end_points: list[np.ndarray] = []
        weights: list[float] = []
        missing_ctrl_batches: list[str] = []

        for batch in trt_batches:
            batch_trt_mask = trt_mask & (frame[batch_key] == batch)
            batch_ctrl_mask = ctrl_mask & (frame[batch_key] == batch)
            if not np.any(batch_ctrl_mask):
                missing_ctrl_batches.append(str(batch))
                continue

            start_points.append(
                frame.loc[batch_ctrl_mask, [x_col, y_col]].to_numpy(dtype=np.float64).mean(axis=0)
            )
            end_points.append(
                frame.loc[batch_trt_mask, [x_col, y_col]].to_numpy(dtype=np.float64).mean(axis=0)
            )
            weights.append(float(np.sum(batch_trt_mask)))

        if len(weights) == 0:
            raise ValueError(
                f"Treatment '{trt}' has no matched '{control_value}' controls in its batches."
            )

        if missing_ctrl_batches:
            warnings.warn(
                f"Treatment '{trt}' missing matched controls in batches {sorted(missing_ctrl_batches)}; "
                "using only batches with controls.",
                stacklevel=2,
            )

        w = np.asarray(weights, dtype=np.float64)
        start = np.average(np.vstack(start_points), axis=0, weights=w)
        end = np.average(np.vstack(end_points), axis=0, weights=w)

        fig.add_trace(
            go.Scatter(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                mode="lines",
                line={"color": color, "width": 3},
                name=f"{trt} vector",
                legendgroup=trt,
                showlegend=True,
                hovertemplate=(
                    f"{trt}: control->{trt}<br>"
                    f"{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>"
                ),
            )
        )
        fig.add_annotation(
            x=float(end[0]),
            y=float(end[1]),
            ax=float(start[0]),
            ay=float(start[1]),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor=color,
            text=str(trt),
            font={"color": color},
        )

    fig.update_layout(
        title=f"{use_rep}: control to treatment vectors",
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=width,
        height=height,
        showlegend=legend,
    )

    if show:
        fig.show()
        return None
    return fig


def visualize_drug_effect(
    adata: ad.AnnData,
    treatment: str | Sequence[str],
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    batch_key: str = "Batch",
    layer: str | None = "normalized",
    top_n: int = 5,
    qvalue_threshold: float = 0.05,
    effect_threshold: float = 0.0,
    legend: bool = True,
    show: bool = True,
) -> pd.DataFrame:
    """
    Generate volcano + boxplots for selected treatment(s) vs matched controls
    and return a top-features results table.

    When `treatment` is a list, all listed treatments are combined into one
    treatment group for the statistical test.
    Controls are batch-matched by default: only `control_value` wells from
    the same batches as the treatment wells are used.
    """
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Column '{treatment_key}' not found in adata.obs.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in adata.obs.")

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")
    if not (0 < qvalue_threshold <= 1):
        raise ValueError("qvalue_threshold must be in (0, 1].")
    if effect_threshold < 0:
        raise ValueError("effect_threshold must be >= 0.")

    treatments = [treatment] if isinstance(treatment, str) else list(treatment)
    if len(treatments) == 0:
        raise ValueError("treatment cannot be empty.")
    treatments = list(dict.fromkeys([str(t) for t in treatments]))
    treatment_label = ", ".join(treatments)

    X = _get_data_matrix(adata, layer=layer)
    frame = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
    frame[treatment_key] = adata.obs[treatment_key].astype(str).values
    frame[batch_key] = adata.obs[batch_key].astype(str).values

    treatment_mask = frame[treatment_key].isin(treatments)
    if not np.any(treatment_mask):
        raise ValueError(
            f"No wells found for treatment(s): {treatments} in column '{treatment_key}'."
        )
    treatment_batches = pd.unique(frame.loc[treatment_mask, batch_key]).tolist()

    control_mask = (frame[treatment_key] == str(control_value)) & (frame[batch_key].isin(treatment_batches))
    control_batches = set(pd.unique(frame.loc[control_mask, batch_key]).tolist())
    missing_control_batches = sorted(set(treatment_batches) - control_batches)
    if missing_control_batches:
        raise ValueError(
            "Matched controls are missing for treatment batches: "
            f"{missing_control_batches}. Expected '{control_value}' in '{treatment_key}'."
        )

    drug_df = frame[treatment_mask].drop(columns=[treatment_key, batch_key])
    ctrl_df = frame[control_mask].drop(columns=[treatment_key, batch_key])

    if len(drug_df) < 2 or len(ctrl_df) < 2:
        raise ValueError(
            f"Not enough replicates for treatment '{treatment_label}' vs control '{control_value}'. "
            f"Need at least 2 wells each."
        )

    effect_size = drug_df.mean(axis=0) - ctrl_df.mean(axis=0)
    _, p_vals = stats.ttest_ind(drug_df, ctrl_df, axis=0, equal_var=False, nan_policy="omit")
    p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)
    p_vals = np.clip(p_vals, np.finfo(np.float64).tiny, 1.0)
    q_vals = _bh_fdr(p_vals.astype(np.float64))
    q_vals = np.clip(q_vals, np.finfo(np.float64).tiny, 1.0)
    log_p = -np.log10(p_vals)
    log_q = -np.log10(q_vals)

    results = pd.DataFrame(
        {
            "feature": adata.var_names,
            "effect_size": effect_size.values,
            "p_value": p_vals,
            "adjusted_p_value": q_vals,
            "log_p_value": log_p,
            "log_q_value": log_q,
        }
    ).set_index("feature")

    top_hits = (
        results.assign(abs_effect=np.abs(results["effect_size"]))
        .sort_values(["adjusted_p_value", "abs_effect"], ascending=[True, False])
        .head(top_n)
        .drop(columns=["abs_effect"])
    )
    sig_mask = (results["adjusted_p_value"] < qvalue_threshold) & (
        results["effect_size"].abs() >= effect_threshold
    )
    sig_results = results[sig_mask]

    # Volcano plot
    volcano = go.Figure()
    volcano.add_trace(
        go.Scattergl(
            x=results["effect_size"],
            y=results["log_q_value"],
            mode="markers",
            name="All features",
            marker={"color": "lightgray", "size": 6, "opacity": 0.6},
            hovertemplate="feature=%{text}<br>effect=%{x:.3f}<br>-log10(q)=%{y:.3f}<extra></extra>",
            text=results.index,
        )
    )
    volcano.add_trace(
        go.Scattergl(
            x=sig_results["effect_size"],
            y=sig_results["log_q_value"],
            mode="markers",
            name=f"q<{qvalue_threshold}",
            marker={"color": "firebrick", "size": 7, "opacity": 0.85},
            hovertemplate="feature=%{text}<br>effect=%{x:.3f}<br>-log10(q)=%{y:.3f}<extra></extra>",
            text=sig_results.index,
        )
    )
    volcano.add_trace(
        go.Scatter(
            x=top_hits["effect_size"],
            y=top_hits["log_q_value"],
            mode="markers",
            name=f"Top {top_n}",
            marker={
                "symbol": "circle-open",
                "color": "red",
                "size": 12,
                "line": {"color": "black", "width": 2},
            },
            hovertemplate="feature=%{text}<br>effect=%{x:.3f}<br>-log10(q)=%{y:.3f}<extra></extra>",
            text=top_hits.index.tolist(),
        )
    )
    volcano.add_hline(
        y=-np.log10(max(qvalue_threshold, np.finfo(np.float64).tiny)),
        line_dash="dot",
        line_color="royalblue",
        annotation_text=f"q={qvalue_threshold}",
    )
    volcano.add_vline(x=0, line_dash="dash", line_color="gray")
    volcano.update_layout(
        title=f"Drug Effect Volcano: {treatment_label} vs {control_value}",
        xaxis_title=f"Phenotypic shift ({layer or 'X'} units)",
        yaxis_title="-log10(q-value)",
        width=950,
        height=650,
        showlegend=legend,
    )

    # Boxplots for top hits with overlaid points (aligned by trace).
    combined = pd.concat(
        [
            drug_df[top_hits.index].assign(Condition=treatment_label),
            ctrl_df[top_hits.index].assign(Condition=control_value),
        ],
        axis=0,
    )
    melted = combined.melt(id_vars="Condition", var_name="Feature", value_name="Value")
    condition_order = [control_value, treatment_label]

    color_map = {control_value: "#636EFA", treatment_label: "#EF553B"}
    boxplot = go.Figure()
    for cond in condition_order:
        cond_df = melted[melted["Condition"] == cond]
        boxplot.add_trace(
            go.Box(
                x=cond_df["Feature"],
                y=cond_df["Value"],
                name=cond,
                marker_color=color_map[cond],
                boxpoints="all",
                jitter=0.35,
                pointpos=0.0,
                whiskerwidth=0.2,
                line_width=1.5,
                fillcolor=color_map[cond],
                opacity=0.7,
            )
        )
    boxplot.update_layout(
        title=f"Top {top_n} Features: {treatment_label} vs {control_value}",
        boxmode="group",
        width=1100,
        height=700,
        xaxis_tickangle=45,
        xaxis=dict(categoryorder="array", categoryarray=top_hits.index.tolist(), title="Feature"),
        yaxis_title=f"Feature intensity ({layer or 'X'})",
        legend_title_text="Condition",
        showlegend=legend,
    )

    if show:
        volcano.show()
        boxplot.show()

    top_table = (
        top_hits.reset_index()
        .rename(
            columns={
                "feature": "Feature",
                "effect_size": "Effect Size",
                "p_value": "P-value",
                "adjusted_p_value": "Adjusted P-value",
            }
        )
        .loc[:, ["Feature", "Effect Size", "P-value", "Adjusted P-value"]]
    )
    return top_table
