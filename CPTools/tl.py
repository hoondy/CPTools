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
