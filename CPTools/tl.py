from __future__ import annotations

from typing import Any, Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from ._utils import to_dense_matrix


def scatter(
    adata: ad.AnnData,
    color: str | None = None,
    use_rep: str = "X_umap",
    width: int = 800,
    height: int = 800,
    title: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """
    Plot a 2D embedding from ``adata.obsm`` using Plotly.

    Example
    -------
    ``cpt.tl.scatter(adata, color="MOA", use_rep="X_umap")``
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

    color_col: str | None = None
    if color is not None:
        if color not in adata.obs.columns:
            raise KeyError(f"Column '{color}' not found in adata.obs.")
        frame[color] = adata.obs[color].astype(str).values
        color_col = color

    fig = px.scatter(
        frame,
        x=f"{use_rep}_1",
        y=f"{use_rep}_2",
        color=color_col,
        width=width,
        height=height,
        title=title or f"{use_rep} scatter",
        **kwargs,
    )
    fig.update_layout(legend_title_text=color_col or "")

    if show:
        fig.show()
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


def _drug_effect_single(
    adata: ad.AnnData,
    treatment: str,
    treatment_key: str,
    control_value: str,
    layer: str | None,
    top_n: int,
    pvalue_threshold: float,
    effect_threshold: float,
    show: bool,
) -> dict[str, Any]:
    if treatment_key not in adata.obs.columns:
        raise KeyError(f"Column '{treatment_key}' not found in adata.obs.")

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    X = _get_data_matrix(adata, layer=layer)
    frame = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
    frame[treatment_key] = adata.obs[treatment_key].astype(str).values

    drug_df = frame[frame[treatment_key] == treatment].drop(columns=treatment_key)
    ctrl_df = frame[frame[treatment_key] == control_value].drop(columns=treatment_key)

    if len(drug_df) < 2 or len(ctrl_df) < 2:
        raise ValueError(
            f"Not enough replicates for treatment '{treatment}' vs control '{control_value}'. "
            f"Need at least 2 wells each."
        )

    effect_size = drug_df.mean(axis=0) - ctrl_df.mean(axis=0)
    _, p_vals = stats.ttest_ind(drug_df, ctrl_df, axis=0, equal_var=False, nan_policy="omit")
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    log_p = -np.log10(p_vals + 1e-300)

    results = pd.DataFrame(
        {
            "feature": adata.var_names,
            "effect_size": effect_size.values,
            "p_value": p_vals,
            "log_p": log_p,
        }
    ).set_index("feature")

    top_hits = results.sort_values("log_p", ascending=False).head(top_n).copy()
    sig_mask = (results["p_value"] < pvalue_threshold) & (results["effect_size"].abs() >= effect_threshold)
    sig_results = results[sig_mask]

    # Volcano plot
    volcano = go.Figure()
    volcano.add_trace(
        go.Scattergl(
            x=results["effect_size"],
            y=results["log_p"],
            mode="markers",
            name="All features",
            marker={"color": "lightgray", "size": 6, "opacity": 0.6},
            hovertemplate="feature=%{text}<br>effect=%{x:.3f}<br>-log10(p)=%{y:.3f}<extra></extra>",
            text=results.index,
        )
    )
    volcano.add_trace(
        go.Scattergl(
            x=sig_results["effect_size"],
            y=sig_results["log_p"],
            mode="markers",
            name="Significant",
            marker={"color": "firebrick", "size": 7, "opacity": 0.85},
            hovertemplate="feature=%{text}<br>effect=%{x:.3f}<br>-log10(p)=%{y:.3f}<extra></extra>",
            text=sig_results.index,
        )
    )
    volcano.add_trace(
        go.Scatter(
            x=top_hits["effect_size"],
            y=top_hits["log_p"],
            mode="markers+text",
            name=f"Top {top_n}",
            text=top_hits.index.tolist(),
            textposition="top center",
            marker={"color": "red", "size": 9, "line": {"color": "black", "width": 1}},
            hovertemplate="feature=%{text}<br>effect=%{x:.3f}<br>-log10(p)=%{y:.3f}<extra></extra>",
        )
    )
    volcano.add_hline(
        y=-np.log10(max(pvalue_threshold, 1e-300)),
        line_dash="dot",
        line_color="royalblue",
        annotation_text=f"p={pvalue_threshold}",
    )
    volcano.add_vline(x=0, line_dash="dash", line_color="gray")
    volcano.update_layout(
        title=f"Drug Effect Volcano: {treatment} vs {control_value}",
        xaxis_title=f"Phenotypic shift ({layer or 'X'} units)",
        yaxis_title="-log10(p-value)",
        width=950,
        height=650,
    )

    # Boxplots for top hits
    combined = pd.concat(
        [
            drug_df[top_hits.index].assign(Condition=treatment),
            ctrl_df[top_hits.index].assign(Condition=control_value),
        ],
        axis=0,
    )
    melted = combined.melt(id_vars="Condition", var_name="Feature", value_name="Value")
    condition_order = [control_value, treatment]

    boxplot = px.box(
        melted,
        x="Feature",
        y="Value",
        color="Condition",
        points="all",
        category_orders={"Condition": condition_order},
        title=f"Top {top_n} Features: {treatment} vs {control_value}",
        width=1100,
        height=700,
    )
    boxplot.update_layout(
        xaxis_tickangle=45,
        yaxis_title=f"Feature intensity ({layer or 'X'})",
    )

    if show:
        volcano.show()
        boxplot.show()

    return {
        "results": results,
        "top_hits": top_hits,
        "volcano": volcano,
        "boxplot": boxplot,
    }


def visualize_drug_effect(
    adata: ad.AnnData,
    treatment: str | Sequence[str],
    treatment_key: str = "Treatment",
    control_value: str = "DMSO",
    layer: str | None = "normalized",
    top_n: int = 5,
    pvalue_threshold: float = 0.01,
    effect_threshold: float = 1.0,
    show: bool = True,
) -> dict[str, Any]:
    """
    Generate volcano plot + top-feature boxplots for treatment(s) vs control.

    Example
    -------
    ``cpt.tl.visualize_drug_effect(adata, treatment=["Triptonide", "Triptolide"])``
    """
    treatments = [treatment] if isinstance(treatment, str) else list(treatment)
    if len(treatments) == 0:
        raise ValueError("treatment cannot be empty.")

    output: dict[str, Any] = {}
    for tx in treatments:
        output[tx] = _drug_effect_single(
            adata=adata,
            treatment=tx,
            treatment_key=treatment_key,
            control_value=control_value,
            layer=layer,
            top_n=top_n,
            pvalue_threshold=pvalue_threshold,
            effect_threshold=effect_threshold,
            show=show,
        )
    return output

