"""Within-plate position effects: diagnostics, and removal by two-way median polish."""
from __future__ import annotations

from typing import Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from ._reference_normalization import _batches, _query_mask, _source

DISTANCE_BINS = (0.0, 1.01, 1.5, 2.5, 4.0, 6.0, 10.0, np.inf)


def _sweep(W, fit, labels, min_wells):
    """Subtract per-label medians of the fit wells from every well; return the effects."""
    effect = np.zeros_like(W)
    short = []
    for value in pd.unique(labels):
        rows = labels == value
        sample = W[rows & fit]
        if len(sample) < min_wells:
            short.append(value)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN feature columns
            median = np.nanmedian(sample, axis=0)
        effect[rows] = np.nan_to_num(median)
    W -= effect
    return effect, short


def spatial_polish(
    adata: ad.AnnData,
    *,
    batch_key: str = "Batch",
    row_key: str = "Row",
    column_key: str = "Column",
    source_layer: str | None = "normalized",
    output_layer: str = "polished",
    fit_query: str | None = None,
    origin_query: str | None = None,
    n_iter: int = 3,
    min_wells: int = 4,
    inplace: bool = True,
) -> ad.AnnData:
    """Remove row and column effects within each plate by two-way median polish.

    For each batch and feature, alternating sweeps subtract the median of every
    row and then every column (Tukey's median polish, the location part of the
    B-score of Brideau et al. 2003, without its MAD rescaling). What remains is
    the well's deviation from its plate's typical well after row and column
    effects, so each plate ends up centered on its typical well.
    ``origin_query`` (e.g. ``"Treatment == 'DMSO'"``) then subtracts the median
    of those wells pooled over all batches, one vector for every plate, so 0
    means that condition again without reintroducing per-plate noise.

    Medians are taken over ``fit_query`` wells (default: all) and subtracted
    from every well in the row or column. This assumes treatments are placed at
    random, so a row or column median reflects position rather than biology.
    Controls in fixed rows or columns, or a set of treatments randomized
    separately into its own rows, break that: exclude them with ``fit_query``;
    their row or column effect then comes from the other wells there. Rows or
    columns with fewer than ``min_wells`` fit wells get no correction (a
    warning names them). Remove locally failed wells (e.g. a damaged block
    found with ``replicate_outliers``) beforehand: a block that pulls its row's
    median would shift the healthy wells of that row. Each median of a few
    wells adds noise to every well of its row or column, so compare replicate
    agreement and ``batch_similarity`` before and after on your data.

    Writes X and ``output_layer``. ``uns['cptools']['spatial_polish']`` holds the
    parameters, per-batch RMS (over wells and features) of the removed row and
    column effects, and the ``origin_shift`` vector.
    """
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
        raise ValueError("n_iter must be an integer >= 1.")
    if not isinstance(min_wells, (int, np.integer)) or min_wells < 1:
        raise ValueError("min_wells must be an integer >= 1.")
    if not output_layer or output_layer == "raw" or output_layer == source_layer:
        raise ValueError("output_layer must be nonempty and distinct from the source and 'raw'.")
    for key in (row_key, column_key):
        if key not in adata.obs:
            raise KeyError(f"Missing {key!r} in obs.")
        if adata.obs[key].isna().any():
            raise ValueError(f"{key!r} must not be missing.")
    batch = _batches(adata, batch_key)
    X = _source(adata, source_layer)
    fit = np.ones(adata.n_obs, dtype=bool) if fit_query is None else _query_mask(adata.obs, fit_query)
    row = adata.obs[row_key].astype(str).to_numpy()
    column = adata.obs[column_key].astype(str).to_numpy()

    output = np.empty(X.shape, dtype=np.float64)
    row_rms, column_rms, uncorrected = {}, {}, []
    for plate in pd.unique(batch):
        wells = batch == plate
        W = X[wells].copy()
        row_effect, column_effect = np.zeros_like(W), np.zeros_like(W)
        for _ in range(n_iter):
            effect, short_rows = _sweep(W, fit[wells], row[wells], min_wells)
            row_effect += effect
            effect, short_columns = _sweep(W, fit[wells], column[wells], min_wells)
            column_effect += effect
        uncorrected += [f"{plate} {row_key} {r}" for r in short_rows]
        uncorrected += [f"{plate} {column_key} {c}" for c in short_columns]
        output[wells] = W
        # The plate's overall level is part of the first sweeps; report effects around it.
        row_rms[plate] = float(np.sqrt(np.nanmean((row_effect - np.nanmedian(row_effect, axis=0)) ** 2)))
        column_rms[plate] = float(np.sqrt(np.nanmean((column_effect - np.nanmedian(column_effect, axis=0)) ** 2)))
    if uncorrected:
        warnings.warn(f"No correction (fewer than {min_wells} fit wells) for {len(uncorrected)} rows/columns, "
                      f"e.g. {uncorrected[:5]}.", stacklevel=2)

    shift = np.zeros(X.shape[1])
    if origin_query is not None:
        origin = _query_mask(adata.obs, origin_query)
        if not origin.any():
            raise ValueError("origin_query selects no wells.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            shift = np.nan_to_num(np.nanmedian(output[origin], axis=0))
        output -= shift

    target = adata if inplace else adata.copy()
    output = output.astype(np.float32)
    target.X = output
    target.layers[output_layer] = output.copy()
    target.uns.setdefault("cptools", {})["spatial_polish"] = {
        "batch_key": batch_key, "row_key": row_key, "column_key": column_key,
        "source_layer": source_layer if source_layer is not None else "",
        "fit_query": fit_query or "", "origin_query": origin_query or "",
        "n_iter": int(n_iter), "min_wells": int(min_wells),
        "batches": np.asarray(list(row_rms), dtype=str),
        "row_effect_rms": np.asarray(list(row_rms.values())),
        "column_effect_rms": np.asarray(list(column_rms.values())),
        "origin_shift": shift,
    }
    return target


def _position_data(adata, group_key, batch_key, row_key, column_key, layer, features, query):
    data = adata if query is None else adata[_query_mask(adata.obs, query)]
    for key in (group_key, row_key, column_key):
        if key not in data.obs:
            raise KeyError(f"Missing {key!r} in obs.")
        if data.obs[key].isna().any():
            raise ValueError(f"{key!r} must not be missing.")
    X = _source(data if features is None else data[:, list(features)], layer)
    if X.shape[1] == 0:
        raise ValueError("Select at least one feature.")
    batch = _batches(data, batch_key)
    group = data.obs[group_key].astype(str).to_numpy()
    row = data.obs[row_key].astype(float).to_numpy()
    column = data.obs[column_key].astype(float).to_numpy()
    return np.asarray(X, dtype=np.float64), batch, group, row, column


def _replicate_residuals(X, group):
    """Each well minus the mean of its group; False in ``usable`` for single wells and zero residuals."""
    residual = X - pd.DataFrame(X).groupby(group).transform("mean").to_numpy()
    size = pd.Series(group).map(pd.Series(group).value_counts()).to_numpy()
    norm = np.linalg.norm(residual, axis=1)
    return residual, (size > 1) & np.isfinite(norm) & (norm > 0)


def position_similarity(
    adata: ad.AnnData,
    *,
    group_key: str = "Treatment",
    batch_key: str = "Batch",
    row_key: str = "Row",
    column_key: str = "Column",
    layer: str | None = None,
    features: Sequence[str] | None = None,
    query: str | None = None,
    bins: Sequence[float] = DISTANCE_BINS,
) -> pd.DataFrame:
    """How alike the noise of two wells is, by their distance on the plate.

    A replicate residual is a well minus the mean of its group's wells on the
    same batch (``group_key``, e.g. compound). It contains no group effect, so
    residuals of wells from different groups resemble each other only through
    technical factors they share. With groups placed at random, similarity that
    falls with distance, or is higher along a row or a column, is a position
    effect (dispensing channel, edge evaporation, imaging order).

    Returns one row per batch and relation: the mean cosine similarity of
    residual pairs from different groups (``cosine``), the number of pairs
    (``pairs``) and their mean distance (``distance``). Relations are the
    distance ``bins`` in well spacings, center to center (1 = adjacent, 1.41 =
    diagonal), labelled like ``"(0, 1.01]"``, plus ``"same row"`` and
    ``"same column"`` at any distance. Groups with a single well on a batch
    have no residual and are skipped. Compare the nearest bin with the
    farthest one on every batch; a correction such as ``spatial_polish``
    should bring both toward 0.
    """
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be at least two increasing edges.")
    labels = [f"({lo:g}, {hi:g}]" for lo, hi in zip(edges[:-1], edges[1:])]
    X, batch, group, row, column = _position_data(adata, group_key, batch_key, row_key, column_key, layer,
                                                  features, query)
    records = []
    for b in pd.unique(batch):
        wells = np.flatnonzero(batch == b)
        residual, usable = _replicate_residuals(X[wells], group[wells])
        wells, residual = wells[usable], residual[usable]
        if len(wells) < 2:
            continue
        unit = residual / np.linalg.norm(residual, axis=1, keepdims=True)
        upper = np.triu_indices(len(wells), 1)
        different = (group[wells][:, None] != group[wells][None, :])[upper]
        cosine = (unit @ unit.T)[upper][different]
        r, c = row[wells], column[wells]
        distance = np.hypot(r[:, None] - r[None, :], c[:, None] - c[None, :])[upper][different]
        which = np.searchsorted(edges, distance, side="left") - 1
        for i, label in enumerate(labels):
            hit = which == i
            if hit.any():
                records.append((b, label, distance[hit].mean(), cosine[hit].mean(), int(hit.sum())))
        for label, same in (("same row", r[:, None] == r[None, :]), ("same column", c[:, None] == c[None, :])):
            hit = same[upper][different]
            if hit.any():
                records.append((b, label, distance[hit].mean(), cosine[hit].mean(), int(hit.sum())))
    if not records:
        raise ValueError("No batch has two wells with replicate residuals.")
    return pd.DataFrame(records, columns=["batch", "relation", "distance", "cosine", "pairs"])


def position_map(
    adata: ad.AnnData,
    *,
    group_key: str = "Treatment",
    batch_key: str = "Batch",
    row_key: str = "Row",
    column_key: str = "Column",
    layer: str | None = None,
    features: Sequence[str] | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    """Typical deviation of each plate position from its replicates, across batches.

    Replicate residuals (each well minus the mean of its group's wells on its
    batch, as in ``position_similarity``) are averaged over batches for every
    position, and the average is summarized as the root mean square over
    features. Group effects cancel within each residual and independent noise
    shrinks with the number of batches (to about its SD / sqrt(n_batches)), so
    positions that deviate the same way on many plates stand out: edges, a
    dispensing channel, a failing tip. Returns a table indexed by row with one
    column per plate column; NaN where no batch has a residual.
    """
    X, batch, group, row, column = _position_data(adata, group_key, batch_key, row_key, column_key, layer,
                                                  features, query)
    residuals, rows, columns = [], [], []
    for b in pd.unique(batch):
        wells = np.flatnonzero(batch == b)
        residual, usable = _replicate_residuals(X[wells], group[wells])
        residuals.append(residual[usable])
        rows.append(row[wells][usable])
        columns.append(column[wells][usable])
    mean = pd.DataFrame(np.vstack(residuals)).groupby([np.concatenate(rows), np.concatenate(columns)]).mean()
    rms = np.sqrt((mean ** 2).mean(axis=1))
    table = rms.unstack()
    for axis in ("index", "columns"):
        labels = getattr(table, axis)
        if np.all(np.mod(labels, 1) == 0):
            setattr(table, axis, labels.astype(int))
    table.index.name, table.columns.name = row_key, column_key
    return table
