from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import clean_feature_name, make_unique


DEFAULT_FEATURE_PREFIX = "Nuclei Selected - "
CONTROL_CONTENT_VALUES = {"c", "c_DM"}


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV-like tables using automatic delimiter detection."""
    table = pd.read_csv(path, sep=None, engine="python")
    table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed")]
    return table


def _read_harmony_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    data_row_idx: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if line.strip() == "[Data]":
                data_row_idx = idx
                break

    if data_row_idx is None:
        raise ValueError(f"Could not find [Data] section in: {path}")

    table = pd.read_csv(path, sep="\t", skiprows=data_row_idx + 1)
    table = table.loc[:, ~table.columns.str.startswith("Unnamed")]
    table = table.dropna(axis=1, how="all")
    return table


def _prepare_schema(
    schema: str | Path,
    control_value: str,
) -> pd.DataFrame:
    """
    Read pre-processed master metadata and validate required join columns.

    Required columns: Batch, Row, Column.
    """
    meta = _read_table(schema).copy()
    required = ["Batch", "Row", "Column"]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise KeyError(f"Schema is missing required columns: {missing}")

    na_counts = meta[required].isna().sum()
    if (na_counts > 0).any():
        missing_summary = {col: int(count) for col, count in na_counts.items() if count > 0}
        raise ValueError(f"Schema has missing values in required columns: {missing_summary}")

    meta["Batch"] = meta["Batch"].astype(str)
    meta["Row"] = pd.to_numeric(meta["Row"], errors="coerce")
    meta["Column"] = pd.to_numeric(meta["Column"], errors="coerce")

    if meta["Row"].isna().any() or meta["Column"].isna().any():
        raise ValueError("Schema Row/Column contain non-numeric values after parsing.")

    meta["Row"] = meta["Row"].astype(int)
    meta["Column"] = meta["Column"].astype(int)
    meta["Batch_row_col"] = (
        meta["Batch"] + "_" + meta["Row"].astype(str) + "_" + meta["Column"].astype(str)
    )

    # Minimal convenience logic for control/treatment labels when present.
    if "Treatment" not in meta.columns:
        treatment_source = "Drug_name" if "Drug_name" in meta.columns else None
        if treatment_source is None and "Drug_code" in meta.columns:
            treatment_source = "Drug_code"
        if treatment_source is None:
            meta["Treatment"] = control_value
        else:
            meta["Treatment"] = meta[treatment_source]

    if "Content" in meta.columns:
        content_is_control = meta["Content"].astype(str).isin(CONTROL_CONTENT_VALUES)
        meta.loc[content_is_control, "Treatment"] = control_value

    invalid_treatment = meta["Treatment"].astype(str).isin({"NA", "nan", "None", ""})
    meta.loc[invalid_treatment, "Treatment"] = control_value

    if "DMSO" not in meta.columns and "Treatment" in meta.columns:
        meta["DMSO"] = np.where(meta["Treatment"].astype(str) == control_value, control_value, "Drug")

    dup_mask = meta["Batch_row_col"].duplicated(keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        example_keys = meta.loc[dup_mask, "Batch_row_col"].head(5).tolist()
        raise ValueError(
            f"Schema has duplicated Batch/Row/Column keys ({n_dup} rows). "
            f"Example keys: {example_keys}"
        )

    return meta


def _build_obs_index(obs: pd.DataFrame) -> pd.Index:
    if "Batch" in obs.columns and "Well" in obs.columns:
        values = obs["Batch"].astype(str) + "_" + obs["Well"].astype(str)
        return pd.Index(make_unique(values.tolist()))

    if "Batch_row_col" in obs.columns:
        return pd.Index(make_unique(obs["Batch_row_col"].astype(str).tolist()))

    return pd.Index(make_unique([str(i) for i in range(len(obs))]))


def _read_single_harmony(
    plate_results_path: str | Path,
    schema_df: pd.DataFrame,
    batch: str,
    cell_type: str | None = None,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
    control_value: str = "DMSO",
) -> ad.AnnData:
    res = _read_harmony_table(plate_results_path)
    if "Row" not in res.columns or "Column" not in res.columns:
        raise KeyError("PlateResults is missing 'Row' and/or 'Column' columns.")

    res = res.copy()
    res["Row"] = pd.to_numeric(res["Row"], errors="coerce")
    res["Column"] = pd.to_numeric(res["Column"], errors="coerce")
    if res["Row"].isna().any() or res["Column"].isna().any():
        raise ValueError(f"PlateResults has non-numeric Row/Column values: {plate_results_path}")
    res["Row"] = res["Row"].astype(int)
    res["Column"] = res["Column"].astype(int)
    res["CellType"] = cell_type if cell_type is not None else res.get("Cell Type", "Unknown")
    res["Batch"] = str(batch)
    res["Batch_row_col"] = (
        res["Batch"].astype(str) + "_" + res["Row"].astype(str) + "_" + res["Column"].astype(str)
    )

    meta_batch = schema_df[schema_df["Batch"].astype(str) == str(batch)].copy()
    if meta_batch.empty:
        raise ValueError(f"No schema rows found for batch '{batch}'.")

    merged = pd.merge(res, meta_batch, how="left", on="Batch_row_col", suffixes=("", "_schema"), indicator=True)
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        n_unmatched = int(unmatched.sum())
        warnings.warn(
            f"{n_unmatched} wells in batch '{batch}' were not found in schema and will be dropped.",
            stacklevel=2,
        )
        merged = merged.loc[~unmatched].copy()
    merged = merged.drop(columns=["_merge"])

    if merged.empty:
        raise ValueError(f"No matched wells left after merging PlateResults with schema for batch '{batch}'.")

    # Keep canonical keys from schema.
    for col in ["Batch", "Row", "Column"]:
        schema_col = f"{col}_schema"
        if schema_col in merged.columns:
            merged[col] = merged[schema_col]
            merged = merged.drop(columns=[schema_col])

    feature_cols = [c for c in merged.columns if c.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(
            "No feature columns were found. Expected columns starting with "
            f"'{feature_prefix}'."
        )

    feature_matrix = merged[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    obs = merged.drop(columns=feature_cols).copy()
    obs.index = _build_obs_index(obs)

    var_names = [clean_feature_name(col) for col in feature_cols]
    var_names = make_unique(var_names)
    var = pd.DataFrame(index=var_names)
    var["feature"] = feature_cols

    adata = ad.AnnData(feature_matrix, obs=obs, var=var, dtype=np.float32)
    adata.layers["raw"] = adata.X.copy()

    if "DMSO" not in adata.obs.columns and "Treatment" in adata.obs.columns:
        adata.obs["DMSO"] = np.where(
            adata.obs["Treatment"].astype(str) == control_value,
            control_value,
            "Drug",
        )
    return adata


def read_harmony(
    plate_results_path: str | Path | Sequence[str | Path],
    schema: str | Path | None = None,
    batch: str | Sequence[str] | None = None,
    cell_type: str | None = None,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
    control_value: str = "DMSO",
    schema_path: str | Path | None = None,
) -> ad.AnnData:
    """
    Read Harmony output and a master schema table into a single AnnData object.

    Parameters
    ----------
    plate_results_path
        Path (or list of paths) to Harmony ``PlateResults.txt`` files.
    schema
        Path to pre-processed metadata table containing at least ``Batch``, ``Row``, ``Column``.
    batch
        Batch id (or list of batch ids aligned to ``plate_results_path``).
        Required when reading multiple plate files.
    cell_type
        Optional cell type annotation to inject into ``adata.obs["CellType"]``.
    feature_prefix
        Prefix used to identify morphology feature columns.
    control_value
        Label used for untreated controls (default ``"DMSO"``).
    schema_path
        Deprecated alias for ``schema``.
    """
    if schema is None:
        if schema_path is None:
            raise TypeError("read_harmony() missing required argument: 'schema'")
        warnings.warn(
            "'schema_path' is deprecated; use 'schema' with a single master metadata file.",
            DeprecationWarning,
            stacklevel=2,
        )
        schema = schema_path
    elif schema_path is not None:
        raise ValueError("Pass only one of 'schema' or deprecated 'schema_path', not both.")

    schema_df = _prepare_schema(schema=schema, control_value=control_value)

    if isinstance(plate_results_path, (str, Path)):
        plate_paths = [plate_results_path]
    else:
        plate_paths = list(plate_results_path)

    if len(plate_paths) == 0:
        raise ValueError("plate_results_path is empty.")

    if batch is None:
        if len(plate_paths) > 1:
            raise ValueError(
                "When reading multiple PlateResults files, provide 'batch' as a list aligned with paths."
            )
        unique_batches = schema_df["Batch"].astype(str).unique().tolist()
        if len(unique_batches) != 1:
            raise ValueError(
                "Batch is ambiguous: schema contains multiple batches. "
                "Provide 'batch' explicitly."
            )
        batches: list[str] = [unique_batches[0]]
    elif isinstance(batch, str):
        batches = [batch] * len(plate_paths)
    else:
        if len(batch) != len(plate_paths):
            raise ValueError("If batch is a sequence, it must match the number of input files.")
        batches = list(batch)

    adatas: list[ad.AnnData] = []
    for res_path, b in zip(plate_paths, batches):
        adatas.append(
            _read_single_harmony(
                plate_results_path=res_path,
                schema_df=schema_df,
                batch=b,
                cell_type=cell_type,
                feature_prefix=feature_prefix,
                control_value=control_value,
            )
        )

    if len(adatas) == 1:
        return adatas[0]

    merged = ad.concat(adatas, join="inner", merge="same")
    merged.layers["raw"] = merged.X.copy()
    return merged
