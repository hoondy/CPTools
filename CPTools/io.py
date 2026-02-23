from __future__ import annotations

from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import clean_feature_name, infer_batch_from_path, make_unique


DEFAULT_FEATURE_PREFIX = "Nuclei Selected - "
CONTROL_CONTENT_VALUES = {"c", "c_DM"}


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
    schema_path: str | Path,
    batch: str,
    control_value: str,
) -> pd.DataFrame:
    meta = pd.read_csv(schema_path, sep="\t")

    if "destination_row_randomized_numeric" not in meta.columns:
        raise KeyError("Schema is missing 'destination_row_randomized_numeric'")
    if "destination_column_randomized" not in meta.columns:
        raise KeyError("Schema is missing 'destination_column_randomized'")

    meta = meta.copy()
    meta["Batch"] = batch
    meta["Well"] = meta.get("DestinationPosition_randomized")
    meta["Batch_row_col"] = (
        meta["Batch"].astype(str)
        + "_"
        + meta["destination_row_randomized_numeric"].astype(str)
        + "_"
        + meta["destination_column_randomized"].astype(str)
    )

    treatment_source = "Drug_name" if "Drug_name" in meta.columns else None
    if treatment_source is None and "Drug_code" in meta.columns:
        treatment_source = "Drug_code"

    if treatment_source is None:
        meta["Treatment"] = control_value
    else:
        meta["Treatment"] = meta[treatment_source]

    content_is_control = (
        meta["Content"].astype(str).isin(CONTROL_CONTENT_VALUES)
        if "Content" in meta.columns
        else False
    )
    control_column = (
        meta["Control"].astype(str).str.contains("DMSO", case=False, na=False)
        if "Control" in meta.columns
        else False
    )
    invalid_treatment = meta["Treatment"].astype(str).isin({"NA", "nan", "None", ""})
    control_mask = content_is_control | control_column | invalid_treatment
    meta.loc[control_mask, "Treatment"] = control_value
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
    schema_path: str | Path,
    batch: str | None = None,
    cell_type: str | None = None,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
    control_value: str = "DMSO",
) -> ad.AnnData:
    batch = batch or infer_batch_from_path(schema_path)

    res = _read_harmony_table(plate_results_path)
    if "Row" not in res.columns or "Column" not in res.columns:
        raise KeyError("PlateResults is missing 'Row' and/or 'Column' columns.")

    meta = _prepare_schema(schema_path=schema_path, batch=batch, control_value=control_value)

    res = res.copy()
    res["CellType"] = cell_type if cell_type is not None else res.get("Cell Type", "Unknown")
    res["Batch"] = batch
    res["Batch_row_col"] = (
        res["Batch"].astype(str) + "_" + res["Row"].astype(str) + "_" + res["Column"].astype(str)
    )

    merged = pd.merge(res, meta, how="left", on="Batch_row_col", suffixes=("", "_schema"))
    merged = merged[~merged["Batch_schema"].isna()].copy() if "Batch_schema" in merged.columns else merged

    feature_cols = [c for c in merged.columns if c.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(
            "No feature columns were found. Expected columns starting with "
            f"'{feature_prefix}'."
        )

    feature_matrix = merged[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    obs = merged.drop(columns=feature_cols).copy()
    if "Batch_schema" in obs.columns:
        obs["Batch"] = obs["Batch_schema"].astype(str)
        obs = obs.drop(columns=["Batch_schema"])
    obs.index = _build_obs_index(obs)

    var_names = [clean_feature_name(col) for col in feature_cols]
    var_names = make_unique(var_names)
    var = pd.DataFrame(index=var_names)
    var["feature"] = feature_cols

    adata = ad.AnnData(feature_matrix, obs=obs, var=var, dtype=np.float32)
    adata.layers["raw"] = adata.X.copy()

    if "Treatment" in adata.obs.columns:
        adata.obs["DMSO"] = np.where(
            adata.obs["Treatment"].astype(str) == control_value,
            control_value,
            "Drug",
        )
    return adata


def read_harmony(
    plate_results_path: str | Path | Sequence[str | Path],
    schema_path: str | Path | Sequence[str | Path],
    batch: str | Sequence[str] | None = None,
    cell_type: str | None = None,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
    control_value: str = "DMSO",
) -> ad.AnnData:
    """
    Read Harmony output and plate schema TSV into a single AnnData object.

    Parameters
    ----------
    plate_results_path
        Path (or list of paths) to Harmony ``PlateResults.txt`` files.
    schema_path
        Path (or list of paths) to schema TSV files with randomized destination wells.
    batch
        Optional batch id (or list). If omitted, inferred from schema filename prefix
        (for example, ``01A`` from ``01A_Plate_results_...tsv``).
    cell_type
        Optional cell type annotation to inject into ``adata.obs["CellType"]``.
    feature_prefix
        Prefix used to identify morphology feature columns.
    control_value
        Label used for untreated controls (default ``"DMSO"``).
    """
    if isinstance(plate_results_path, (str, Path)):
        return _read_single_harmony(
            plate_results_path=plate_results_path,
            schema_path=schema_path,  # type: ignore[arg-type]
            batch=batch if isinstance(batch, str) or batch is None else batch[0],
            cell_type=cell_type,
            feature_prefix=feature_prefix,
            control_value=control_value,
        )

    if not isinstance(schema_path, Sequence) or isinstance(schema_path, (str, Path)):
        raise TypeError("When plate_results_path is a sequence, schema_path must also be a sequence.")

    if len(plate_results_path) != len(schema_path):
        raise ValueError("plate_results_path and schema_path must have the same length.")

    if batch is None:
        batches: list[str | None] = [None] * len(plate_results_path)
    elif isinstance(batch, str):
        batches = [batch] * len(plate_results_path)
    else:
        if len(batch) != len(plate_results_path):
            raise ValueError("If batch is a sequence, it must match the number of input files.")
        batches = list(batch)

    adatas: list[ad.AnnData] = []
    for res_path, sch_path, b in zip(plate_results_path, schema_path, batches):
        adatas.append(
            _read_single_harmony(
                plate_results_path=res_path,
                schema_path=sch_path,
                batch=b,
                cell_type=cell_type,
                feature_prefix=feature_prefix,
                control_value=control_value,
            )
        )

    merged = ad.concat(adatas, join="inner", merge="same")
    merged.layers["raw"] = merged.X.copy()
    return merged
