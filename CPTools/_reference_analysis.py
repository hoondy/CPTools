"""Validation and rescue summaries in the shared reference feature space."""
from __future__ import annotations

from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._reference_normalization import _batches, _query_mask, _source, fit_reference_normalization


def reference_control_cv(
    adata: ad.AnnData,
    *,
    reference_query: str,
    positive_control_queries: Mapping[str, str],
    batch_key: str = "Batch",
    source_layer: str | None = "raw",
    n_splits: int = 3,
    random_state: int = 0,
    **fit_options,
) -> pd.DataFrame:
    """Hold out reference replicates within every plate and condition.

    Compare raw, WT-centered and positive-gain-corrected profiles. Fits use only
    training wells, including scale and consensus estimation. All methods are
    evaluated in the *same* training pooled-SD units (before gain), preventing
    changes to the final normalization denominator from changing the metric.

    plate_dispersion_rms is the RMS deviation of held-out plate/condition means
    from their across-plate mean. Smaller values suggest better agreement but
    can also reflect signal compression: inspect mean_effect_rms (relative to
    the held-out WT mean), gain parameters and biological profiles together.
    These are descriptive CV diagnostics, not p-values or biological repeats.
    With three wells/condition, use three folds: two train, one test per plate.
    """
    if not positive_control_queries or "reference" in positive_control_queries:
        raise ValueError("Supply positive controls with names other than 'reference'.")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    conditions = {"reference": reference_query, **positive_control_queries}
    masks = {name: _query_mask(adata.obs, query) for name, query in conditions.items()}
    union = np.sum(list(masks.values()), axis=0)
    if (union > 1).any():
        raise ValueError("Reference conditions must not overlap.")
    # Work on the small reference matrix only; all plates must still be represented.
    original_plates = np.unique(_batches(adata, batch_key))
    anchors = adata[union == 1].copy()
    batch = _batches(anchors, batch_key)
    plates = np.unique(batch)
    if not np.array_equal(original_plates, plates):
        raise ValueError("Some plates have no reference conditions.")
    X = _source(anchors, source_layer)
    condition_masks = {name: _query_mask(anchors.obs, query) for name, query in conditions.items()}
    folds = np.full(anchors.n_obs, -1, dtype=int)
    rng = np.random.default_rng(random_state)
    required = fit_options.get("min_controls", 2)
    for plate in plates:
        for name, mask in condition_masks.items():
            indices = np.flatnonzero((batch == plate) & mask)
            if len(indices) < n_splits or len(indices) - int(np.ceil(len(indices) / n_splits)) < required:
                raise ValueError(f"Plate {plate!r}, {name!r}: too few wells for {n_splits} folds and {required} training wells.")
            for fold, chunk in enumerate(np.array_split(rng.permutation(indices), n_splits)):
                folds[chunk] = fold

    records = []
    for fold in range(n_splits):
        train = anchors[folds != fold].copy()
        base = fit_reference_normalization(
            train, reference_query=reference_query, batch_key=batch_key,
            source_layer=source_layer, **fit_options,
        )
        gain = fit_reference_normalization(
            train, reference_query=reference_query, batch_key=batch_key,
            source_layer=source_layer, positive_control_queries=positive_control_queries, **fit_options,
        )
        valid = base["valid_features"] & gain["valid_features"]
        test = folds == fold
        # One common evaluation feature set per fold; no test values fit this set.
        valid &= np.isfinite(X[test]).all(axis=0)
        if not valid.any():
            raise ValueError(f"Fold {fold}: no evaluable features.")
        denominator = base["pooled_sd_before_gain"][valid]
        methods = {
            "raw": X[:, valid].copy(),
            "wt_centered": X[:, valid].copy(),
            "wt_positive_gain": X[:, valid].copy(),
        }
        for p, plate in enumerate(plates):
            selected = batch == plate
            methods["wt_centered"][selected] -= base["centers"][p, valid]
            methods["wt_positive_gain"][selected] = (
                (methods["wt_positive_gain"][selected] - gain["centers"][p, valid])
                / gain["gains"][p, valid]
            )
        for method, matrix in methods.items():
            matrix = matrix / denominator
            centroids = {
                name: np.stack([matrix[test & (batch == plate) & mask].mean(axis=0) for plate in plates])
                for name, mask in condition_masks.items()
            }
            wt_mean = centroids["reference"].mean(axis=0)
            for name, values in centroids.items():
                records.append({
                    "fold": fold, "condition": name, "method": method,
                    "n_features": int(valid.sum()),
                    "n_test_wells": int((test & condition_masks[name]).sum()),
                    "plate_dispersion_rms": float(np.sqrt(np.mean((values - values.mean(axis=0))**2))),
                    "mean_effect_rms": float(np.sqrt(np.mean((values.mean(axis=0) - wt_mean)**2))),
                })
    return pd.DataFrame(records)


def rescue_scores(
    adata: ad.AnnData,
    *,
    reference_query: str,
    disease_control_query: str,
    treatment_query: str,
    batch_key: str = "Batch",
    genotype_key: str = "line",
    treatment_key: str = "Treatment",
    dose_key: str | None = "Concentration_uM",
    reference_genotype: str = "WT",
    source_layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    min_replicates: int = 2,
    min_disease_norm: float = 1e-8,
) -> pd.DataFrame:
    """Score each mutant/treatment/dose against same-plate WT and mutant controls.

    R = -P.D / D.D; remaining_distance = ||D+P||/||D||. Also returns the
    orthogonal drug shift ||P+R*D||/||D||. R=1 alone is not complete rescue.
    Uses arithmetic well centroids in a fixed feature space, never embeddings.
    Excludes reference_genotype from scoring. Requires adequate controls and
    treatment wells; no pooling across plates or silently dropping nonfinite
    features. A tiny disease vector yields NaN scores and a status, not a hit.
    Results are descriptive: validate separation, toxicity and independent runs.
    """
    if not isinstance(min_replicates, (int, np.integer)) or min_replicates < 2:
        raise ValueError("min_replicates must be an integer >= 2.")
    if not np.isfinite(min_disease_norm) or min_disease_norm <= 0:
        raise ValueError("min_disease_norm must be finite and > 0.")
    keys = [batch_key, genotype_key, treatment_key] + ([dose_key] if dose_key else [])
    for key in keys:
        if key not in adata.obs:
            raise KeyError(f"Missing {key!r} in obs.")
    data = adata if features is None else adata[:, list(features)]
    X = _source(data, source_layer)
    if data.n_vars == 0:
        raise ValueError("Select at least one feature.")
    batch = _batches(data, batch_key)
    if data.obs[genotype_key].isna().any():
        raise ValueError("Genotypes must not be missing.")
    genotype = data.obs[genotype_key].astype(str).to_numpy()
    reference = _query_mask(data.obs, reference_query)
    disease = _query_mask(data.obs, disease_control_query)
    treated = _query_mask(data.obs, treatment_query) & (genotype != reference_genotype)
    if (reference & (genotype != reference_genotype)).any():
        raise ValueError("reference_query includes non-reference genotypes.")
    if np.any(treated & (reference | disease)):
        raise ValueError("treatment_query overlaps baseline controls.")
    if not treated.any():
        raise ValueError("No mutant treatment wells selected.")
    if not np.isfinite(X[reference | disease | treated]).all():
        raise ValueError("Selected profiles contain nonfinite values; QC and select a common finite feature set first.")
    records = []
    # Positional row indices avoid ambiguity when obs labels are not integers.
    frame = data.obs[keys].copy()
    frame["__position"] = np.arange(data.n_obs)
    for values, group in frame.loc[treated].groupby(keys, dropna=False, observed=True, sort=False):
        plate, line, treatment = values[:3]
        wt = reference & (batch == str(plate))
        mt = disease & (batch == str(plate)) & (genotype == str(line))
        if min(wt.sum(), mt.sum(), len(group)) < min_replicates:
            raise ValueError(f"Insufficient WT, disease, or treatment replicates for {plate}/{line}/{treatment}.")
        D = X[mt].mean(axis=0) - X[wt].mean(axis=0)
        P = X[group["__position"].to_numpy()].mean(axis=0) - X[mt].mean(axis=0)
        norm = float(np.linalg.norm(D))
        R = float(-np.dot(P, D) / norm**2) if norm > min_disease_norm else np.nan
        record = dict(zip(keys, values))
        record.update({
            "n_wt": int(wt.sum()), "n_disease": int(mt.sum()), "n_treatment": len(group),
            "n_features": data.n_vars, "disease_norm": norm, "rescue_projection": R,
            "remaining_distance": float(np.linalg.norm(D + P) / norm) if norm > min_disease_norm else np.nan,
            "orthogonal_shift": float(np.linalg.norm(P + R * D) / norm) if norm > min_disease_norm else np.nan,
            "status": "ok" if norm > min_disease_norm else "disease_vector_too_small",
        })
        records.append(record)
    return pd.DataFrame(records)
