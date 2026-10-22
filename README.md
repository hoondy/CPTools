# CPTools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scanpy](https://img.shields.io/badge/dependency-Scanpy-brightgreen.svg)](https://scanpy.readthedocs.io/)

Cell Painting utilities for:
- Harmony + master schema ingestion into `AnnData`
- Control-based robust normalization
- Feature filtering and reproducibility-aware selection
- ZCA whitening for improved geometric comparisons
- Plotly-based visualization helpers (scatter, volcano, boxplots, arrows)

## Features

- **Explicit Normalization**: Normalize with `robust_zscore_norm` or `reference_normalize`, then use `funnel` for filtering and feature selection.
- **Shared WT References**: `reference_normalize` separates WT reference wells from mutant vehicle controls, pools within-plate reference variability, and optionally calibrates positive-control gains.
- **Control-based Feature QC**: `control_agreement_filter` flags features on which two copies of the same control (e.g. dedicated and scattered WT+DMSO wells) disagree beyond well noise, and reports whether a disagreement is too broad to filter away.
- **Robust References**: `robust_reference_normalize` estimates plate offsets from several shared anchors, shrinks them when they are not larger than their noise, and can scale by replicate residuals of all replicated groups.
- **Bias-aware Geometry**: `group_separation`, `profile_shift` and `feature_contrasts` test genotype separation and treatment shifts along a phenotype axis, correcting distances and projections for well-sampling noise. `profile_shift_screen` scores every compound of a screen on one axis, with BH q-values.
- **Hit Calling with Nulls**: `replicate_map` scores replicate retrieval (copairs-style mAP) with label-shuffle p-values; `pseudo_treatment_null` turns random same-plate DMSO well groups into pseudo-drugs, so any score gets an empirical null via `empirical_pvalues`.
- **Batch-Aware Statistics**: Normalization and treatment effects are calculated relative to matched controls in each batch.
- **AnnData Compatibility**: Built directly on top of `AnnData` for seamless integration with `Scanpy`.
- **High-Quality Visualization**: Interactive Plotly-based plots for exploratory analysis.

## Core Philosophy

Cell Painting and morphological profiling produce very high-dimensional, highly correlated, and often noisy data.  
If raw extracted features are used directly for clustering or machine learning, models can overfit technical artifacts (batch, plate position, segmentation instability) rather than biology.

CPTools provides the following steps to convert raw measurements into reproducible profiles. Normalization and whitening are explicit calls; `funnel` performs filtering and feature selection only:

1. `robust_zscore_norm`: Robust plate-wise normalization
- Problem: plate-to-plate intensity drift from staining, imaging, and instrument variation.
- Method: for each batch, normalize each feature against control wells (default `DMSO`) using median and MAD.
- Why it matters: aligns all batches to a shared baseline where `0` represents control phenotype, while resisting outliers.

2. `blocklist_filter`: Remove known technical artifact features
- Problem: some extracted columns are technical/positional rather than biological.
- Method: drop/flag exact matches from the packaged `default_blocklists.yaml` curated CellProfiler blocklist.
- Why it matters: prevents models from learning acquisition artifacts.

3. `nan_obs_filter`: Remove bad wells/cells before feature filtering
- Problem: a small number of failed observations can contain many NaNs and make otherwise useful features look broken.
- Method: in `funnel`, drop/flag observations whose non-finite fraction exceeds `obs_nan_threshold`.
- Why it matters: protects feature-level filtering from being driven by a few failed rows.

4. `nan_filter`: Remove broken/unstable features
- Problem: NaN features arise from segmentation failures or invalid transforms.
- Method: drop/flag features containing non-finite values.
- Why it matters: avoids failures and instability in PCA/UMAP/whitening.

5. `variance_filter`: Remove low-information features
- Problem: features with very low variance carry little or no perturbation signal.
- Method: in `funnel`, `variance_threshold` is treated as a variance quantile (for example `0.01` removes bottom 1% variance features).
- Why it matters: improves efficiency and denoises feature space.

6. `correlation_filter`: De-redundancy
- Problem: many morphology features are near-duplicates.
- Method: keep high-variance representatives from correlated feature clusters.
- Why it matters: avoids overweighting one biological axis (for example "size") in distance-based methods.

7. `snr_feature_selection`: Reproducibility-based selection
- Problem: a feature can vary strongly but still be noisy if replicates disagree.
- Method: signal = variance across treatment means (excluding controls); noise = within-treatment replicate variance (including controls); rank by SNR.
- Why it matters: prioritizes features that are both perturbation-responsive and replicate-consistent.

8. `zca_whiten`: Control-defined whitening
- Problem: multivariate covariance structure can hide subtle but meaningful phenotypes.
- Method: use matched controls to estimate covariance and whiten the feature space.
- Why it matters: improves geometric comparability so distances better reflect biological dissimilarity.

By default, `funnel` annotates features (`adata.var["highly_variable"]`) instead of removing them; set `subset=True` to physically subset.

## Install

```bash
pip install git+https://github.com/hoondy/CPTools
```

## Quick Start

```python
import CPTools as cpt

# data I/O (single pre-processed schema for all plates)
res_list = ["Plate01A/PlateResults.txt", "Plate01B/PlateResults.txt"]
batch_list = ["01A", "01B"]
adata = cpt.io.read_harmony(
    plate_results_path=res_list,
    schema="/path/to/schema.csv",
    batch=batch_list,
    cell_type="iGLUT",
)

# required schema columns: Batch, Row, Column

# normalization
adata = cpt.pp.robust_zscore_norm(adata)
adata.layers["normalized"] = adata.X.copy()

# feature selection by replicate SNR (flags adata.var['highly_variable'])
cpt.pp.snr_feature_selection(adata)

# optional whitening
adata = cpt.pp.zca_whiten(adata)
```

## Typical Funnel

Run normalization first. `funnel` does **not** normalize the matrix or perform whitening.

```python
cpt.pp.funnel(
    adata,
    batch_key="Batch",
    treatment_key="Treatment",
    control_value="DMSO",
    source_layer=None,  # use current adata.X; use "normalized" to read that layer
    obs_nan_threshold=0.2,  # drop observations with >20% non-finite features
    variance_threshold=0.01,  # drop bottom 1% by variance
    corr_threshold=0.9,
    snr_threshold=0.6,  # exclude bottom 60% by SNR
    snr_groups="all",  # SNR contrast: every condition; or e.g. snr_key="line", snr_groups=["WT", "MUT"]
    verbose=True,  # prints per-step filtered/selected feature counts
)
```

To use the funnel mainly as a NaN-feature cleanup path on the current matrix, skip the optional variance, correlation, and SNR steps:

```python
cpt.pp.funnel(
    adata,
    batch_key="Batch",
    treatment_key="Treatment",
    control_value="DMSO",
    source_layer=None,  # use current adata.X
    obs_nan_threshold=0.2,
    variance_threshold=None,
    corr_threshold=None,
    snr_threshold=None,
    subset=True,
)
```

`funnel` does **not** remove features by default. It writes:
- `adata.var["pass_funnel_prefilter"]`
- `adata.var["highly_variable"]`
- `adata.var["replicate_snr"]`

For multiple genotypes or doses, pass
`replicate_groupby=["Batch", "line", "Concentration_uM"]` to `funnel` or
`snr_feature_selection`. Treatment is always included in the grouping. This
prevents genotype-dependent responses from being treated as replicate noise.

## Shared WT controls on genotype-specific plates

Use explicit reference roles when WT and mutant wells both have `Treatment="DMSO"`:

```python
import anndata as ad
import CPTools as cpt

adata = ad.read_h5ad("260908_iGLUT_bitbio_ADPD_merged_clean.h5ad")
normalized = cpt.pp.reference_normalize(
    adata,
    reference_query="line == 'WT' and Drug_code == 'c_DM'",
    source_layer="raw",
    output_layer="normalized",
    inplace=False,
)
```

This centers every plate on its dedicated WT references and uses one pooled
within-plate WT SD per feature. It does not center mutant DMSO wells separately.
Insufficient plate references raise an error; invalid-scale features are flagged
in `var["pass_reference_normalization"]` and set to NaN. Keep `layers["raw"]`
as the input to repeated fits. Positive-control calibration is optional and should
be evaluated with held-out controls before use.

See the [method and API guide](docs/reference_normalization.md).

For few references per plate, compare `cpt.pp.robust_reference_normalize`
(multi-anchor plate offsets with James-Stein shrinkage and an optional
replicate-residual scale) using `reference_control_cv(..., robust_options=...)`.
`rescue_scores` also reports noise-debiased projections; see the guide.

To subset features physically, opt in with:

```python
cpt.pp.funnel(adata, batch_key="Batch", treatment_key="Treatment", control_value="DMSO", subset=True)
```

Individual filters follow the same rule:
- `cpt.pp.blocklist_filter(...)`, `cpt.pp.nan_filter(...)`,
  `cpt.pp.variance_filter(...)`, `cpt.pp.correlation_filter(...)`,
  `cpt.pp.control_agreement_filter(...)`
- default is annotation-only (`subset=False`)
- pass `subset=True` to actually remove features

## Hit calling and empirical nulls

```python
activity = cpt.tl.replicate_map(adata, treatment_key="Treatment", control_value="DMSO",
                                batch_key="Batch", layer="normalized")
null = cpt.tl.pseudo_treatment_null(adata, treatment_key="Treatment", control_value="DMSO",
                                    batch_key="Batch", n_per_group=3, n_draws=1000)
```

`replicate_map` returns one row per treatment. Each well retrieves the
treatment's other wells against the DMSO wells of its plate by cosine
similarity, and the row reports the resulting mAP, a label-shuffle p-value
(exact when the labelings can be enumerated) and a BH q-value. With
`annotation_key="MOA"`, it scores MOA consistency instead. With triplicates
and 8 DMSO wells per plate, the smallest possible p-value is 1/165.
`pseudo_treatment_null` makes pseudo-drugs from DMSO triples on the same plate.
Score them like real drugs, e.g. with `treatment_vectors` +
`rank_treatment_correlations` or with `rescue_scores`, and compare the two sets
with `empirical_pvalues`. The [API reference](docs/api.md#hit-calling-and-empirical-nulls-replicate_map-pseudo_treatment_null-empirical_pvalues)
has worked examples for hit calling, connectivity ranking and rescue scores.

## Downstream Dimensionality Reduction / Clustering

`funnel`/`snr_feature_selection` keep all features by default and write:
- `adata.var["highly_variable"]`
- `adata.var["replicate_snr"]`

Example with Scanpy:

```python
import scanpy as sc

sc.pp.pca(adata, use_highly_variable=True)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

## API Documentation

For a full list of available functions and their parameters, see the [API Reference](docs/api.md).

## Visualization

```python
# embedding scatter (Plotly)
cpt.tl.scatter(adata, color="MOA", use_rep="X_umap")
cpt.tl.scatter(adata, color="MOA", use_rep="X_umap", legend=False)

# multi-panel scatter
cpt.tl.scatter(adata, color=["Batch", "DMSO"], use_rep="X_umap", wspace=0.4)

# treatment vs control effect plots (volcano + boxplots)
top_hits = cpt.tl.visualize_drug_effect(
    adata,
    treatment=["Triptonide", "Triptolide"],
    treatment_key="Treatment",
    control_value="DMSO",
    batch_key="Batch",  # uses matched controls from these treatment batches
    layer="normalized",
    top_n=5,
    qvalue_threshold=0.05,
    legend=False,
)
# returns a DataFrame with:
# Feature, Effect Size, P-value, Adjusted P-value
```
