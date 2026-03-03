# CPTools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scanpy](https://img.shields.io/badge/dependency-Scanpy-brightgreen.svg)](https://scanpy.readthedocs.io/)

Cell Painting utilities for:
- Harmony + master schema ingestion into `AnnData`
- Control-based robust normalization
- Feature filtering and reproducibility-aware selection
- ZCA whitening for improved geometric comparisons
- Plotly-based visualization helpers (scatter, volcano, boxplots, arrows)

## Features

- **Integrated Pipeline**: The `funnel` function combines normalization and multiple filtering steps into one robust workflow.
- **Batch-Aware Statistics**: Normalization and treatment effects are calculated relative to matched controls in each batch.
- **AnnData Compatibility**: Built directly on top of `AnnData` for seamless integration with `Scanpy`.
- **High-Quality Visualization**: Interactive Plotly-based plots for exploratory analysis.

## Core Philosophy

Cell Painting and morphological profiling produce very high-dimensional, highly correlated, and often noisy data.  
If raw extracted features are used directly for clustering or machine learning, models can overfit technical artifacts (batch, plate position, segmentation instability) rather than biology.

CPTools uses a strict funnel to convert raw measurements into stable, reproducible biological signal:

1. `robust_zscore_norm`: Robust plate-wise normalization
- Problem: plate-to-plate intensity drift from staining, imaging, and instrument variation.
- Method: for each batch, normalize each feature against control wells (default `DMSO`) using median and MAD.
- Why it matters: aligns all batches to a shared baseline where `0` represents control phenotype, while resisting outliers.

2. `blocklist_filter`: Remove known technical artifact features
- Problem: some extracted columns are technical/positional rather than biological.
- Method: drop/flag known artifact patterns (for example location/execution-style features).
- Why it matters: prevents models from learning acquisition artifacts.

3. `nan_filter`: Remove broken/unstable features
- Problem: NaN features arise from segmentation failures or invalid transforms.
- Method: drop/flag features containing non-finite values.
- Why it matters: avoids failures and instability in PCA/UMAP/whitening.

4. `variance_filter`: Remove low-information features
- Problem: features with very low variance carry little or no perturbation signal.
- Method: in `funnel`, `variance_threshold` is treated as a variance quantile (for example `0.01` removes bottom 1% variance features).
- Why it matters: improves efficiency and denoises feature space.

5. `correlation_filter`: De-redundancy
- Problem: many morphology features are near-duplicates.
- Method: keep high-variance representatives from correlated feature clusters.
- Why it matters: avoids overweighting one biological axis (for example "size") in distance-based methods.

6. `snr_feature_selection`: Reproducibility-based selection
- Problem: a feature can vary strongly but still be noisy if replicates disagree.
- Method: signal = variance across treatment means (excluding controls); noise = within-treatment replicate variance (including controls); rank by SNR.
- Why it matters: prioritizes features that are both perturbation-responsive and replicate-consistent.

7. `zca_whiten`: Control-defined whitening
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

```python
cpt.pp.funnel(
    adata,
    batch_key="Batch",
    treatment_key="Treatment",
    control_value="DMSO",
    variance_threshold=0.01,  # drop bottom 1% by variance
    corr_threshold=0.9,
    snr_threshold=0.8,  # exclude bottom 80% by SNR
    verbose=True,  # prints per-step filtered/selected feature counts
)
```

`funnel` does **not** remove features by default. It writes:
- `adata.var["pass_funnel_prefilter"]`
- `adata.var["highly_variable"]`
- `adata.var["replicate_snr"]`

To subset features physically, opt in with:

```python
cpt.pp.funnel(adata, batch_key="Batch", treatment_key="Treatment", control_value="DMSO", subset=True)
```

Individual filters follow the same rule:
- `cpt.pp.blocklist_filter(...)`, `cpt.pp.nan_filter(...)`,
  `cpt.pp.variance_filter(...)`, `cpt.pp.correlation_filter(...)`
- default is annotation-only (`subset=False`)
- pass `subset=True` to actually remove features

## Downstream DR / Clustering

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
