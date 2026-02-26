# CPTools

Cell Painting utilities for:
- Harmony + master schema ingestion into `AnnData`
- Control-based robust normalization
- Feature filtering and reproducibility-aware selection
- Plotly-based visualization helpers

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
