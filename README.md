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
    variance_threshold=1e-2,
    corr_threshold=0.9,
    snr_keep_top_fraction=0.2,
)
```

## Downstream DR / Clustering

`snr_feature_selection` keeps all features by default and writes:
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

# treatment vs control effect plots (volcano + boxplots)
cpt.tl.visualize_drug_effect(
    adata,
    treatment=["Triptonide", "Triptolide"],
    treatment_key="Treatment",
    control_value="DMSO",
    layer="normalized",
    top_n=5,
)
```
