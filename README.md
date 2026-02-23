# CPTools

Cell Painting utilities for:
- Harmony + schema ingestion into `AnnData`
- Control-based robust normalization
- Feature filtering and reproducibility-aware selection

## Install

```bash
pip install -e .
```

## Quick Start

```python
import CPTools as cpt

# data I/O
adata = cpt.read_harmony(
    "PlateResults.txt",
    "01A_Plate_results_Schema_MCE_partA_100Dx3x1conc_SP_001_randomized_Fri_Sep_19_16_05_32_2025__rand4834.tsv",
)

# normalization
cpt.pp.robust_zscore_norm(adata)

# feature selection by replicate SNR (flags adata.var['highly_variable'])
cpt.pp.snr_feature_selection(adata)
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
