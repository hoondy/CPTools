# CPTools Tutorial: End-to-End Cell Painting Analysis with the Package API

This tutorial demonstrates a clean, package-first workflow for Cell Painting analysis using **CPTools** and **Scanpy**.

It covers:
1. Loading Harmony-style results into `AnnData`
2. Robust control-based normalization
3. Feature selection (full funnel)
4. ZCA whitening
5. PCA / neighbors / UMAP
6. Mechanism-of-action visualization

---

## 1) Setup

```python
import CPTools as cpt
import scanpy as sc
import numpy as np
```

---

## 2) Read Data

Use CPTools I/O to load feature matrix + metadata directly into an `AnnData` object:

```python
res_list = ["./data/01A/PlateResults.txt", "./data/01B/PlateResults.txt"]
batch_list = ["01A", "01B"]
schema_path = "./data/Schema_MCE_master_v2.csv"

adata = cpt.read_harmony(
    plate_results_path=res_list,
    schema=schema_path,
    batch=batch_list,
)
adata
```

Expected key metadata columns for downstream steps include (names can vary by dataset):
- `Batch`
- `Treatment`
- optional: `MOA`, `PlateSet_DrugCode`, etc.

---

## 3) Robust Normalization (DMSO-centered)

Normalize each feature by batch using DMSO controls via robust z-score:

```python
adata = cpt.pp.robust_zscore_norm(adata)
```

If needed, keep a pre-whitening interpretable copy for later feature-level plots:

```python
adata.layers["normalized"] = adata.X.copy()
```

---

## 4) Feature Selection

### Recommended: full funnel

Use CPTools’ integrated funnel to perform standard feature filtering/selection:

```python
adata = cpt.pp.funnel(
    adata,
    batch_key="Batch",
    treatment_key="Treatment",
    control_value="DMSO",
)
```

This is the preferred production path.

### Optional: run replicate SNR selection directly

If you want SNR selection as a standalone step:

```python
adata = cpt.pp.snr_feature_selection(
    adata,
    treatment_key="Treatment",
    control_value="DMSO",
)
```

---

## 5) ZCA Whitening

Apply whitening for improved geometric comparability across features:

```python
adata = cpt.pp.zca_whitening(adata)
```

> Tip: perform interpretation from `adata.layers["normalized"]` (or another non-whitened layer), not the whitened matrix.

---

## 6) Embedding with Scanpy (PCA / neighbors / UMAP)

Use standard Scanpy APIs for dimensionality reduction and visualization:

```python
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

sc.pl.umap(adata, color=["Batch", "Treatment"], wspace=0.4)
```

Add any additional labels available in your metadata (e.g., `MOA`).

---

## 7) Mechanism of Action Visualization

Use CPTools tooling for drug-vs-control feature interpretation:

```python
cpt.tl.visualize_drug_mechanism(
    adata,
    drug_code="13A_Drug88",
    control_code="13A_DMSO",
    layer="normalized",  # recommended for interpretation
    top_n=5,
)
```

This typically highlights the strongest differentiating features and significance patterns between treatment and matched control.

---

## 8) Minimal End-to-End Script

```python
import CPTools as cpt
import scanpy as sc

# 1) Load
adata = cpt.read_harmony(
    plate_results_path=["./data/01A/PlateResults.txt", "./data/01B/PlateResults.txt"],
    schema="./data/Schema_MCE_master_v2.csv",
    batch=["01A", "01B"],
)

# 2) Normalize
adata = cpt.pp.robust_zscore_norm(adata)
adata.layers["normalized"] = adata.X.copy()

# 3) Feature selection (recommended integrated pipeline)
adata = cpt.pp.funnel(
    adata,
    batch_key="Batch",
    treatment_key="Treatment",
    control_value="DMSO",
)

# 4) Whitening
adata = cpt.pp.zca_whitening(adata)

# 5) Embedding
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

# 6) MoA visualization
cpt.tl.visualize_drug_mechanism(
    adata,
    drug_code="13A_Drug88",
    control_code="13A_DMSO",
    layer="normalized",
    top_n=5,
)
```

---

## Practical Notes

- Ensure controls (`DMSO`) are present in each batch for robust normalization.
- Keep layered data (`raw`, `normalized`, whitened `X`) for reproducibility and interpretation.
- Use CPTools preprocessing (`cpt.pp.*`) for morphology-specific steps and Scanpy for general manifold learning.
