# CPTools Tutorial: End-to-End Cell Painting Analysis with the Package API

This tutorial demonstrates a clean, package-first workflow for Cell Painting analysis using **CPTools** and **Scanpy**.

It covers:
1. Loading Harmony-style results into `AnnData`
2. Robust control-based normalization
3. Feature selection (full funnel)
4. ZCA whitening
5. PCA / neighbors / UMAP
6. Plotly visualization (`scatter` + `visualize_drug_effect`)

---

## 1) Setup

```python
import CPTools as cpt
import scanpy as sc
```

---

## 2) Read Data

Use CPTools I/O to load feature matrix + metadata directly into an `AnnData` object:

```python
res_list = ["./data/01A/PlateResults.txt", "./data/01B/PlateResults.txt"]
batch_list = ["01A", "01B"]
schema_path = "./data/Schema_MCE_master_v2.csv"

adata = cpt.io.read_harmony(
    plate_results_path=res_list,
    schema=schema_path,
    batch=batch_list,
)
adata
```

Required metadata columns:
- `Batch`
- `Row`
- `Column`
- `Treatment` (required for normalization/selection/Drug effect visualization)
- optional: `MOA`, `PathWay`, `Research_Area`, etc.

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
    verbose=True,  # prints per-step filtered/selected feature counts
)
```

This is the preferred production path. By default it does not remove features;
it marks them via:
- `adata.var["pass_funnel_prefilter"]`
- `adata.var["highly_variable"]`
- `adata.var["replicate_snr"]`

Use `subset=True` only if you explicitly want to remove non-selected features.

The same behavior applies to individual filters:
- `blocklist_filter`, `drop_nan_features`, `variance_filter`, `correlation_filter`
- default: annotate pass/fail masks in `adata.var`
- `subset=True`: physically subset features

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
adata = cpt.pp.zca_whiten(adata)
```

> Tip: perform interpretation from `adata.layers["normalized"]` (or another non-whitened layer), not the whitened matrix.

---

## 6) Embedding with Scanpy (PCA / neighbors / UMAP)

Use standard Scanpy APIs for dimensionality reduction and visualization:

```python
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
```

Plot the embedding with Plotly:

```python
cpt.tl.scatter(adata, color="MOA", use_rep="X_umap")
```

---

## 7) Drug Effect Visualization

Use CPTools tooling for treatment-vs-control feature interpretation:

```python
cpt.tl.visualize_drug_effect(
    adata,
    treatment=["Triptonide", "Triptolide"],
    treatment_key="Treatment",
    control_value="DMSO",
    layer="normalized",
    top_n=5,
)
```

This generates:
- Volcano plots per treatment vs control
- Boxplots for top differentiating features

---

## 8) Minimal End-to-End Script

```python
import CPTools as cpt
import scanpy as sc

# 1) Load
adata = cpt.io.read_harmony(
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
    subset=False,  # default: keep all features, mark highly_variable
    verbose=True,
)

# 4) Whitening
adata = cpt.pp.zca_whiten(adata)

# 5) Embedding
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
cpt.tl.scatter(adata, color="MOA", use_rep="X_umap")

# 6) Drug-effect visualization
cpt.tl.visualize_drug_effect(
    adata,
    treatment=["Triptonide", "Triptolide"],
    treatment_key="Treatment",
    control_value="DMSO",
    layer="normalized",
    top_n=5,
)
```

---

## Practical Notes

- Ensure controls (`DMSO`) are present in each batch for robust normalization.
- Keep layered data (`raw`, `normalized`, whitened `X`) for reproducibility and interpretation.
- Use CPTools preprocessing (`cpt.pp.*`) for morphology-specific steps and Plotly/Scanpy for visualization and manifold learning.
