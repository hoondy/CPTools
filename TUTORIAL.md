# CPTools Tutorial: End-to-End Cell Painting Analysis with AnnData/Scanpy

This tutorial walks through a practical **CPTools** workflow for Cell Painting / high-content screening data, based on a real notebook pipeline.

It covers:
1. Data ingestion (`cp2h5ad`) and multi-plate merge
2. Robust per-batch normalization with DMSO controls
3. Multi-stage feature selection
4. ZCA whitening
5. PCA + Harmony + UMAP integration/visualization
6. Mechanism-of-action (MoA) feature interpretation
7. Phenotypic activity scoring by distance from control

---

## 1) Introduction

**CPTools** is a Scanpy/AnnData-centered analysis workflow for Cell Painting data. It converts plate-level outputs into a unified matrix (`.h5ad`), harmonizes metadata, applies robust normalization and feature engineering, then supports interpretable downstream analyses (MoA plots, phenotypic distance, embedding visualization).

At a high level:

- **Input:** plate-level quantified morphological features + plate metadata
- **Core processing:** batch-aware control normalization, feature reduction, whitening
- **Output:** integrated embeddings and interpretable feature-level comparisons

---

## 2) Data Ingestion: PlateResults → AnnData (`cp2h5ad`) and Batch Merge

## 2.1 Build unified metadata table

The workflow first merges many plate metadata files into one master table and standardizes identifiers.

Key logic:
- Construct a unique key (`Batch_row_col`) to align image-derived features with plate map metadata
- Mark controls (`DMSO`) consistently
- Create useful annotations (`Treatment`, `Batch`, `PlateSet_DrugCode`, `MOA`, etc.)
- Merge external drug info table

Example pattern:

```python
meta_merged['Batch_row_col'] = [
    f"{b}_{r}_{c}" for b, r, c in zip(meta_merged.Batch,
                                       meta_merged.destination_row_randomized_numeric,
                                       meta_merged.destination_column_randomized)
]

meta_merged['Treatment'] = meta_merged['Drug_name']
meta_merged.loc[meta_merged['Content'].isin(['c', 'c_DM']), 'Treatment'] = 'DMSO'
```

Save this as a master CSV used by all plates.

## 2.2 Convert each plate to `.h5ad` with `cp2h5ad`

`cp2h5ad(...)` does the heavy lifting:

- Reads `PlateResults.txt`
- Adds plate/cell type context
- Merges with metadata via `Batch_row_col`
- Splits into:
  - `X`: morphological features (`Nuclei Selected - ...` columns)
  - `obs`: well-level annotations
  - `var`: feature metadata (cleaned names)
- Stores raw matrix in `adata.layers['raw']`
- Adds categorical annotations (`DMSO`, `MOA`, `PathWay`, `Research_Area`)
- Writes one h5ad per plate

Core function shape:

```python
def cp2h5ad(prefix, CellType, Batch, res_path, meta_path):
    # read plate results + metadata
    # merge on Batch_row_col
    # build sparse X from feature columns
    # build obs/var annotations
    # keep raw layer
    # save prefix_CellType_Batch.h5ad
```

## 2.3 Merge all plates

After generating one h5ad per plate:

```python
list_adata = [ad.read_h5ad(x) for x in list_h5ad]
adata = ad.concat(list_adata, merge="same")
adata.write('260118_iGLUT_merged.h5ad')
```

Optional QC filtering (example from notebook): remove problematic compounds (e.g., failed dispensing cases such as Tedizolid/Penicillamine in specific wells), then save filtered merged object.

---

## 3) Preprocessing & Normalization: Robust Z-score by Batch (DMSO-based)

The key normalization uses **DMSO controls within each batch/plate**.

For each feature and batch:

\[
Z = \frac{X - \text{median}(\text{DMSO})}{\text{MAD}(\text{DMSO}) + \epsilon}
\]

where MAD is scaled to normal consistency (`scale='normal'` in `median_abs_deviation`).

Why this is good for Cell Painting:
- Robust to outliers (median/MAD > mean/std)
- Corrects batch-specific shifts
- Centers controls near 0 for each feature

Function used:

```python
adata = robust_zscore_normalize(
    adata,
    batch_key='Batch',
    treatment_key='Treatment',
    control_val='DMSO'
)
```

Sanity check:

```python
dmso_idx = adata.obs['Treatment'] == 'DMSO'
print(np.mean(adata.X[dmso_idx]))  # should be near 0
```

---

## 4) Feature Selection Pipeline

Feature selection is done in stages to keep biologically meaningful, reproducible signal.

## 4.1 Unsupervised reduction (`unsupervised_feature_reduction`)

Steps:
1. **Blocklist removal** (technical/artifactual feature families):
   - `Manders`, `RWC`, `Location`, `Granularity`, `Execution`, `Euler`
2. **Drop NaN-containing features**
3. **Low variance filter** (e.g., `VarianceThreshold(1e-2)`)

```python
adata = unsupervised_feature_reduction(adata, variance_threshold=1e-2)
```

## 4.2 Correlation reduction (`variance_based_correlation_reduction`)

Goal: remove redundancy among highly correlated features.

Method:
- Compute absolute correlation matrix
- Sort features by variance (high → low)
- Keep the first feature in each high-correlation cluster
- Drop correlated followers (threshold e.g. 0.9)

```python
adata = variance_based_correlation_reduction(adata, threshold=0.9)
```

This keeps high-information representatives while reducing dimensional redundancy.

## 4.3 Replicate SNR selection (`feature_selection_by_replicates`)

Uses a signal-to-noise ratio per feature:

- **Signal:** variance across treatment means
- **Noise:** average within-treatment replicate variance
- **SNR:** `signal / (noise + eps)`

Then drop bottom quantile features by SNR.

```python
adata = feature_selection_by_replicates(
    adata,
    treatment_key='Treatment',
    control_key='DMSO',
    quantile_threshold=0.8
)
```

> Note: In this implementation, `quantile_threshold=0.8` keeps roughly top 20% SNR features.

---

## 5) Data Whitening: ZCA (`zca_whitening`)

After normalization and feature selection, apply **ZCA whitening** using control covariance.

Procedure:
1. Compute covariance matrix from DMSO controls
2. SVD/eigendecomposition
3. Build whitening matrix:

\[
W = U\,\text{diag}\left(\frac{1}{\sqrt{\lambda + \epsilon}}\right)U^T
\]

4. Transform all samples (`X_whitened = X_centered @ W`)

```python
adata = zca_whitening(adata, treatment_key='Treatment', control_key='DMSO')
```

Important practical point from the notebook:
- Keep an interpretable pre-whitened layer for feature-level interpretation:

```python
adata.layers['pre_whitening'] = adata.X.copy()
adata.X = X_whitened
```

Whitened space is excellent for distance geometry and downstream embeddings, but less directly interpretable for per-feature biological explanation.

---

## 6) Dimensionality Reduction & Integration (PCA, Harmony, UMAP)

## 6.1 PCA

```python
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=15)
```

Inspect PCA colored by `Batch` and `DMSO` to evaluate structure and batch effects.

## 6.2 Harmony batch correction

```python
import scanpy.external as sce
sce.pp.harmony_integrate(adata, "Batch", max_iter_harmony=20)
```

This creates `adata.obsm['X_pca_harmony']`.

## 6.3 Neighbor graph + UMAP

```python
sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=15)
sc.tl.umap(adata)
```

Plot UMAP by:
- `Batch` (batch mixing quality)
- `DMSO` (control behavior)
- `MOA` (biological grouping)

---

## 7) Downstream Analysis & Visualization

## 7.1 Mechanism of Action visualization (`visualize_drug_mechanism`)

This function compares one drug condition vs matched control (`PlateSet_DrugCode`) and generates:

1. **Volcano-like plot**
   - x-axis: effect size (mean drug - mean control)
   - y-axis: `-log10(p)` from Welch’s t-test
2. **Box/strip plots** for top differentiating features

Use an interpretable layer (`raw` or normalized but non-whitened), e.g.:

```python
adata.layers['normalized'] = adata.X.copy()  # before whitening preferred
visualize_drug_mechanism(
    adata,
    drug_code='13A_Drug88',
    control_code='13A_DMSO',
    layer='normalized',
    top_n=5
)
```

## 7.2 Phenotypic distance from control

In whitened space, control phenotype is centered near the origin. A simple activity metric is Euclidean distance from origin:

```python
distances = np.linalg.norm(adata.X, axis=1)
adata.obs['distance_from_control'] = distances

mean_dist = (
    adata.obs
    .groupby('Treatment')['distance_from_control']
    .mean()
    .sort_values(ascending=False)
)

print(mean_dist.head())      # most phenotypically active treatments
print(mean_dist['DMSO'])     # should be low
```

This gives a fast ranking of compounds by global morphological perturbation strength.

---

## Recommended End-to-End Order

1. Build/clean metadata table
2. Run `cp2h5ad` for each plate
3. Merge h5ad files
4. Optional outlier/problematic-well filtering
5. Robust DMSO-based normalization
6. Unsupervised + correlation + replicate SNR feature selection
7. Save interpretable layer
8. ZCA whitening
9. PCA/Harmony/UMAP
10. MoA plotting + phenotypic distance ranking

---

## Practical Notes & Caveats

- Always ensure each batch has enough DMSO controls before robust normalization.
- Perform whitening **after** feature reduction for speed/stability.
- Don’t interpret per-feature biological meaning from heavily transformed/whitened matrices.
- Keep versioned layers (`raw`, `normalized`, `pre_whitening`) for reproducibility.
- For reproducible comparisons, keep matched drug/control plate context (`PlateSet_DrugCode`).

---

## Minimal Skeleton Script

```python
# 0) load merged filtered adata
adata = ad.read_h5ad('merged_filtered.h5ad')
adata.X = adata.layers['raw'].copy()

# 1) normalization
adata = robust_zscore_normalize(adata)

# 2) feature selection
adata = unsupervised_feature_reduction(adata, variance_threshold=1e-2)
adata = variance_based_correlation_reduction(adata, threshold=0.9)
adata = feature_selection_by_replicates(adata, quantile_threshold=0.8)

# 3) keep interpretable layer
adata.layers['normalized'] = adata.X.copy()

# 4) whitening
adata = zca_whitening(adata)

# 5) integration + embedding
sc.tl.pca(adata)
import scanpy.external as sce
sce.pp.harmony_integrate(adata, 'Batch', max_iter_harmony=20)
sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=15)
sc.tl.umap(adata)

# 6) downstream analyses
visualize_drug_mechanism(adata, '13A_Drug88', '13A_DMSO', layer='normalized', top_n=5)
adata.obs['distance_from_control'] = np.linalg.norm(adata.X, axis=1)
```

---

If you want, this tutorial can be converted directly into a notebook template (`.ipynb`) with runnable cells for the `hoondy/CPTools` repository.
