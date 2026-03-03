# API Reference

This page provides a comprehensive reference for the CPTools API, organized by module.

- [Data I/O (`cpt.io`)](#data-io-cptio)
- [Preprocessing (`cpt.pp`)](#preprocessing-cptpp)
- [Tools and Visualization (`cpt.tl`)](#tools-and-visualization-cpttl)

---

## Data I/O (`cpt.io`)

### `read_harmony`
`cpt.io.read_harmony(plate_results_path, schema=None, batch=None, cell_type=None, feature_prefix='Nuclei Selected - ', control_value='DMSO', schema_path=None)`

Read Harmony output and a master schema table into a single AnnData object.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plate_results_path` | `str \| Path \| Sequence[str \| Path]` | | Path (or list of paths) to Harmony `PlateResults.txt` files. |
| `schema` | `str \| Path \| None` | `None` | Path to pre-processed metadata table containing at least `Batch`, `Row`, `Column`. |
| `batch` | `str \| Sequence[str] \| None` | `None` | Batch id (or list of batch ids aligned to `plate_results_path`). Required when reading multiple plate files. |
| `cell_type` | `str \| None` | `None` | Optional cell type annotation to inject into `adata.obs["CellType"]`. |
| `feature_prefix` | `str` | `'Nuclei Selected - '` | Prefix used to identify morphology feature columns. |
| `control_value` | `str` | `'DMSO'` | Label used for untreated controls. |
| `schema_path` | `str \| Path \| None` | `None` | **Deprecated** alias for `schema`. |

**Returns**
- `anndata.AnnData`: An AnnData object containing the merged plate results and metadata. The raw features are stored in `adata.layers["raw"]`.

**Notes**
- The schema file must contain `Batch`, `Row`, and `Column` columns to match with the Harmony results.
- If `Treatment` is not in the schema, it will be inferred from `Drug_name` or `Drug_code` if available.
- Features are automatically renamed using `clean_feature_name` to be more compact.

**Example**
```python
import CPTools as cpt
adata = cpt.io.read_harmony(
    plate_results_path=["Plate01_Results.txt", "Plate02_Results.txt"],
    schema="metadata.csv",
    batch=["P01", "P02"]
)
```

---

## Preprocessing (`cpt.pp`)

### `robust_zscore_norm`
`cpt.pp.robust_zscore_norm(adata, batch_key='Batch', treatment_key='Treatment', control_value='DMSO', epsilon=1e-06, source_layer='raw', inplace=True)`

Robust Z-score normalization per batch using control wells.

**Formula**
`X_norm = (X - median(controls)) / (MAD(controls) * 1.4826 + epsilon)`

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to normalize. |
| `batch_key` | `str` | `'Batch'` | Key in `adata.obs` identifying batches. |
| `treatment_key` | `str` | `'Treatment'` | Key in `adata.obs` identifying treatments. |
| `control_value` | `str` | `'DMSO'` | Value in `treatment_key` column representing controls. |
| `epsilon` | `float` | `1e-06` | Small constant to prevent division by zero. |
| `source_layer` | `str \| None` | `'raw'` | Layer to use as input. If `None`, uses `adata.X`. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The normalized AnnData object.

**Notes**
- If a batch has no controls, a warning is issued and the raw values are kept for that batch.
- Normalization parameters are stored in `adata.uns["cptools"]["robust_zscore_norm"]`.

---

### `blocklist_filter`
`cpt.pp.blocklist_filter(adata, keywords=DEFAULT_BLOCKLIST_KEYWORDS, subset=False, inplace=True)`

Mark or drop features containing known technical artifact keywords.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `keywords` | `Sequence[str]` | `DEFAULT_BLOCKLIST_KEYWORDS` | List of substrings to filter out. Default includes: "Manders", "RWC", "Location", "Granularity", "Execution", "Euler". |
| `subset` | `bool` | `False` | If `True`, physically removes the features. Otherwise, only marks them in `adata.var["pass_blocklist"]`. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The filtered AnnData object.

---

### `nan_filter`
`cpt.pp.nan_filter(adata, source_layer=None, subset=False, inplace=True)`

Mark or drop features with non-finite values (NaN or Inf) in any well.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `source_layer` | `str \| None` | `None` | Layer to check for NaNs. If `None`, uses `adata.X`. |
| `subset` | `bool` | `False` | If `True`, physically removes the features. Otherwise, only marks them in `adata.var["pass_non_nan"]`. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The filtered AnnData object.

---

### `variance_filter`
`cpt.pp.variance_filter(adata, threshold=0.01, source_layer=None, subset=False, inplace=True)`

Mark or drop near-constant features.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `threshold` | `float` | `0.01` | Variance threshold. Features with variance <= threshold are flagged/dropped. |
| `source_layer` | `str \| None` | `None` | Layer to use for variance calculation. |
| `subset` | `bool` | `False` | If `True`, physically removes the features. Otherwise, only marks them in `adata.var["pass_variance"]`. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The filtered AnnData object.

---

### `correlation_filter`
`cpt.pp.correlation_filter(adata, threshold=0.9, source_layer=None, subset=False, inplace=True)`

Mark or drop highly correlated features, keeping the representative with the highest variance.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `threshold` | `float` | `0.9` | Absolute correlation threshold (0 to 1). |
| `source_layer` | `str \| None` | `None` | Layer to use for correlation calculation. |
| `subset` | `bool` | `False` | If `True`, physically removes the features. Otherwise, only marks them in `adata.var["pass_correlation"]`. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The filtered AnnData object.

---

### `snr_feature_selection`
`cpt.pp.snr_feature_selection(adata, treatment_key='Treatment', control_value='DMSO', keep_top_fraction=0.2, quantile_threshold=None, min_replicates=2, noise_aggregation='pooled', source_layer=None, subset=False, inplace=True)`

Select features by replicate signal-to-noise ratio (SNR).

**Method**
- **Signal**: Variance of treatment means (excluding controls).
- **Noise**: Within-treatment (replicates) variance, aggregated across groups.
- **SNR**: `Signal / (Noise + 1e-9)`

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to process. |
| `treatment_key` | `str` | `'Treatment'` | Key in `adata.obs` identifying treatments. |
| `control_value` | `str` | `'DMSO'` | Value identifying controls. |
| `keep_top_fraction` | `float` | `0.2` | Fraction of top SNR features to keep (used if `quantile_threshold` is `None`). |
| `quantile_threshold` | `float \| None` | `None` | SNR quantile threshold (e.g., 0.8 excludes bottom 80%). |
| `min_replicates` | `int` | `2` | Minimum replicates required for a treatment group to be included. |
| `noise_aggregation` | `str` | `'pooled'` | Method to aggregate within-group variances: `'pooled'`, `'mean'`, or `'median'`. |
| `source_layer` | `str \| None` | `None` | Layer to use for calculations. |
| `subset` | `bool` | `False` | If `True`, physically removes non-selected features. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The AnnData object with SNR annotations.

**Notes**
- Annotates `adata.var["replicate_snr"]`, `adata.var["highly_variable"]`, and `adata.var["highly_variable_rank"]`.

---

### `zca_whiten`
`cpt.pp.zca_whiten(adata, treatment_key='Treatment', control_value='DMSO', epsilon=1e-05, source_layer=None, inplace=True)`

Apply control-based regularized ZCA whitening.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to whiten. |
| `treatment_key` | `str` | `'Treatment'` | Key in `adata.obs` identifying treatments. |
| `control_value` | `str` | `'DMSO'` | Value identifying controls. |
| `epsilon` | `float` | `1e-05` | Regularization parameter for the covariance matrix. |
| `source_layer` | `str \| None` | `None` | Layer to use as input. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Returns**
- `anndata.AnnData`: The whitened AnnData object (updates `adata.X`).

---

### `funnel`
`cpt.pp.funnel(adata, batch_key='Batch', treatment_key='Treatment', control_value='DMSO', variance_threshold=0.01, corr_threshold=0.9, snr_threshold=0.8, snr_keep_top_fraction=None, subset=False, verbose=True, inplace=True)`

A comprehensive feature filtering and selection pipeline.

**Pipeline Steps**
1. `robust_zscore_norm`
2. `blocklist_filter` (subsetting in temporary copy)
3. `nan_filter` (subsetting in temporary copy)
4. `variance_filter` (subsetting in temporary copy)
5. `correlation_filter` (subsetting in temporary copy)
6. `snr_feature_selection`

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to process. |
| `batch_key` | `str` | `'Batch'` | Key for batches. |
| `treatment_key` | `str` | `'Treatment'` | Key for treatments. |
| `control_value` | `str` | `'DMSO'` | Key for controls. |
| `variance_threshold` | `float` | `0.01` | Interpreted as a variance quantile to drop. |
| `corr_threshold` | `float` | `0.9` | Correlation threshold for `correlation_filter`. |
| `snr_threshold` | `float \| None` | `0.8` | SNR quantile threshold (fraction to exclude from bottom). |
| `snr_keep_top_fraction` | `float \| None` | `None` | **Deprecated** alias for `1 - snr_threshold`. |
| `subset` | `bool` | `False` | If `True`, physically subsets to highly variable features. |
| `verbose` | `bool` | `True` | Whether to print progress and counts. |
| `inplace` | `bool` | `True` | Whether to update in place. |

**Returns**
- `anndata.AnnData`: The processed AnnData object.

---

## Tools and Visualization (`cpt.tl`)

### `scatter`
`cpt.tl.scatter(adata, color=None, use_rep='X_umap', width=800, height=800, wspace=0.1, legend=True, title=None, show=True, **kwargs)`

Plot a 2D embedding from `adata.obsm` using Plotly.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to plot. |
| `color` | `str \| Sequence[str] \| None` | `None` | Column name(s) in `adata.obs` to color by. |
| `use_rep` | `str` | `'X_umap'` | Key in `adata.obsm` containing the 2D coordinates. |
| `width`, `height` | `int` | `800` | Figure dimensions. |
| `wspace` | `float` | `0.1` | Horizontal spacing between subplots (if multiple colors provided). |
| `legend` | `bool` | `True` | Whether to show the legend. |
| `title` | `str \| None` | `None` | Plot title. |
| `show` | `bool` | `True` | If `True`, calls `fig.show()`. Otherwise, returns the Figure object. |

**Returns**
- `plotly.graph_objects.Figure \| None`: The Figure object if `show=False`.

---

### `treatment_vectors`
`cpt.tl.treatment_vectors(adata, treatment_key='Treatment', control_value='DMSO', batch_key='Batch', layer='normalized', use_rep=None, treatments=None, use_highly_variable=False)`

Compute batch-matched control -> treatment vectors.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | Input AnnData. |
| `treatment_key` | `str` | `'Treatment'` | Column identifying treatments. |
| `control_value` | `str` | `'DMSO'` | Column identifying controls. |
| `batch_key` | `str` | `'Batch'` | Column identifying batches. |
| `layer` | `str \| None` | `'normalized'` | Layer to use for features. |
| `use_rep` | `str \| None` | `None` | If provided, uses this representation from `adata.obsm` instead of features. |
| `treatments` | `str \| Sequence[str] \| None` | `None` | Specific treatments to compute vectors for. |
| `use_highly_variable` | `bool` | `False` | Whether to only use features marked `highly_variable`. |

**Returns**
- `pd.DataFrame`: A DataFrame where rows are treatments and columns are features (or embedding components).

---

### `rank_treatment_correlations`
`cpt.tl.rank_treatment_correlations(vectors, treatment, method='spearman', top_n=10, bottom_n=0, legend=True, show=True)`

Rank correlations between one treatment vector and all other treatment vectors.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vectors` | `pd.DataFrame` | | Output from `cpt.tl.treatment_vectors`. |
| `treatment` | `str` | | Treatment to compare against. |
| `method` | `str` | `'spearman'` | Correlation method: `'spearman'` or `'pearson'`. |
| `top_n`, `bottom_n` | `int` | `10`, `0` | Number of top/bottom correlations to display. |
| `show` | `bool` | `True` | Whether to show the Plotly bar chart. |

**Returns**
- `pd.DataFrame`: Ranked correlation table.

---

### `umap_treatment_arrows`
`cpt.tl.umap_treatment_arrows(adata, treatment, treatment_key='Treatment', control_value='DMSO', batch_key='Batch', use_rep='X_umap', legend=True, width=1000, height=800, show=True)`

Visualize control -> treatment arrows on a 2D embedding (e.g., UMAP).

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | Input AnnData. |
| `treatment` | `str \| Sequence[str]` | | Treatment(s) to draw arrows for. |
| `use_rep` | `str` | `'X_umap'` | Embedding to use. |

**Returns**
- `plotly.graph_objects.Figure \| None`: The Figure object if `show=False`.

---

### `visualize_drug_effect`
`cpt.tl.visualize_drug_effect(adata, treatment, treatment_key='Treatment', control_value='DMSO', batch_key='Batch', layer='normalized', top_n=5, qvalue_threshold=0.05, effect_threshold=0.0, legend=True, show=True)`

Generate volcano plot and boxplots for treatment(s) vs matched controls.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | Input AnnData. |
| `treatment` | `str \| Sequence[str]` | | Treatment(s) to analyze. |
| `layer` | `str \| None` | `'normalized'` | Layer to use for stats and plotting. |
| `top_n` | `int` | `5` | Number of top hits to show in boxplots. |
| `qvalue_threshold`| `float` | `0.05` | FDR threshold for the volcano plot. |

**Returns**
- `pd.DataFrame`: Table of top hits with Feature, Effect Size, P-value, and Adjusted P-value.

---

### `visualize_drug_effect_rescue`
`cpt.tl.visualize_drug_effect_rescue(adata, treatment, rescue, treatment_key='Treatment', control_value='DMSO', batch_key='Batch', layer='normalized', top_n=5, qvalue_threshold=0.05, effect_threshold=0.0, legend=True, show=True)`

Generate treatment-vs-control volcano plot and boxplots including rescue groups.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | Input AnnData. |
| `treatment` | `str \| Sequence[str]` | | Main treatment(s) for the statistical test. |
| `rescue` | `str \| Sequence[str]` | | Rescue treatment(s) to overlay on boxplots. |

**Returns**
- `pd.DataFrame`: Table of top hits (treatment vs control).
