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

### Shared-reference normalization

`cpt.pp.reference_normalize(adata, *, reference_query, batch_key='Batch', source_layer='raw', output_layer='normalized', inplace=True, **fit_options)`

Plate-specific reference centering and pooled within-plate reference SD, with
optional regularized positive-control gains. Use explicit WT reference roles
when WT and mutant DMSO wells coexist. The separate
`fit_reference_normalization` / `apply_reference_normalization` functions support
held-out validation. See [full signatures, formulas and diagnostics](reference_normalization.md).

`cpt.pp.robust_reference_normalize(adata, *, reference_query, batch_key='Batch', source_layer='raw', output_layer='normalized', inplace=True, **fit_options)`

Robust variant for few reference wells per plate
(`fit_robust_reference_normalization`): plate offsets estimated jointly from
the reference and optional `anchor_queries` (conditions repeated on every
plate), shrunk by James-Stein toward the across-plate reference mean when they
are not larger than their noise (including plate × condition interaction), and
a choice of scale (`"reference"`, `"anchors"`, or `"replicate"`: within-group
SD of replicate groups given by `scale_groupby`). The model is applied with
`apply_reference_normalization`. Single-plate designs use
`plate_shrinkage=False` with `scale="replicate"`.

Related analysis functions: `cpt.tl.reference_control_cv` (pass
`robust_options` to add the robust centering, evaluated leave-condition-out)
and `cpt.tl.rescue_scores` (now with noise-debiased columns), documented in the
same guide.

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
`cpt.pp.blocklist_filter(adata, keywords=None, blocklist_path=None, blocklist_key='default', subset=False, inplace=True)`

Mark or drop known-noisy CellProfiler features. By default, CPTools uses the
packaged `default_blocklists.yaml` curated blocklist and exact feature-name
matching. Pass `keywords` only for backward-compatible substring filtering.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `keywords` | `Sequence[str] \| None` | `None` | Optional backward-compatible substring filters. If provided, this overrides YAML blocklist matching. |
| `blocklist_path` | `str \| Path \| None` | `None` | Optional YAML file. If `None`, uses CPTools' packaged `default_blocklists.yaml`. |
| `blocklist_key` | `str` | `'default'` | Top-level YAML key containing the feature-name blocklist. |
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

### `nan_obs_filter`
`cpt.pp.nan_obs_filter(adata, max_nan_fraction=0.2, source_layer=None, subset=False, inplace=True)`

Mark or drop observations with too many non-finite feature values.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `max_nan_fraction` | `float` | `0.2` | Maximum allowed fraction of non-finite feature values per observation. |
| `source_layer` | `str \| None` | `None` | Layer to check for NaNs. If `None`, uses `adata.X`. |
| `subset` | `bool` | `False` | If `True`, physically removes observations. Otherwise, only marks them in `adata.obs["pass_non_nan"]`. |
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

### `control_agreement_filter`
`cpt.pp.control_agreement_filter(adata, *, copies, batch_key=None, source_layer=None, alpha=0.05, subset=False, inplace=True)`

Flag features on which two copies of the same control condition disagree
beyond well noise, e.g. dedicated vs scattered WT+DMSO wells. Uses control
wells only, so no genotype or treatment effect enters the selection.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to filter. |
| `copies` | `Mapping[str, tuple[str, str]]` | | Name → two `obs.query` selections of the same condition in different well sets. |
| `batch_key` | `str \| None` | `None` | Compare the copies within each batch where both are present, so plate offsets cancel. With `None`, the queries must restrict the comparison themselves. |
| `source_layer` | `str \| None` | `None` | Layer to use (`None` = `adata.X`). Raw and normalized input give the same result within a batch. |
| `alpha` | `float` | `0.05` | Features with Benjamini–Hochberg q below `alpha` fail. |
| `subset` | `bool` | `False` | If `True`, removes failing features. Otherwise only marks them. |
| `inplace` | `bool` | `True` | Whether to update the AnnData object in place. |

**Method.** For each comparison k: z_k = (mean_b − mean_a) / (s·√(1/n_a + 1/n_b)),
with s the feature's pooled within-copy SD. The statistic is mean_k(z_k²),
about 1 when copies differ only by noise, tested against F(K, df).

**Writes** `var["control_agreement_stat"]`, `var["control_agreement_q"]`,
`var["pass_control_agreement"]`, and `uns["cptools"]["control_agreement_filter"]`,
including `median_abs_z_ratio` per comparison. A ratio near 1 means copies agree in
typical features. A ratio well above 1 (a warning is raised above 1.5) means the
disagreement is broad: dropping features will not remove it, so correct it instead.

**Example**
```python
copies = {
    "DMSO": ("line == 'WT' and Drug_code == 'c_DM'", "line == 'WT' and Drug_code == 'c_DM_rand'"),
    "cells_only": ("line == 'WT' and Drug_code == 'c'", "line == 'WT' and Drug_code == 'c_rand'"),
}
cpt.pp.control_agreement_filter(adata, copies=copies, batch_key="Batch", source_layer="raw")
keep = adata.var["pass_control_agreement"]
```

---

### `snr_feature_selection`
`cpt.pp.snr_feature_selection(adata, treatment_key='Treatment', control_value='DMSO', keep_top_fraction=0.2, quantile_threshold=None, min_replicates=2, noise_aggregation='pooled', source_layer=None, subset=False, inplace=True, replicate_groupby=None)`

`replicate_groupby` adds metadata keys to treatment when defining replicates, e.g.
`['Batch', 'line', 'Concentration_uM']`. Missing doses form explicit groups.

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
`cpt.pp.funnel(adata, batch_key='Batch', treatment_key='Treatment', control_value='DMSO', source_layer=None, obs_nan_threshold=0.2, variance_threshold=0.01, corr_threshold=0.9, snr_threshold=0.6, snr_keep_top_fraction=None, subset=False, verbose=True, inplace=True, replicate_groupby=None, snr_groups=None, snr_key=None)`

The SNR step requires a contrast. `snr_groups` lists the values of `snr_key` (default `treatment_key`) whose wells define the SNR, and all of them count as signal. For example, `snr_key="line", snr_groups=["WT", "MUT"]` ranks features by a genotype contrast; restrict the wells first if needed, e.g. controls only. `snr_groups="all"` uses every condition with `control_value` excluded from the signal, the previous default. With `snr_threshold` set and no `snr_groups`, `funnel` raises. Ranking by all conditions favors whatever dominates the variance, often strong treatments, rather than the contrast being tested. `snr_threshold` is the fraction excluded from the bottom; the default 0.6 keeps the top 40%.

Does not normalize or whiten. `replicate_groupby` is forwarded to SNR selection.

A comprehensive feature filtering and selection pipeline.

By default, filtering is calculated from the current `adata.X`. Pass
`source_layer="normalized"` or another layer name to calculate filters from a
specific layer without replacing `adata.X`.

**Pipeline Steps**
1. `blocklist_filter` (subsetting in temporary copy)
2. `nan_obs_filter` (subsetting bad observations in temporary copy; skipped if `obs_nan_threshold=None`)
3. `nan_filter` (subsetting features in temporary copy)
4. `variance_filter` (subsetting in temporary copy; skipped if `variance_threshold=None`)
5. `correlation_filter` (subsetting in temporary copy; skipped if `corr_threshold=None`)
6. `snr_feature_selection` (skipped if `snr_threshold=None`)

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `anndata.AnnData` | | The AnnData object to process. |
| `batch_key` | `str` | `'Batch'` | Key for batches. |
| `treatment_key` | `str` | `'Treatment'` | Key for treatments. |
| `control_value` | `str` | `'DMSO'` | Key for controls. |
| `source_layer` | `str \| None` | `None` | Matrix source for filtering; `None` uses current `adata.X`. |
| `obs_nan_threshold` | `float \| None` | `0.2` | Drop observations with non-finite fraction above this threshold before feature-level NaN filtering; `None` skips observation filtering. |
| `variance_threshold` | `float \| None` | `0.01` | Interpreted as a variance quantile to drop; `None` skips `variance_filter`. |
| `corr_threshold` | `float \| None` | `0.9` | Correlation threshold for `correlation_filter`; `None` skips `correlation_filter`. |
| `snr_threshold` | `float \| None` | `0.6` | SNR quantile threshold (fraction to exclude from bottom); `None` skips `snr_feature_selection`. |
| `snr_groups` | `str \| Sequence[str] \| None` | `None` | Required when `snr_threshold` is set: values of `snr_key` defining the SNR contrast, or `"all"`. |
| `snr_key` | `str \| None` | `None` | Column whose values `snr_groups` refers to; defaults to `treatment_key`. |
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

---

### Group geometry: `group_separation`, `profile_shift`, `feature_contrasts`

Groups are named `obs.query` selections, e.g.
`groups = {"WT_DMSO": "line == 'WT' and Treatment == 'DMSO'", ...}`.
All three require a common finite feature set (`features=`) in a normalized
layer. Squared distances and dot products are corrected for well-sampling
noise (trace of each group's covariance / n).

`cpt.tl.group_separation(adata, *, groups, pairs, layer='normalized', features=None, n_perm=2000, random_state=0)`

One row per `(a, b)` pair and a `combined` row: debiased squared centroid
distance per feature, signal-to-noise, permutation p-value (labels permuted
within each pair), leave-one-well-out nearest-centroid accuracy, and transfer
accuracy (centroids trained on the other pairs). Give pairs in a consistent
orientation, e.g. `[("WT_untreated", "MUT_untreated"), ("WT_DMSO", "MUT_DMSO")]`.

`cpt.tl.profile_shift(adata, *, groups, axis, shift, toward=None, layer='normalized', features=None, n_boot=2000, random_state=0)`

Returns a Series with the debiased `projection` S·A/|A|² of the shift
`S = shift[1] - shift[0]` on the axis `A = axis[1] - axis[0]`, its delta-method
SE, t interval and `p_projection`, `cosine_debiased` (and plain `cosine`), shift
and orthogonal lengths relative to |A|, `remaining_distance` to `toward`, and
axis/shift SNR. Non-projection intervals are basic bootstrap intervals over
wells. Prefer axis and shift groups that share no wells.

`cpt.tl.feature_contrasts(adata, *, groups, contrasts, layer='normalized', features=None)`

Per-feature Welch tests for `{name: (a, b)}` contrasts (b − a): difference,
Cohen's d, t, p and BH q within each contrast, as a long table.

---

### Hit calling and empirical nulls: `replicate_map`, `pseudo_treatment_null`, `empirical_pvalues`

`cpt.tl.replicate_map(adata, *, treatment_key='Treatment', control_value='DMSO', batch_key='Batch', annotation_key=None, treatments=None, layer='normalized', features=None, n_perm=1000, random_state=0)`

Replicate retrieval by mean average precision (mAP), as in copairs
(Kalinin et al. 2024). Each well of a treatment is a query. It ranks the
treatment's other wells (positives, on any plate) against the control wells of
its own plate (negatives) by cosine similarity. Average precision summarizes
one query's ranking, and mAP averages it over the treatment's wells. On each
plate, the treatment's wells and the controls are centered on their joint mean
before cosine similarity.

One row per treatment: `n_wells`, `n_batches`, `n_controls`, `mAP`,
`null_mean` (expected mAP under no effect), `p_value`, BH `q_value`, `p_min`
(the smallest attainable p-value), `n_perm` and `exact`.

- **Null.** Treatment and control labels are shuffled within each plate, keeping
  the per-plate counts, and mAP is recomputed. When there are at most `n_perm`
  distinct labelings, all of them are enumerated and the p-value is exact.
  The test is valid when treatment and control wells of a plate are
  exchangeable; controls in fixed positions (e.g. one column) break that.
- **Centering.** The joint mean does not depend on which wells are labeled
  treated, so the label shuffle stays exact. Any per-plate shift already in the
  layer cancels. Centering on the controls alone, as a layer centered on DMSO
  per plate does, gives a treatment's wells a shared offset (the noise of the
  control mean), and inert replicates then retrieve each other. In simulations
  with 8 controls and triplicates, a null of random ranks (as in copairs) then
  gave p < 0.05 for 8%, 28% and 92% of inert treatments at 20, 100 and 500
  features. The label shuffle here gave 3–7%. Per-plate scales estimated from
  the controls (robust z-scores) are not undone.
- **Resolution.** Triplicates and 8 controls on one plate give C(11, 3) = 165
  labelings, so `p_min` = 1/165 ≈ 0.006. Clear hits reach 1/165 to about
  4/165. With thousands of compounds, BH q < 0.05 therefore needs a large
  fraction of active compounds. With this layout, rank on mAP and use
  p < 0.05 as a filter. 16 controls per plate raise the number of labelings
  to C(19, 3) = 969.
- **Consistency** (`annotation_key="MOA"`): each treatment's consensus profile
  (its mean well, centered on the mean of all analyzed wells on its plate)
  retrieves treatments with the same annotation against all other annotated
  treatments. The result has one row per annotation value shared by at least
  2 treatments. The null draws random treatment groups of the same size.
  Treatments without an annotation are dropped. Restrict the analysis to
  phenotypically active treatments with `treatments=`. Centering on the
  plate's controls instead would make treatments on one plate retrieve each
  other: with no MOA effect and each MOA on a single plate, 85–100% of MOAs
  had p < 0.05 in simulation, against none with the plate mean. Annotations
  that cluster on plates remain confounded with plate.

Hit calling in a drug screen, with one row per treatment:

```python
hv = adata.var_names[adata.var["highly_variable"].to_numpy()]
activity = cpt.tl.replicate_map(adata, treatment_key="Treatment", control_value="DMSO",
                                batch_key="Batch", layer="normalized", features=hv)
info = (adata.obs.astype({"Treatment": str}).drop_duplicates("Treatment")
        .set_index("Treatment")[["Target", "MOA"]])
activity = activity.join(info, on="treatment").sort_values(["p_value", "mAP"], ascending=[True, False])
active = activity.loc[activity["p_value"] < 0.05, "treatment"]
moa = cpt.tl.replicate_map(adata, annotation_key="MOA", treatments=active, layer="normalized", features=hv)
```

`cpt.tl.pseudo_treatment_null(adata, *, treatment_key='Treatment', control_value='DMSO', batch_key='Batch', n_per_group=3, n_draws=1000, groupby=None, query=None, include_query=None, min_remaining=2, random_state=0)`

This function draws random groups of `n_per_group` control wells from the same
plate as fake treatments ("DMSO pseudo-drugs"). It returns a new AnnData in
which every draw is a pseudo-plate `"<batch>|pseudo_<d>"`. The pseudo-plate
holds copies of all control wells of the source plate, and the drawn wells are
relabeled `"pseudo_<d>"` in `treatment_key`. Any score that compares a
treatment with controls on the same plate therefore runs unchanged on the
result, and each pseudo-drug is compared with the controls that remain on its
plate.

- `query` restricts which control wells may be drawn (e.g. the drug-screen
  plates). `groupby` draws within cells, e.g. plate × genotype. A cell needs
  `n_per_group + min_remaining` drawable wells. `include_query` copies further
  wells unchanged into each pseudo-plate, e.g. WT references labeled
  differently from the drawn vehicle wells.
- Draws are spread evenly over plates and are distinct within a plate. With 8
  controls per plate, a plate offers at most C(8, 3) = 56 triples. If fewer
  distinct triples exist than requested, all of them are used and a warning
  reports the smaller total.
- Added obs columns: `pseudo_draw`, `pseudo_role` (`"pseudo_treatment"` or
  `"control"`), `pseudo_source_batch` and `pseudo_source_well`. X, layers,
  obsm, var and varm are copied, and parameters go to
  `uns["pseudo_treatment_null"]`. The result writes to H5AD.
- A pseudo-drug has `n_per_group` fewer controls than a real drug, so magnitude
  scores of pseudo-drugs are slightly noisier. That makes the null
  conservative for those scores. Correlations are scale-free and unaffected.
- Pseudo-drugs carry no position, solvent or drug-well effects. If most inert
  compounds beat the pseudo-drugs, the shared component lives in the drug
  wells, not in the noise of the DMSO centroid.

`cpt.tl.empirical_pvalues(observed, null, *, alternative='greater')`

This helper takes observed scores and null scores from pseudo-treatments. With
n null scores, `"greater"` gives p = (1 + #{null ≥ score}) / (1 + n); `"less"`
uses null ≤ score; `"two-sided"` is min(1, 2 × the smaller one-sided p). It
returns `score`, `null_z`, `p_value` and BH `q_value`, indexed like
`observed`. The smallest attainable p is 1/(1 + n), and twice that for
two-sided tests. Pseudo-drugs from one plate share wells, so the effective
null size is smaller than n.

Connectivity ranking with a null. Here the disease vector comes from a mutant
line scored as a treatment (`MUT`) against WT wells labeled `DMSO` on its own
plates, and pseudo-drugs are drawn from the drug plates only:

```python
import pandas as pd

null = cpt.tl.pseudo_treatment_null(adata, treatment_key="Treatment", control_value="DMSO",
                                    batch_key="Batch", n_per_group=3, n_draws=1000,
                                    query="plate_type == 'drug'")
vectors = cpt.tl.treatment_vectors(adata, layer="normalized")
null_vectors = cpt.tl.treatment_vectors(null, layer="normalized")
ranked = cpt.tl.rank_treatment_correlations(pd.concat([vectors, null_vectors]),
                                            treatment="MUT", show=False)
score = ranked.set_index("treatment")["correlation"]
is_null = score.index.isin(null_vectors.index)
reversal = cpt.tl.empirical_pvalues(score[~is_null], score[is_null], alternative="less")
```

Rescue scores need pseudo-drugs drawn from mutant vehicle wells, within plate ×
genotype cells, and a comparison within each genotype. Here WT references are
WT + DMSO wells on every plate. If references carry their own label (e.g.
`Treatment == 'reference'`), pass that selection as `include_query` so they are
copied into each pseudo-plate:

```python
null = cpt.tl.pseudo_treatment_null(adata, treatment_key="Treatment", control_value="DMSO",
                                    groupby="line", query="line != 'WT'", n_draws=500)
roles = dict(reference_query="line == 'WT' and Treatment == 'DMSO'",
             disease_control_query="line != 'WT' and Treatment == 'DMSO'")
pseudo = cpt.tl.rescue_scores(null, treatment_query="pseudo_role == 'pseudo_treatment'", **roles)
real = cpt.tl.rescue_scores(adata, treatment_query="Treatment != 'DMSO'", **roles)
for line, rows in real.groupby("line"):
    reference = pseudo.loc[pseudo["line"] == line, "rescue_projection_debiased"]
    p = cpt.tl.empirical_pvalues(rows["rescue_projection_debiased"], reference)
```

Triplicate pseudo-drugs need at least 5 vehicle wells per genotype and plate.
With fewer, use a positional null instead, e.g. WT wells placed in screening
positions and scored as a pseudo-disease.
