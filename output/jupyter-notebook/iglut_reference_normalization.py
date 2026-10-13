# %% [markdown]
# # WT-reference normalization for the six-plate iGLUT screen
#
# Load measured well profiles, audit the shared WT references, compare control-only
# normalization using held-out wells, and calculate exploratory mutant rescue scores.
# Requires the updated CPTools checkout in this notebook's Python environment:
# `python -m pip install -e /path/to/CPTools` and `python -m pip install ipykernel`.
# Use your HPC kernel. This notebook does not contain results from your experiment.
#
# Workflow: metadata/QC → pooled WT normalization → held-out control checks →
# spatial diagnostics → feature selection and exploratory rescue ranking.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import plotly.express as px
from IPython import get_ipython
from IPython.display import display
from scipy import sparse
import CPTools as cpt

assert hasattr(cpt.pp, "reference_normalize"), "Install the updated CPTools checkout in this kernel."
print("CPTools loaded from:", cpt.__file__)

# %%
DATA_PATH = Path("260908_iGLUT_bitbio_ADPD_merged_clean.h5ad")
OUTPUT_DIR = Path("results/iglut_reference_normalization")
APPLY_POSITIVE_GAIN = False  # Change only after reviewing independent control diagnostics.
SHOW_PLOTS = get_ipython() is not None
SNR_EXCLUDE_FRACTION = None  # Optional: e.g. 0.5; examine sensitivity of disease/rescue signals.
EXCLUDE_WELLS = []  # Explicit (Batch, Well) tuples from image/segmentation QC.

REFERENCE = "line == 'WT' and Drug_code == 'c_DM'"
POSITIVES = {
    "Ivermectin_B1a": "line == 'WT' and Drug_code == 'c_Ivermectin'",
    "Selamectin": "line == 'WT' and Drug_code == 'c_Selamectin'",
}
DISEASE_CONTROL = "Drug_code == 'c_DM_rand'"

# %% [markdown]
# ## 1. Read the raw measurement layer and audit the design
# `X` may already contain processed values. Every normalization fit below explicitly
# uses `layers['raw']`. The dedicated reference roles are defined by `Drug_code`
# and `line`; selecting all DMSO wells would mix WT and mutant baselines.

# %%
adata = ad.read_h5ad(DATA_PATH)
required = {"Batch", "Well", "Row", "Column", "line", "Drug_code", "Treatment", "Concentration_uM"}
assert required <= set(adata.obs), f"Missing metadata: {required - set(adata.obs)}"
assert "raw" in adata.layers, "The raw measurement layer is required."
assert adata.obs_names.is_unique and adata.var_names.is_unique
assert not adata.obs.duplicated(["Batch", "Well"]).any(), "Expected one observation per well."
assert not adata.obs[["Batch", "Well", "line", "Drug_code", "Treatment"]].isna().any().any()
expected_batches = {"WT_CTRL", "APP_HET", "APP_HOM", "PSEN1_HOM", "PINK1_HOM", "PRKN_HOM"}
assert set(adata.obs["Batch"].astype(str)) == expected_batches
for code, treatment in {"c_DM": "DMSO", "c_Ivermectin": "Ivermectin_B1a", "c_Selamectin": "Selamectin"}.items():
    selected = adata.obs["Drug_code"].eq(code)
    assert adata.obs.loc[selected, "line"].eq("WT").all()
    assert adata.obs.loc[selected, "Treatment"].eq(treatment).all()
print(adata.shape, "well profiles × features")
display(adata.obs.groupby(["Batch", "line"], observed=True).size().rename("wells").to_frame())
control_codes = ["c_DM", "c_Ivermectin", "c_Selamectin", "c", "c_DM_rand", "c_rand"]
display(pd.crosstab(adata.obs["Batch"], adata.obs["Drug_code"]).reindex(columns=control_codes, fill_value=0))

# %% [markdown]
# ## 2. Apply basic numerical QC
# Add image-QC failures to EXCLUDE_WELLS above. Inspect cell counts, segmentation,
# and toxicity separately; this numerical filter does not validate image quality.
# Do not automatically regress out cell count: it may be biological.
# Features with nonfinite values are removed here; zero control variance is handled
# by the normalizer. These QC decisions do not use rescue ranks.

# %%
raw = adata.layers["raw"].toarray() if sparse.issparse(adata.layers["raw"]) else np.asarray(adata.layers["raw"])
manual_exclude = np.array([(str(b), str(w)) in set(EXCLUDE_WELLS)
                           for b, w in zip(adata.obs["Batch"], adata.obs["Well"])])
keep_wells = (np.isfinite(raw).mean(axis=1) >= 0.8) & ~manual_exclude
display(adata.obs.loc[~keep_wells, ["Batch", "Well", "line", "Treatment"]])
adata = adata[keep_wells].copy()
raw = raw[keep_wells]
cpt.pp.blocklist_filter(adata)
keep_features = np.isfinite(raw).all(axis=0) & adata.var["pass_blocklist"].to_numpy()
print("Keeping", int(keep_features.sum()), "of", adata.n_vars, "features after basic QC")
adata = adata[:, keep_features].copy()
del raw
assert adata.n_vars > 0, "No features survived QC. Inspect the original measurements."
reference_counts = adata.obs.query(REFERENCE).groupby("Batch", observed=True).size().reindex(sorted(expected_batches), fill_value=0)
display(reference_counts.rename("WT reference wells after QC").to_frame())
assert (reference_counts >= 2).all(), "Insufficient references; inspect the failed plate before analysis."

# %% [markdown]
# ## 3. Fit WT centering and pooled within-plate scaling
# Default: subtract each plate's mean WT+DMSO profile, then divide by pooled
# within-plate WT SD. There are 18 dedicated WT+DMSO wells before QC, giving 12
# residual degrees of freedom. The three dispersed WT+DMSO wells on WT_CTRL are
# reserved for a separate positional check. Median centering is an optional
# sensitivity analysis; the scale estimator remains pooled SD.
#
# Positive-control gains are a candidate correction only. Their bounds, ridge
# penalty and signal threshold below are starting settings, not fitted biological
# facts. Never optimize them using the rescue rankings.

# %%
baseline = cpt.pp.reference_normalize(
    adata, reference_query=REFERENCE, batch_key="Batch", source_layer="raw",
    center="mean", output_layer="normalized", inplace=False,
)
positive_counts = pd.DataFrame({name: adata.obs.query(query).groupby("Batch", observed=True).size()
                                for name, query in POSITIVES.items()}).reindex(sorted(expected_batches)).fillna(0)
can_fit_gain = (positive_counts >= 2).all().all()
gain_candidate = None
if can_fit_gain:
    gain_candidate = cpt.pp.reference_normalize(
        adata, reference_query=REFERENCE, positive_control_queries=POSITIVES,
        source_layer="raw", center="mean", gain_ridge=10.0,
        gain_bounds=(0.5, 2.0), min_positive_snr=2.0,
        output_layer="normalized", inplace=False,
    )
    gain_model = gain_candidate.uns["cptools"]["reference_normalize"]
    display(pd.DataFrame({
        "Batch": gain_model["batches"],
        "WT_reference_n": gain_model["reference_counts"],
        "features_with_supported_gain": gain_model["gain_applied"].sum(axis=1),
        "median_gain": np.median(gain_model["gains"], axis=1),
    }))
else:
    print("Positive gain fitting unavailable after QC; baseline normalization remains available.")
    assert not APPLY_POSITIVE_GAIN
print("Invalid baseline features:", int((~baseline.var["pass_reference_normalization"]).sum()))

# %% [markdown]
# ## 4. Hold out one well per condition per plate
# With three replicates, each fold fits two and tests one. All fitted quantities
# exclude test wells. Compare methods in the same training-SD units. Lower held-out
# plate dispersion is useful only if positive-control effect magnitude is retained.
# Folds share training wells: their variability is not an independent-replicate CI.
# The three reference conditions are evaluated separately, not pooled together.

# %%
conditions = {"reference": REFERENCE, **POSITIVES}
counts = pd.DataFrame({name: adata.obs.query(query).groupby("Batch", observed=True).size()
                       for name, query in conditions.items()}).reindex(sorted(expected_batches)).fillna(0)
cv = pd.DataFrame()
if (counts >= 3).all().all():
    cv = cpt.tl.reference_control_cv(
        adata, reference_query=REFERENCE, positive_control_queries=POSITIVES,
        source_layer="raw", n_splits=3, random_state=7,
        gain_ridge=10.0, gain_bounds=(0.5, 2.0), min_positive_snr=2.0,
    )
    display(cv.groupby(["condition", "method"])[["plate_dispersion_rms", "mean_effect_rms", "n_features"]].mean())
    fig = px.strip(cv, x="method", y="plate_dispersion_rms", color="method", facet_col="condition",
                   hover_data=["fold", "n_features", "mean_effect_rms"],
                   title="Held-out reference agreement in common training-SD units")
    if SHOW_PLOTS:
        fig.show()
else:
    print("Three-fold control validation unavailable after QC: fewer than three wells in a reference group.")
    display(counts)
    assert not APPLY_POSITIVE_GAIN, "Keep gain correction disabled without an independent validation plan."

# %% [markdown]
# ## 5. Leave an entire positive compound out of gain fitting
# This additionally tests whether a correction learned from one positive transfers
# to the other. Unlike the replicate CV above, the test compound never fits the
# gains. Compare dispersion and effect magnitude in the baseline pooled-SD units.
# Two related positives provide limited coverage of morphology; this check cannot
# establish correction of every biological direction.

# %%
base_model = baseline.uns["cptools"]["reference_normalize"]
X = adata.layers["raw"].toarray() if sparse.issparse(adata.layers["raw"]) else np.asarray(adata.layers["raw"])
transfer_records = []
for test_name, test_query in POSITIVES.items():
    if not can_fit_gain:
        break
    train_positives = {name: query for name, query in POSITIVES.items() if name != test_name}
    fitted = cpt.pp.fit_reference_normalization(
        adata, reference_query=REFERENCE, positive_control_queries=train_positives,
        source_layer="raw", gain_ridge=10.0,
    )
    valid = base_model["valid_features"] & fitted["valid_features"]
    test_ids = adata.obs.query(test_query).index
    for method in ["wt_centered", "wt_positive_gain"]:
        profiles = []
        for p, plate in enumerate(fitted["batches"]):
            mask = adata.obs["Batch"].eq(plate) & adata.obs.index.isin(test_ids)
            effect = X[mask][:, valid].mean(axis=0) - fitted["centers"][p, valid]
            if method == "wt_positive_gain":
                effect = effect / fitted["gains"][p, valid]
            profiles.append(effect / base_model["pooled_sd"][valid])
        profiles = np.asarray(profiles)
        transfer_records.append({
            "held_out_compound": test_name, "method": method,
            "plate_dispersion_rms": np.sqrt(np.mean((profiles - profiles.mean(axis=0))**2)),
            "mean_effect_rms": np.sqrt(np.mean(profiles.mean(axis=0)**2)),
            "n_features": int(valid.sum()),
        })
transfer = pd.DataFrame(transfer_records, columns=["held_out_compound", "method", "plate_dispersion_rms", "mean_effect_rms", "n_features"])
display(transfer)
del X

# %% [markdown]
# ## 6. Choose a normalization and check control placement
# The default remains WT centering plus pooled scaling. Enable APPLY_POSITIVE_GAIN
# only after the held-out comparisons and feature-level inspection support it.
# Check whether control position changes the profile on the WT plate. This uses
# both copies of each treatment on WT_CTRL, not different genotypes.
# A difference can reflect position or handling. Do not automatically extrapolate
# a spatial correction from one WT plate to every mutant plate.

# %%
normalized = gain_candidate if APPLY_POSITIVE_GAIN else baseline
print("Selected:", "WT + positive-control gain" if APPLY_POSITIVE_GAIN else "WT centering + pooled SD")
analysis = normalized[:, normalized.var["pass_reference_normalization"].to_numpy()].copy()
placement_records = []
for name, dedicated, dispersed in [
    ("DMSO", "c_DM", "c_DM_rand"),
    ("Ivermectin_B1a", "c_Ivermectin", "Drug4"),
    ("Selamectin", "c_Selamectin", "Drug77"),
    ("cells_only", "c", "c_rand"),
]:
    a = analysis.obs["Batch"].eq("WT_CTRL") & analysis.obs["Drug_code"].eq(dedicated)
    b = analysis.obs["Batch"].eq("WT_CTRL") & analysis.obs["Drug_code"].eq(dispersed)
    delta = analysis.X[b].mean(axis=0) - analysis.X[a].mean(axis=0) if a.any() and b.any() else np.full(analysis.n_vars, np.nan)
    placement_records.append({"condition": name, "column23_n": int(a.sum()), "screen_n": int(b.sum()),
                              "profile_shift_rms": np.sqrt(np.mean(delta**2))})
placement = pd.DataFrame(placement_records)
display(placement)

# %% [markdown]
# ## 7. Inspect a feature across plates and control roles
# Choose a biologically meaningful feature below after reviewing its measurements.
# Here the first retained feature is used only to make the example runnable.
# WT controls need not form a perfect cluster: held-out variability should remain.

# %%
FEATURE = analysis.var_names[0]
j = analysis.var_names.get_loc(FEATURE)
control_view = analysis.obs["Drug_code"].isin(["c_DM", "c_DM_rand", "c_Ivermectin", "c_Selamectin"])
view = analysis.obs.loc[control_view, ["Batch", "Well", "line", "Treatment", "Drug_code"]].copy()
view["normalized_value"] = analysis.X[control_view, j]
fig = px.box(view, x="Batch", y="normalized_value", color="Drug_code", points="all",
             hover_data=["Well", "line", "Treatment"], title=f"Reference and disease controls: {FEATURE}")
if SHOW_PLOTS:
    fig.show()

# %% [markdown]
# ## 8. Select a common feature space and calculate exploratory rescue scores
# Use screening wells for feature selection. Genotype, plate and dose define
# replicate groups; the same drug in different genotypes is not one replicate group.
# SNR selection is off initially because selecting only drug-responsive features
# may discard disease features. Compare rankings if enabling it later.
#
# For each genotype separately: R=1 reverses the disease vector along its axis;
# remaining_distance=0 also requires reaching WT in the full selected space.
# orthogonal_shift reports unrelated changes. Require a reproducible disease
# baseline, inspect toxicity and repeat candidates in independent experiments.
# These scores have no p-values and are not evidence of therapeutic efficacy.

# %%
is_drug = analysis.obs["Drug_code"].astype(str).str.fullmatch(r"Drug\d+")
analysis.obs["is_screen_drug"] = is_drug.to_numpy()
screen = analysis[is_drug | analysis.obs["Drug_code"].eq("c_DM_rand")].copy()
cpt.pp.funnel(
    screen, source_layer="normalized", variance_threshold=None,
    corr_threshold=0.9, snr_threshold=SNR_EXCLUDE_FRACTION,
    replicate_groupby=["Batch", "line", "Concentration_uM"], subset=False,
)
features = screen.var_names[screen.var["highly_variable"]].tolist()
assert features, "No selected features remain."
normalized.var["selected_for_rescue"] = normalized.var_names.isin(features)
normalized.uns["iglut_analysis"] = {
    "input_file": str(DATA_PATH), "positive_gain_enabled": APPLY_POSITIVE_GAIN,
    "snr_exclude_fraction": "disabled" if SNR_EXCLUDE_FRACTION is None else str(SNR_EXCLUDE_FRACTION),
    "score_interpretation": "descriptive single-run well-level morphology rescue",
}
scores = cpt.tl.rescue_scores(
    analysis, reference_query=REFERENCE, disease_control_query=DISEASE_CONTROL,
    treatment_query="is_screen_drug == True", source_layer="normalized", features=features,
)
display(scores.groupby(["Batch", "line"])[["disease_norm", "n_wt", "n_disease"]].first())
ranked = scores.sort_values(["line", "remaining_distance"])
display(ranked.groupby("line", observed=True).head(5))

# %% [markdown]
# ## 9. Save reproducible outputs
# The original H5AD is never overwritten. The new file retains raw measurements,
# the normalized layer, reference feature flags, selected-feature flags, and fitted
# parameters in `uns['cptools']['reference_normalize']`. Invalid-scale features
# remain NaN and flagged in this full saved matrix. Subset them for downstream tools.
# String metadata is converted to categoricals in the output copy for H5AD compatibility.

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
export = normalized.copy()
for key in export.obs.select_dtypes(include=["object", "str", "string"]).columns:
    export.obs[key] = export.obs[key].astype("category")
for key in export.var.select_dtypes(include=["object", "str", "string"]).columns:
    export.var[key] = export.var[key].astype("category")
export.write_h5ad(OUTPUT_DIR / "iglut_reference_normalized.h5ad")
if not cv.empty:
    cv.to_csv(OUTPUT_DIR / "heldout_control_metrics.csv", index=False)
transfer.to_csv(OUTPUT_DIR / "heldout_compound_metrics.csv", index=False)
placement.to_csv(OUTPUT_DIR / "wt_control_placement.csv", index=False)
ranked.to_csv(OUTPUT_DIR / "exploratory_rescue_scores.csv", index=False)
print("Saved analysis outputs to", OUTPUT_DIR.resolve())

# %% [markdown]
# ## Sensitivity exercise
# Refit with `center='median'` and compare selected candidate rankings and held-out
# reference agreement. Does a single WT reference well dominate a feature? With
# three controls, an apparent improvement on fitted controls alone is not validation.
# The scaffold below is optional and does not change the selected normalization.

# %%
# median_result = cpt.pp.reference_normalize(
#     adata, reference_query=REFERENCE, center="median", source_layer="raw", inplace=False,
# )
# median_cv = cpt.tl.reference_control_cv(
#     adata, reference_query=REFERENCE, positive_control_queries=POSITIVES,
#     center="median", n_splits=3, random_state=7,
# )
