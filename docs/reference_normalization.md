# Shared-reference normalization

This workflow is for well-level profiles with the **same biological reference
condition on every plate**. It is intended to preserve mutant–WT differences
when most screening wells on each plate have a different genotype.

## Choosing the reference wells

Record each control well's role in the metadata (e.g. a `role` column; see the
[metadata guide](metadata.md)) and select the reference by role, genotype and
treatment, e.g. `line == 'WT' and role == 'reference'`. Selecting all
`Treatment == 'DMSO'` mixes WT and mutant controls on genotype-specific
plates. Selecting WT + DMSO without the role also picks up WT vehicle wells
placed among the screening wells; keep those for positional validation.

When the dedicated references sit together (e.g. in one column), a correction
learned there need not remove spatial effects elsewhere on the plate. Compare
dedicated and scattered copies of the same condition on plates that carry
both, and do not transfer a spatial correction estimated on one plate to
others without checking it there. Replicate wells on a plate are not
independent culture runs.

## Default estimator

For plate p and feature f, estimate the reference mean m[p,f] and within-plate
sum of squared deviations SSE[p,f]. Let n[p,f] count finite reference values:

```
pooled_sd[f] = sqrt(sum_p SSE[p,f] / sum_p (n[p,f] - 1))
normalized[i,f] = (raw[i,f] - m[plate(i),f]) / pooled_sd[f]
```

The scale is a pooled sample SD, **not** MAD and not variability pooled before
removing plate offsets. With n finite references on each of P plates, the
scale has P(n - 1) residual degrees of freedom (e.g. 8 for three wells on
four plates). `center="median"` changes location
only; variance is always calculated around within-plate means with n-1 degrees
of freedom. Exclude documented image/QC failures before fitting.

No mutation information or screening drug profile fits the transformation.
Features require at least `min_controls` finite reference measurements on every
plate and pooled SD greater than `min_scale`. Invalid features become NaN with a
false `pass_reference_normalization` flag. They are never amplified by division
by a tiny epsilon. `min_scale` is an explicit absolute threshold in the feature's
input units; review near-zero scales and feature transformations for your assay.

## Public preprocessing functions

```python
cpt.pp.fit_reference_normalization(
    adata, *, reference_query, batch_key="Batch", source_layer="raw",
    fit_query=None, center="mean", min_controls=2, min_scale=1e-8,
    positive_control_queries=None, gain_ridge=10.0, gain_bounds=(0.5, 2.0),
    min_positive_snr=2.0, max_gain_disagreement=2.0,
)
cpt.pp.apply_reference_normalization(
    adata, model, *, source_layer="raw", output_layer="normalized", inplace=True,
)
cpt.pp.reference_normalize(
    adata, *, reference_query, batch_key="Batch", source_layer="raw",
    output_layer="normalized", inplace=True, **fit_options,
)
```

Queries use pandas `obs.query` expressions supplied by the analyst. The fit
function returns a serializable parameter dictionary without changing its input.
`fit_query` restricts **all** fitting, including scales and positive consensus,
to training wells. Application requires known plates and identical ordered
features; apply to held-out rows before subsetting features. Unknown plates
require their own reference fit. There is no silent layer fallback.

Application writes `X` and the output layer, preserves the source, and stores:

- `var["pass_reference_normalization"]`: valid feature mask.
- `var["reference_pooled_sd"]`: final scale per feature.
- `uns["cptools"]["reference_normalize"]`: feature and plate order, centers,
  gains, counts, residual degrees of freedom, validity, fit queries, fitting
  reference observation names and positive-control diagnostics.

`output_layer` cannot overwrite the source or `raw`. When `source_layer=None`,
the current X is also saved as `raw` if that layer does not yet exist. Refit
from raw, not the output of a previous transformation.

## Optional positive-control gains

Pass a mapping of names to queries selecting the **same WT condition at the same
dose and time on every plate**. Queries must not overlap each other or the
negative reference. At least three plates and `min_controls` fitting wells per
positive per plate are required.

For each feature, calculate positive effects relative to each plate's WT
reference. Their median across plates defines consensus q[c,f]. In baseline
pooled-SD units, set u=q/SD and v=plate_effect/SD and estimate:

```
gain[p,f] = (sum_c u[c,f]*v[p,c,f] + gain_ridge)
            / (sum_c u[c,f]**2 + gain_ridge)
```

Ridge shrinks the gain to 1. Retain it only if every supplied positive is finite,
has absolute effect at least `min_positive_snr` in both consensus and that plate,
agrees in sign, and the largest/smallest inferred positive-specific gain ratio
is <= `max_gain_disagreement`. Estimates outside `gain_bounds` revert to 1;
they are not clipped. `gain_applied` records supported fits. Unsupported features
retain gain 1. Recalculate pooled WT SD from reference residuals divided by their
plate's gain. The final transformation divides the centered profile by gain*SD.

These defaults are conservative candidate settings, not established assay
constants. Strong positive responses may make the ridge penalty relatively weak.
Two positives cannot identify a general high-dimensional transformation, and a
technical effect on WT may not act identically on mutants. Do not fit mutant
positive responses to a WT consensus or assume the positives are rescue controls.

## Robust variant: multi-anchor, shrunk plate offsets, replicate scale

With three reference wells per plate, the plate center m[p,f] has sampling
error SD/sqrt(3) in every feature, and that error is added to every well on the
plate. Per-plate WT centering can then improve held-out agreement of the
reference itself while *worsening* agreement of positive controls on the same
plates. That pattern means the plate shift estimated from a few adjacent
reference wells does not transfer to neighbouring conditions, let alone to
dispersed wells. `fit_robust_reference_normalization`
addresses this in three ways.

```python
model = cpt.pp.fit_robust_reference_normalization(
    adata, reference_query=REFERENCE,
    anchor_queries=POSITIVES,            # optional: same WT condition on every plate
    plate_shrinkage=True,                # James-Stein; needs >= 4 plates
    scale="replicate",                   # or "reference" / "anchors"
    scale_groupby=["line", "Treatment"], # replicate groups (Batch is added)
    scale_query="role != 'reference'",   # e.g. dispersed wells only
    scale_estimator="median",            # resists toxic/failed groups
)
normalized = cpt.pp.apply_reference_normalization(adata, model, inplace=False)
```

1. **Anchors.** Per feature, fit loc[c,p] = mu[c] + delta[p] across the
   reference and anchor conditions (weighted by finite well counts). Anchors
   only inform the shared plate offset delta[p]; the origin is still
   mu[reference]. One condition reproduces the plain per-plate means.
2. **Shrinkage.** center[p] = mu[reference] + B * delta[p] with
   B = max(0, 1 - (P - 3) v / sum_p delta[p]^2) (positive-part Efron-Morris).
   v is the variance of an estimated offset: anchor well noise plus, with two
   or more anchors, the plate × condition interaction variance estimated from
   additive-fit residuals. When plate shifts are small or do not transfer
   between conditions, B -> 0 and plates are centered on the pooled reference
   mean. When shared shifts are large, B -> 1. With few plates B is noisy:
   at P = 6 its median is about 0.3 even without any plate effect.
3. **Scale.** `"replicate"` pools within-group residual variance of replicate
   groups (e.g. every compound triplicate on every plate: hundreds of df
   instead of a dozen). Group means never enter it. Adjacent reference wells
   (e.g. one column) likely understate the well-to-well noise of dispersed
   wells, which inflates every
   z-score and every disease norm. Compare `model["pooled_sd"]` for
   `scale="reference"` and `"replicate"` on your data.

The returned model is compatible with `apply_reference_normalization` and H5AD
storage. Extra diagnostics: `plate_offsets` (unshrunk delta),
`shrinkage_factor`, `plate_offset_noise_var`, `plate_condition_interaction_var`,
`condition_means`, `condition_within_sd`, `scale_dof`.

In simulations with three adjacent reference wells per plate, local effects
at the reference position and small plate-wide shifts, the robust variant
roughly halved the RMS error of the plate center relative to the true
plate-wide WT center, compared with the per-plate WT mean. Held-out positives
improved over WT centering in every scenario tested and nearly matched no
centering when there was no plate effect. Validate on
your plates with the CV below (`robust_options`), not with rescue rankings.

## Feature QC from repeated controls

When the same control condition occupies two well sets, such as dedicated
reference wells and scattered vehicle wells on a WT plate, their disagreement
estimates how much a feature responds to well handling rather than to biology.
`cpt.pp.control_agreement_filter` tests this per feature using control wells
only, and reports each comparison's median |z| relative to pure noise:

```python
copies = {
    "DMSO": ("line == 'WT' and role == 'reference' and Treatment == 'DMSO'",
             "line == 'WT' and role == 'vehicle' and Treatment == 'DMSO'"),
    "positive_A": ("line == 'WT' and role == 'positive' and Treatment == 'positive_A'",
                   "line == 'WT' and role == 'screen' and Treatment == 'positive_A'"),
}
cpt.pp.control_agreement_filter(adata, copies=copies, batch_key="Batch", source_layer="raw")
adata.uns["cptools"]["control_agreement_filter"]["median_abs_z_ratio"]
```

If the median ratio is near 1, a minority of features is responsible, and
dropping the flagged features is appropriate. If it is well above 1, the copies
differ in typical features. Filtering then removes much of the feature space
without removing the difference. Every disease vector that compares scattered
mutant wells with dedicated WT wells also contains that difference, so correct
it rather than filter it. A design with both copies on every plate gives one
comparison per plate and much better per-feature power.

## Held-out diagnostics

```python
POSITIVES = {
    "positive_A": "line == 'WT' and role == 'positive' and Treatment == 'positive_A'",
    "positive_B": "line == 'WT' and role == 'positive' and Treatment == 'positive_B'",
}
qc = cpt.tl.reference_control_cv(
    adata,
    reference_query="line == 'WT' and role == 'reference'",
    positive_control_queries=POSITIVES,
    source_layer="raw", n_splits=3, random_state=7,
    include_positive_gain=False,
    robust_options={"anchor_queries": POSITIVES, "plate_shrinkage": True},  # optional
)
```

`robust_options` adds method `"robust"`. When a held-out condition is itself an
anchor, it is dropped from the anchors for its own evaluation
(leave-condition-out), so the positive rows test whether a plate center
*transfers* to a condition that did not fit it. The reference row rewards
fitting local, reference-specific shifts and favors per-plate WT centering by
construction; give the positive rows more weight when the goal is correcting
screening wells. `noise_floor_rms` is the dispersion expected from held-out
well noise alone; no method can beat it on average. Scale options are ignored
in the CV because all methods share one denominator.

Each fold holds out wells within every plate/condition. Training alone determines
all normalizer parameters. Three replicates permit two training wells and one
test well per fold. Missing reference groups or insufficient training counts
raise errors. The helper operates on reference rows only for efficiency.

Returns one row per fold/condition/method with `plate_dispersion_rms` and
`mean_effect_rms`. With `include_positive_gain=False`, methods are `raw` and
`wt_centered`; positive controls are evaluated but never fit the correction.
The backward-compatible default, `include_positive_gain=True`, additionally fits
and evaluates `wt_positive_gain`. All methods use the **same baseline training SD**
for evaluation, so changing the normalization denominator cannot manufacture
improvement. Dispersion is RMS
variation of held-out plate means around their across-plate mean. Effect magnitude
is the RMS across-feature mean effect relative to held-out WT references.
Inspect both: compression can improve dispersion while reducing biological signal.
Feature counts can differ between folds and are returned explicitly. These are
descriptive metrics, not statistical significance tests. Improvement for the
reference alone does not establish correction of the positive conditions or of
spatial effects away from the reference wells.

## Genotype-aware selection and rescue scoring

`snr_feature_selection` and `funnel` accept `replicate_groupby`, e.g.
`["Batch", "line", "Concentration_uM"]`. Treatment is always included; missing
doses remain explicit groups instead of dropping vehicle wells. Restrict selection
to screening roles to avoid mixing positional reference roles. `funnel` does not
normalize or whiten the input. SNR selection is optional: drug-responsive feature
selection can discard disease biology, so assess sensitivity to the feature set.

```python
scores = cpt.tl.rescue_scores(
    analysis,
    reference_query="line == 'WT' and role == 'reference'",
    disease_control_query="line != 'WT' and role == 'vehicle'",
    treatment_query="role == 'screen'",
    source_layer="normalized", features=selected_features,
)
```

`rescue_scores` uses `Batch`, `line`, `Treatment`, `Concentration_uM` by default
(configurable keys; `dose_key=None` omits dose). `reference_genotype="WT"`
is excluded from candidate scoring. It requires at least `min_replicates=2`
wells for WT, untreated genotype, and each treatment/dose on the same plate.
Choose a common finite feature set; nonfinite selected values raise an error.

For D = mutant_DMSO - same_plate_WT_DMSO and P = mutant_drug - mutant_DMSO:

- `rescue_projection = -P.D / D.D`: 0=no axial reversal, 1=full axial reversal,
  >1=overshoot, <0=further in the disease direction.
- `remaining_distance = ||D+P|| / ||D||`: 0=WT centroid, 1=untreated distance.
- `orthogonal_shift = ||P + rescue_projection*D|| / ||D||`: changes orthogonal
  to the disease vector.

Results also contain group keys, replicate counts, feature count and disease norm.

**Shared-control bias.** P and D share the mutant-DMSO centroid with opposite
signs. Its sampling noise adds -tr(Cov_MT)/n_MT to P.D, so every inert drug gets
R ~ tr(Cov_MT)/n_MT / ||D||^2 > 0. With unit noise per feature, n_MT = 3,
1,000 features and ||D|| = 75, that spurious "rescue" is about 0.06, and it
grows when the dispersed wells are noisier than the reference.
`rescue_projection_debiased` and `remaining_distance_debiased` subtract the
expected noise terms using same-plate within-group variances. `disease_snr` is
the debiased ||D||^2 over its noise. Debiased scores are NaN when D is not above
noise. Rank on the debiased columns and report `disease_snr`.

**Identical layouts.** When every plate uses the same drug positions, position
effects repeat on every plate. Agreement of a hit across genotypes is then not
independent replication. A WT plate with the screening layout can supply a
positional null: score its scattered WT wells as a pseudo-disease (e.g. relabel
`line` of its non-reference wells as `"WT_null"`). Drugs that "rescue" this pseudo-disease
vector reflect position or generic effects.
If the norm is <= `min_disease_norm=1e-8`, scores are NaN and status is
`disease_vector_too_small`. This is only a numerical guard; it does not establish
biological separability. The scores have no p-values, confidence intervals or
toxicity adjustment. Bootstrap controls and treatment wells, validate the disease
signature and confirm candidates in independent culture runs before claiming rescue.
