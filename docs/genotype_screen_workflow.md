# Genotype × compound screens

This guide combines CPTools functions for screens in which mutant cell lines
are treated with compounds and compared with WT cells. It asks three
questions:

1. How does each mutant differ from WT (its **genotype axis**)?
2. Which compounds move the mutant toward WT along that axis (**rescue**)?
3. Is the rescue specific to the mutant, or does the compound move WT cells
   the same way?

Group names below (`WT_REF`, `MUT_DMSO`, …) are placeholders for `obs.query`
selections; the [metadata guide](metadata.md) describes the columns they use.

## 1. Plate layout

The layout decides which questions the data can answer.

- **WT vehicle wells at scattered positions on every plate**, mirroring the
  mutant vehicle positions. If WT references sit only in a dedicated column,
  every genotype axis also contains whatever distinguishes that column from
  the rest of the plate, and the data cannot separate the two.
- **8–12 vehicle wells per genotype and plate.** Genotype axes from three
  wells are noisy, and differences between mutant lines stay at noise level.
- **Several genotypes per plate**, rotating across plates, so that genotype is
  not confounded with plate. Two or more independent differentiations per
  line make p-values describe biological rather than technical replicates.
- **Positive controls with distinct mechanisms.** Two compounds of one class
  behave as a single condition, and feature selection on them favors that
  mechanism.
- **Randomized compound positions across plates**, so that a compound's
  effect is not tied to one well position on every plate.
- A few **untreated wells** per plate test whether the vehicle itself changes
  the profile; more are rarely needed.
- Export **cell counts** and **per-well acquisition times** with the
  features, to separate toxicity from morphology and to test imaging order.

## 2. Normalize on the shared reference

```python
REFERENCE = "line == 'WT' and role == 'reference'"
ANCHORS = {"positive_A": "line == 'WT' and role == 'positive' and Treatment == 'positive_A'",
           "untreated": "line == 'WT' and role == 'untreated'"}
normalized = cpt.pp.robust_reference_normalize(
    adata, reference_query=REFERENCE, anchor_queries=ANCHORS, plate_shrinkage=True,
    scale="replicate", scale_groupby=["line", "Treatment"], scale_estimator="median",
    source_layer="raw", output_layer="normalized", inplace=False)
```

Each plate is centered on its reference, with plate offsets estimated jointly
from the reference and the anchors and shrunk when they are not larger than
their noise; the scale is the well-to-well SD of replicate groups. Validate
the choice on held-out control wells with `cpt.tl.reference_control_cv(...,
robust_options=...)`; see [shared-reference normalization](reference_normalization.md).

## 3. Select features without favoring a genotype

```python
wt_controls = normalized.obs.eval("line == 'WT' and role in ['reference', 'positive', 'untreated']")
normalized.obs["wt_control"] = normalized.obs["Treatment"].astype(str).where(wt_controls, "")
cpt.pp.funnel(normalized, batch_key="Batch", treatment_key="Treatment", control_value="DMSO",
              source_layer="normalized", variance_threshold=0.05, corr_threshold=0.9,
              snr_threshold=0.5, snr_key="wt_control", snr_groups=["DMSO", "untreated", "positive_A"])
features = normalized.var_names[normalized.var["highly_variable"].to_numpy()].tolist()
nonredundant = normalized.var_names[normalized.var["pass_funnel_prefilter"].to_numpy()].tolist()
```

Ranking features by their reproducibility on WT control wells only (signal:
differences between the control conditions; noise: their well-to-well
variation across plates) favors no genotype and no screening compound.
Conditions that barely differ, such as vehicle and untreated, add wells to the
noise estimate but almost nothing to the signal. Check that the conclusions
hold with other feature sets, e.g. all non-redundant features
(`nonredundant`).

## 4. Do the vehicle wells separate by genotype?

Look at the wells before summarizing them. A genotype difference should be
consistent across replicate vehicle wells: replicates of a mutant resemble
each other and differ from WT.

```python
import numpy as np
import scanpy as sc

vehicle = normalized[normalized.obs.eval("Treatment == 'DMSO'").to_numpy(), features].copy()
vehicle.X = np.asarray(vehicle.layers["normalized"], dtype=float)
sc.pp.pca(vehicle)                                     # color X_pca by "line"
centered = vehicle.X - vehicle.X.mean(axis=0)
unit = centered / np.linalg.norm(centered, axis=1, keepdims=True)
well_cosine = unit @ unit.T                            # well-by-well heatmap
```

Report, per genotype, the mean cosine among its replicate wells, with WT wells
and with other mutants' wells, and whether each well's most similar well has
its genotype. `cpt.tl.group_separation` adds a noise-corrected distance, its
SNR and leave-one-well-out accuracy per pair of groups. Reference wells sit
close together partly by construction, because the normalization centers
them.

## 5. Genotype axes

Within each plate, the genotype axis runs from the mutant vehicle wells to
the WT reference wells. WT cells treated with the same compounds give each
compound's WT effect; compare them with the WT reference of their own plate
(`P00` below, which may be the same plate as the mutant):

```python
groups = {
    "WT_REF": "Batch == 'P01' and line == 'WT' and role == 'reference'",
    "MUT_DMSO": "Batch == 'P01' and line == 'MUT1' and role == 'vehicle'",
    "MUT_untr": "Batch == 'P01' and line == 'MUT1' and role == 'untreated'",
    "WT_untr": "Batch == 'P01' and line == 'WT' and role == 'untreated'",
    "WT_DMSO": "Batch == 'P00' and line == 'WT' and role == 'reference'",
    **{f"MUT_{c}": f"Batch == 'P01' and line == 'MUT1' and Treatment == '{c}'" for c in compounds},
    **{f"WT_{c}": f"Batch == 'P00' and line == 'WT' and Treatment == '{c}'" for c in compounds},
}
axis = ("MUT_DMSO", "WT_REF")                          # 0 = mutant, 1 = WT
```

- `cpt.tl.profile_shift` compares axes: the projection of one line's axis on
  another's (e.g. whether a homozygous line reaches further than the
  heterozygous one) and their noise-corrected cosine.
- `axis_snr` (returned by `profile_shift` and `profile_shift_screen`) is the
  noise-corrected squared length of the axis over its noise. Below about 3,
  the axis is mostly noise and compound scores on it mean little.
- An axis built from untreated wells (`("MUT_untr", "WT_untr")`) should point
  the same way as the vehicle axis.

## 6. Compound scores

```python
rescue = cpt.tl.profile_shift_screen(normalized, groups=groups, axis=axis, features=features,
                                     toward="WT_REF", shifts={c: ("MUT_DMSO", f"MUT_{c}") for c in compounds})
wt_effect = cpt.tl.profile_shift_screen(normalized, groups=groups, axis=axis, features=features,
                                        shifts={c: ("WT_DMSO", f"WT_{c}") for c in compounds})
specific = cpt.tl.profile_shift_screen(normalized, groups=groups, axis=axis, features=features, shifts={
    c: {f"MUT_{c}": 1, "MUT_DMSO": -1, f"WT_{c}": -1, "WT_DMSO": 1} for c in compounds})
```

All scores are in units of the genotype gap and corrected for well noise:

| Score | Column | Meaning |
|---|---|---|
| rescue | `rescue["projection"]` | fraction of the gap the compound covers in the mutant (1 = all the way to WT) |
| WT effect | `wt_effect["projection"]` | the same compound's move along the axis in WT cells |
| mutant-specific rescue | `specific["projection"]` | rescue minus WT effect |
| direction | `cosine_debiased` | how much of the compound's whole change points toward WT |
| other changes | `orthogonal_vs_axis_length` | change outside the axis |
| distance to WT | `remaining_distance` | after treatment ÷ before (< 1 = closer to WT) |

`q_projection` is the Benjamini–Hochberg q-value of the projection across the
compounds of one call.

Interpreting them:

- **Rescue against other changes.** Most compounds change much more than the
  genotype axis. Among compounds with a significant rescue, the ones that no
  other compound beats on both rescue (higher) and off-axis change (lower)
  form the lower-right edge of a rescue vs `orthogonal_vs_axis_length` plot.
  That edge runs from small, clean moves (`remaining_distance` < 1) to large
  moves that point toward WT but overshoot, candidates for a lower dose.
- **Generic vs specific.** If rescue and WT effect correlate strongly across
  compounds, most of the rescue is a generic compound effect. Strong
  compounds also drive both genotypes toward a similar treated state, so even
  the typical compound shows some apparent specificity. Compare each
  compound's specific rescue with the median compound of its line rather
  than with 0.
- **Reproducibility.** Different salt forms or repeated library entries of
  one compound should score alike; `cpt.tl.group_retrieval` and the
  correlation of their scores against random compound pairs test this.

## 7. Visualizing compounds along the axis

`cpt.tl.axis_pca` places wells or compound means in a plane: x is the
position along the genotype axis (0 = mutant vehicle, 1 = WT), y the main
direction of the remaining change of the `fit_query` wells. Both coordinates
are in units of the genotype gap, so distances in the plane are true
distances. y shows one off-axis direction only; plots that must show the
distance to WT, such as rescue vs `orthogonal_vs_axis_length` with the circle
`remaining_distance = 1`, need the whole off-axis length instead.

## 8. Which features differ between mutant and WT?

Test each line against WT on its own plate, then combine the lines:

```python
# one (WT reference, mutant vehicle) pair of groups per plate, defined like those above
contrasts = {"MUT1": ("P01_WT_REF", "P01_MUT1_DMSO"), "MUT2": ("P02_WT_REF", "P02_MUT2_DMSO")}
per_line = cpt.tl.feature_contrasts(normalized, groups=groups, contrasts=contrasts,
                                    features=nonredundant, variance="pooled")
meta = cpt.tl.feature_meta_analysis(per_line)                    # fixed effect
meta_random = cpt.tl.feature_meta_analysis(per_line, method="random")
```

- With three wells per group, each group's own variance is unstable, and a few
  chance-small standard errors dominate inverse-variance weights.
  `variance="pooled"` pools one well-noise variance per feature over all
  groups of the contrasts.
- The fixed-effect estimate asks whether these lines differ from WT on
  average; `I2` shows how much they disagree beyond noise, and the
  random-effects model lets them differ.
- Plot `estimate` against −log10 `p_value` (a volcano plot), colored by
  channel or compartment. When genotype changes many features at once, most
  features can pass q < 0.05; then ask which channels dominate the largest
  effects, e.g. the share of each channel among the N most significant
  decreases and increases, rather than counting significant features.
- Test all non-redundant features rather than the SNR-selected set: the
  correlation filter removes near-duplicates that would inflate counts, and
  SNR selection on control wells did not look at genotype.

## 9. Checks and nulls

- **Activity calls.** `cpt.tl.replicate_map` scores replicate retrieval;
  `cpt.tl.pseudo_treatment_null` (needs at least five vehicle wells per
  genotype and plate for triplicates) and `cpt.tl.scrambled_treatments` give
  nulls for any score.
- **Batch effects.** `cpt.tl.batch_similarity` measures whether compounds
  resemble others from their own plate or imaging day more than chance.
- **Repeated controls.** `cpt.pp.control_agreement_filter` compares dedicated
  and scattered copies of one control condition where both exist.
- **Positional null.** With identical layouts on every plate, a WT plate in
  the screening layout can be scored as a pseudo-disease; compounds that
  "rescue" it reflect position or generic effects.
- **Outlier wells.** `cpt.tl.replicate_outliers` flags wells far from their
  replicates for image review.

## Caveats

- Wells on one plate are technical replicates. Without independent
  differentiations, p-values describe these wells, not the cell lines.
- When each genotype has its own plate, genotype and plate are confounded;
  only layouts with several genotypes per plate separate them.
- A single dose cannot distinguish rescue from toxicity; dose series and cell
  counts can.
