import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CPTools as cpt


REFERENCE = "line == 'WT' and Drug_code == 'c_DM'"
POSITIVES = {
    "ivermectin": "line == 'WT' and Drug_code == 'c_Ivermectin'",
    "selamectin": "line == 'WT' and Drug_code == 'c_Selamectin'",
}


def experiment(gains=(1, 1, 1), constant=False):
    rows, matrix = [], []
    noise = np.array([[-1, -2], [0, 0], [1, 2]], dtype=float)
    for p, gain in enumerate(gains):
        for code, line, treatment, effect in [
            ("c_DM", "WT", "DMSO", [0, 0]),
            ("c_Ivermectin", "WT", "Ivermectin_B1a", [8, 16]),
            ("c_Selamectin", "WT", "Selamectin", [-12, 24]),
            ("c_DM_rand", "MT", "DMSO", [4, 6]),
            ("Drug1", "MT", "rescue", [0, 0]),
        ]:
            for r in range(3):
                rows.append({"Batch": f"P{p}", "line": line, "Drug_code": code,
                             "Treatment": treatment, "Concentration_uM": "10uM" if "DMSO" != treatment else None,
                             "replicate": r})
                matrix.append(p * 100 + gain * (np.array(effect) + noise[r]))
    X = np.array(matrix, dtype=float)
    if constant:
        X = np.column_stack([X, np.ones(len(X))])
    obs = pd.DataFrame(rows, index=[f"well{i}" for i in range(len(rows))])
    result = ad.AnnData(X, obs=obs)
    result.layers["raw"] = result.X.copy()
    return result


def test_pooled_sd_preserves_disease_and_source():
    data = experiment()
    before = data.X.copy()
    out = cpt.pp.reference_normalize(data, reference_query=REFERENCE, inplace=False)
    model = out.uns["cptools"]["reference_normalize"]
    np.testing.assert_allclose(model["pooled_sd"], [1, 2])
    np.testing.assert_array_equal(model["residual_dof"], [6, 6])
    for plate in data.obs.Batch.unique():
        wt = (data.obs.Batch == plate) & (data.obs.Drug_code == "c_DM")
        mt = (data.obs.Batch == plate) & (data.obs.Drug_code == "c_DM_rand")
        np.testing.assert_allclose(out.X[wt].mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(out.X[mt].mean(axis=0), [4, 3], atol=1e-6)
    np.testing.assert_array_equal(data.X, before)
    np.testing.assert_array_equal(out.layers["raw"], before)
    np.testing.assert_array_equal(out.layers["normalized"], out.X)


def test_gain_recovers_known_affine_effects_and_ignores_mutant_values():
    data = experiment(gains=(0.8, 1, 1.2))
    model = cpt.pp.fit_reference_normalization(
        data, reference_query=REFERENCE, positive_control_queries=POSITIVES, gain_ridge=0,
    )
    np.testing.assert_allclose(model["gains"], np.array([0.8, 1, 1.2])[:, None] * np.ones((1, 2)), atol=1e-5)
    out = cpt.pp.apply_reference_normalization(data, model, inplace=False)
    for p in range(3):
        selected = (data.obs.Batch == f"P{p}") & (data.obs.Drug_code == "c_DM_rand")
        np.testing.assert_allclose(out.X[selected].mean(axis=0), [4, 3], atol=1e-5)
    data.layers["raw"][data.obs.line == "MT"] += 1e6
    again = cpt.pp.fit_reference_normalization(data, reference_query=REFERENCE,
                                              positive_control_queries=POSITIVES, gain_ridge=0)
    np.testing.assert_array_equal(model["centers"], again["centers"])
    np.testing.assert_array_equal(model["gains"], again["gains"])


def test_unsupported_gain_reverts_to_one():
    data = experiment(gains=(0.8, 1, 1.2))
    mask = (data.obs.Batch == "P0") & (data.obs.Drug_code == "c_Selamectin")
    data.layers["raw"][mask] *= -1
    model = cpt.pp.fit_reference_normalization(data, reference_query=REFERENCE,
                                              positive_control_queries=POSITIVES, gain_ridge=0)
    np.testing.assert_array_equal(model["gains"][0], [1, 1])
    assert not model["gain_applied"][0].any()


def test_heldout_wells_do_not_affect_fit():
    data = experiment()
    model = cpt.pp.fit_reference_normalization(data, reference_query=REFERENCE,
                                              positive_control_queries=POSITIVES, fit_query="replicate != 2")
    data.layers["raw"][data.obs.replicate == 2] += 10000
    again = cpt.pp.fit_reference_normalization(data, reference_query=REFERENCE,
                                              positive_control_queries=POSITIVES, fit_query="replicate != 2")
    for key in ["centers", "pooled_sd", "gains", "positive_consensus"]:
        np.testing.assert_allclose(model[key], again[key])
    assert len(model["fit_reference_obs_names"]) == 18


def test_missing_plate_references_fail_without_mutation():
    data = experiment()
    data.obs.loc[(data.obs.Batch == "P1") & (data.obs.Drug_code == "c_DM"), "Drug_code"] = "missing"
    original = data.X.copy()
    with pytest.raises(ValueError, match="P1.*fitting references"):
        cpt.pp.reference_normalize(data, reference_query=REFERENCE)
    np.testing.assert_array_equal(original, data.X)
    assert "normalized" not in data.layers


def test_zero_scale_and_missing_features_are_flagged():
    data = experiment(constant=True)
    data.layers["raw"][(data.obs.Batch == "P0") & (data.obs.Drug_code == "c_DM"), 1] = np.nan
    with pytest.warns(UserWarning, match="2 features"):
        out = cpt.pp.reference_normalize(data, reference_query=REFERENCE)
    np.testing.assert_array_equal(out.var.pass_reference_normalization, [True, False, False])
    assert np.isnan(out.X[:, 1:]).all()
    assert np.isfinite(out.X[:, 0]).all()


@pytest.mark.parametrize("gains", [None, POSITIVES])
def test_h5ad_roundtrip_and_apply(gains, tmp_path):
    data = experiment()
    out = cpt.pp.reference_normalize(data, reference_query=REFERENCE, positive_control_queries=gains, inplace=False)
    # Modern AnnData deliberately requires explicit conversion of string metadata.
    for key in out.obs.select_dtypes(include=["object", "str", "string"]).columns:
        out.obs[key] = out.obs[key].astype("category")
    path = tmp_path / "normalized.h5ad"
    out.write_h5ad(path)
    restored = ad.read_h5ad(path)
    applied = cpt.pp.apply_reference_normalization(data, restored.uns["cptools"]["reference_normalize"], inplace=False)
    np.testing.assert_allclose(applied.X, out.X)


def test_model_rejects_new_plates_reordered_features_and_raw_overwrite():
    data = experiment()
    model = cpt.pp.fit_reference_normalization(data, reference_query=REFERENCE)
    with pytest.raises(ValueError, match="ordered features"):
        cpt.pp.apply_reference_normalization(data[:, ::-1].copy(), model)
    with pytest.raises(ValueError, match="distinct"):
        cpt.pp.apply_reference_normalization(data, model, output_layer="raw")
    data.obs["Batch"] = "new"
    with pytest.raises(ValueError, match="no reference parameters"):
        cpt.pp.apply_reference_normalization(data, model)


def test_sparse_input_and_median_location():
    data = experiment()
    data.layers["raw"] = sparse.csr_matrix(data.layers["raw"])
    out = cpt.pp.reference_normalize(data, reference_query=REFERENCE, center="median")
    assert np.isfinite(out.X).all()
    np.testing.assert_allclose(out.var.reference_pooled_sd, [1, 2])


def test_cv_uses_heldout_replicates_and_common_scale():
    data = experiment(gains=(0.8, 1, 1.2))
    qc = cpt.tl.reference_control_cv(data, reference_query=REFERENCE,
                                    positive_control_queries=POSITIVES, gain_ridge=0)
    assert len(qc) == 27
    assert (qc.n_test_wells == 3).all()
    assert (qc.n_features == 2).all()
    means = qc.groupby("method").plate_dispersion_rms.mean()
    assert means["wt_centered"] < means["raw"]
    assert means["wt_positive_gain"] < means["wt_centered"]
    with pytest.raises(ValueError, match="too few wells"):
        cpt.tl.reference_control_cv(data, reference_query=REFERENCE, positive_control_queries=POSITIVES, n_splits=4)


def test_snr_grouping_preserves_genotype_interactions_and_missing_dose():
    rows, values = [], []
    for genotype, sign in [("WT", 1), ("MT", -1)]:
        for treatment, effect in [("DMSO", 0), ("A", 5), ("B", -5)]:
            for noise in [-0.1, 0.1]:
                rows.append({"Treatment": treatment, "line": genotype,
                             "dose": None if treatment == "DMSO" else "10uM"})
                values.append([sign * effect + noise, noise])
    data = ad.AnnData(np.array(values), obs=pd.DataFrame(rows, index=[str(i) for i in range(len(rows))]))
    pooled = cpt.pp.snr_feature_selection(data, inplace=False)
    grouped = cpt.pp.snr_feature_selection(data, replicate_groupby=["line", "dose"], inplace=False)
    assert grouped.var.replicate_snr.iloc[0] > 100
    assert pooled.var.replicate_snr.iloc[0] < 1
    funnel = cpt.pp.funnel(data, replicate_groupby=["line", "dose"], corr_threshold=None,
                         variance_threshold=None, verbose=False, inplace=False)
    np.testing.assert_allclose(funnel.var.replicate_snr, grouped.var.replicate_snr)


def test_rescue_geometry_and_absent_baseline():
    data = experiment()
    data = cpt.pp.reference_normalize(data, reference_query=REFERENCE)
    scores = cpt.tl.rescue_scores(data, reference_query=REFERENCE,
                                disease_control_query="Drug_code == 'c_DM_rand'",
                                treatment_query="Drug_code == 'Drug1'")
    np.testing.assert_allclose(scores.rescue_projection, 1)
    np.testing.assert_allclose(scores.remaining_distance, 0, atol=1e-6)
    np.testing.assert_allclose(scores.orthogonal_shift, 0, atol=1e-6)
    data.obs.loc[data.obs.Drug_code == "c_DM_rand", "Drug_code"] = "missing"
    with pytest.raises(ValueError, match="Insufficient"):
        cpt.tl.rescue_scores(data, reference_query=REFERENCE,
                             disease_control_query="Drug_code == 'c_DM_rand'",
                             treatment_query="Drug_code == 'Drug1'")


def test_legacy_normalizer_remains_available():
    data = experiment()
    cpt.pp.robust_zscore_norm(data)
    assert "robust_zscore_norm" in data.uns["cptools"]


def test_off_axis_change_is_not_full_rescue_and_zero_disease_is_unscored():
    data = experiment()
    data = cpt.pp.reference_normalize(data, reference_query=REFERENCE)
    drug = data.obs.Drug_code == "Drug1"
    # D=[4,3]; this shift is orthogonal to D and has the same length.
    data.layers["normalized"][drug] += [-3, 4]
    options = dict(reference_query=REFERENCE, disease_control_query="Drug_code == 'c_DM_rand'",
                   treatment_query="Drug_code == 'Drug1'")
    scores = cpt.tl.rescue_scores(data, **options)
    np.testing.assert_allclose(scores.rescue_projection, 1, atol=1e-6)
    np.testing.assert_allclose(scores.remaining_distance, 1, atol=1e-6)
    np.testing.assert_allclose(scores.orthogonal_shift, 1, atol=1e-6)
    for plate in data.obs.Batch.unique():
        disease = (data.obs.Batch == plate) & (data.obs.Drug_code == "c_DM_rand")
        reference = (data.obs.Batch == plate) & (data.obs.Drug_code == "c_DM")
        data.layers["normalized"][disease] = data.layers["normalized"][reference]
    scores = cpt.tl.rescue_scores(data, **options)
    assert scores.rescue_projection.isna().all()
    assert (scores.status == "disease_vector_too_small").all()


@pytest.mark.parametrize("options,match", [
    ({"positive_control_queries": {"bad": REFERENCE}}, "overlaps"),
    ({"min_controls": 1}, "integer >= 2"),
    ({"gain_bounds": (-1, 2)}, "positive"),
    ({"center": "wrong"}, "center"),
])
def test_invalid_fit_options(options, match):
    with pytest.raises(ValueError, match=match):
        cpt.pp.fit_reference_normalization(experiment(), reference_query=REFERENCE, **options)


def test_inplace_and_refitting_use_raw():
    data = experiment()
    before = data.layers["raw"].copy()
    out = cpt.pp.reference_normalize(data, reference_query=REFERENCE)
    assert out is data
    normalized = data.X.copy()
    cpt.pp.reference_normalize(data, reference_query=REFERENCE)
    np.testing.assert_array_equal(data.X, normalized)
    np.testing.assert_array_equal(data.layers["raw"], before)


def test_source_x_snapshot_preserves_float64_values():
    data = experiment()
    del data.layers["raw"]
    data.X += 0.000000123456789
    before = data.X.copy()
    cpt.pp.reference_normalize(data, reference_query=REFERENCE, source_layer=None)
    np.testing.assert_array_equal(data.layers["raw"], before)
