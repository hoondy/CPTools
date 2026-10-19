"""Bias-aware comparisons between groups of well profiles.

Groups are named pandas ``obs.query`` selections. Centroids are arithmetic well
means in a fixed feature space (normally a pooled-SD normalized layer). Every
squared norm or dot product of centroid differences is corrected for the
sampling noise it contains: a centroid of n wells carries noise with expected
squared length trace(Cov)/n, and a group shared by two difference vectors adds
that noise to their dot product. Without the correction, a shift and an axis
that share a baseline group look aligned even when they are unrelated.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from ._reference_normalization import _query_mask, _source
from ._utils import bh_fdr


def _prepare(adata, groups, names, layer, features):
    unknown = [name for name in names if name not in groups]
    if unknown:
        raise KeyError(f"Unknown group names: {unknown}")
    data = adata if features is None else adata[:, list(features)]
    if data.n_vars == 0:
        raise ValueError("Select at least one feature.")
    X = _source(data, layer)
    names = list(dict.fromkeys(names))
    masks = {name: _query_mask(data.obs, groups[name]) for name in names}
    for (a, ma), (b, mb) in combinations(masks.items(), 2):
        if np.any(ma & mb):
            raise ValueError(f"Groups {a!r} and {b!r} share wells; groups must be disjoint.")
    index = {}
    for name, mask in masks.items():
        if mask.sum() < 2:
            raise ValueError(f"Group {name!r} has {int(mask.sum())} wells; need at least 2.")
        index[name] = np.flatnonzero(mask)
    used = np.concatenate(list(index.values()))
    if not np.isfinite(X[used]).all():
        raise ValueError("Selected profiles contain nonfinite values; select a common finite feature set first.")
    return X, index


def _moments(X, index):
    means = {g: X[i].mean(axis=0) for g, i in index.items()}
    noise = {g: float(X[i].var(axis=0, ddof=1).sum() / len(i)) for g, i in index.items()}
    return means, noise


def _shift_statistics(X, index, axis, shift, toward, debias=True):
    means, noise = _moments(X, index)
    if not debias:
        noise = dict.fromkeys(noise, 0.0)
    beta = {axis[1]: 1.0, axis[0]: -1.0}
    alpha = {shift[1]: 1.0, shift[0]: -1.0}
    A = means[axis[1]] - means[axis[0]]
    S = means[shift[1]] - means[shift[0]]
    dot = float(S @ A)
    dot_ub = dot - sum(alpha.get(g, 0) * beta.get(g, 0) * noise[g] for g in index)
    axis_noise = sum(b * b * noise[g] for g, b in beta.items())
    shift_noise = sum(a * a * noise[g] for g, a in alpha.items())
    axis_sq = float(A @ A) - axis_noise
    shift_sq = float(S @ S) - shift_noise
    out = {
        "dot": dot,
        "axis_snr": axis_sq / axis_noise if axis_noise > 0 else np.inf,
        "shift_snr": shift_sq / shift_noise if shift_noise > 0 else np.inf,
        "cosine": dot / np.sqrt(float(A @ A) * float(S @ S)),
        "projection": dot_ub / axis_sq if axis_sq > 0 else np.nan,
        "cosine_debiased": (dot_ub / np.sqrt(axis_sq * shift_sq)) if min(axis_sq, shift_sq) > 0 else np.nan,
        "shift_vs_axis_length": np.sqrt(max(shift_sq, 0) / axis_sq) if axis_sq > 0 else np.nan,
    }
    out["orthogonal_vs_axis_length"] = (
        np.sqrt(max(shift_sq - dot_ub**2 / axis_sq, 0) / axis_sq) if axis_sq > 0 else np.nan)
    if toward is not None:
        def distance_sq(g, h):
            d = means[g] - means[h]
            return float(d @ d) - noise[g] - noise[h]
        before, after = distance_sq(shift[0], toward), distance_sq(shift[1], toward)
        out["remaining_distance"] = np.sqrt(max(after, 0) / before) if before > 0 else np.nan
    return out


def _projection_inference(X, index, axis, shift, projection, axis_sq):
    """Delta-method SE and Welch-Satterthwaite df for the debiased projection.

    First order: each group's wells are projected on the gradient of S.A/|A|^2
    with respect to its centroid, so noise in the shift *and* in the axis both
    count. Second order adds the noise x noise term of S.A, which matters when
    either vector is close to its noise floor.
    """
    means = {g: X[i].mean(axis=0) for g, i in index.items()}
    centered = {g: X[i] - means[g] for g, i in index.items()}
    beta = {axis[1]: 1.0, axis[0]: -1.0}
    alpha = {shift[1]: 1.0, shift[0]: -1.0}
    A = means[axis[1]] - means[axis[0]]
    S = means[shift[1]] - means[shift[0]]
    used = [g for g in index if g in alpha or g in beta]
    first = {}
    for g in used:
        a, b = alpha.get(g, 0.0), beta.get(g, 0.0)
        gradient = (a * A + b * S - 2 * projection * b * A) / axis_sq
        first[g] = float(np.var(centered[g] @ gradient, ddof=1) / len(index[g]))
    second = 0.0
    for g, h in combinations(used, 2):
        c = alpha.get(g, 0) * beta.get(h, 0) + alpha.get(h, 0) * beta.get(g, 0)
        if c:
            cross = np.sum((centered[g] @ centered[h].T) ** 2) / ((len(index[g]) - 1) * (len(index[h]) - 1))
            second += c**2 * cross / (len(index[g]) * len(index[h]))
    for g in used:
        c = alpha.get(g, 0) * beta.get(g, 0)
        if c:
            own = np.sum((centered[g] @ centered[g].T) ** 2) / (len(index[g]) - 1) ** 2
            second += 2 * c**2 * own / len(index[g]) ** 2
    variance = sum(first.values()) + second / axis_sq**2
    df = sum(first.values()) ** 2 / sum(v**2 / (len(index[g]) - 1) for g, v in first.items() if v > 0)
    return float(np.sqrt(variance)), float(df)


def profile_shift(
    adata: ad.AnnData,
    *,
    groups: Mapping[str, str],
    axis: tuple[str, str],
    shift: tuple[str, str],
    toward: str | None = None,
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    n_boot: int = 2000,
    random_state: int = 0,
) -> pd.Series:
    """Measure how far a shift moves profiles along a reference axis.

    ``axis=(a0, a1)`` defines A = centroid(a1) - centroid(a0), e.g. WT ->
    mutant. ``shift=(s0, s1)`` defines S = centroid(s1) - centroid(s0), e.g.
    mutant + vehicle -> mutant + drug. Returned (noise-debiased) values:

    - ``projection``: S.A / |A|^2. 1 = S travels the full axis length toward
      a1; negative values move toward a0 (e.g. rescue for an a1-side shift).
      ``projection_se``, ``projection_ci_low/high`` and ``p_projection`` (H0:
      projection = 0) use a delta-method t approximation with
      Welch-Satterthwaite df; it accounts for noise in both vectors.
    - ``cosine_debiased``: directional agreement corrected for noise in both
      vectors (``cosine`` is the plain, noise-attenuated value).
    - ``shift_vs_axis_length``, ``orthogonal_vs_axis_length``: |S| and its part
      orthogonal to A, in units of |A|.
    - ``remaining_distance`` (with ``toward``): |s1 - toward| / |s0 - toward|.
    - ``axis_snr``/``shift_snr``: debiased squared length over its noise; values
      near or below 1 mean the vector is mostly sampling noise and the ratios
      above are unstable.
    - ``<metric>_ci_low/high`` for the cosine, lengths and remaining distance:
      basic bootstrap intervals (wells resampled within groups, centered on the
      plug-in value so the debiasing is not undone, widened by sqrt(n/(n-1))).

    All uncertainty is between wells of one plate: technical, not biological,
    replication. Prefer axis and shift groups that share no wells (e.g. axis
    from DMSO wells, shift from untreated wells); shared groups are debiased.
    """
    if len(axis) != 2 or len(shift) != 2 or axis[0] == axis[1] or shift[0] == shift[1]:
        raise ValueError("axis and shift must each be two different group names.")
    if toward is not None and toward == shift[0]:
        raise ValueError("toward must differ from the shift start group.")
    names = [*axis, *shift] + ([toward] if toward is not None else [])
    X, index = _prepare(adata, groups, names, layer, features)
    observed = _shift_statistics(X, index, axis, shift, toward)
    se = df = np.nan
    if np.isfinite(observed["projection"]):
        means, noise = _moments(X, index)
        A = means[axis[1]] - means[axis[0]]
        axis_sq = float(A @ A) - noise[axis[0]] - noise[axis[1]]
        se, df = _projection_inference(X, index, axis, shift, observed["projection"], axis_sq)
    result = {
        "axis": f"{axis[0]} -> {axis[1]}", "shift": f"{shift[0]} -> {shift[1]}",
        "toward": toward or "",
        **{f"n_{g}": len(i) for g, i in index.items()},
        "n_features": X.shape[1],
        **{k: v for k, v in observed.items() if k != "dot"},
        "projection_se": se, "projection_df": df,
        "projection_ci_low": observed["projection"] - stats.t.ppf(0.975, df) * se,
        "projection_ci_high": observed["projection"] + stats.t.ppf(0.975, df) * se,
        "p_projection": float(2 * stats.t.sf(abs(observed["projection"]) / se, df)) if se > 0 else np.nan,
    }

    rng = np.random.default_rng(random_state)
    keys = ["cosine_debiased", "shift_vs_axis_length"] + (["remaining_distance"] if toward else [])
    boot = {key: [] for key in keys}
    for _ in range(int(n_boot)):
        sample = {g: rng.choice(i, size=len(i), replace=True) for g, i in index.items()}
        values = _shift_statistics(X, sample, axis, shift, toward)
        for key in keys:
            boot[key].append(values[key])
    plugin = _shift_statistics(X, index, axis, shift, toward, debias=False)
    # Bootstrap spread of a mean of n wells is sqrt((n-1)/n) too small.
    n_min = min(len(i) for i in index.values())
    inflate = np.sqrt(n_min / (n_min - 1))
    for key in keys:
        values = np.asarray(boot[key], dtype=float)
        values = values[np.isfinite(values)]
        low = high = np.nan
        if n_boot and len(values) >= max(20, 0.9 * n_boot) and np.isfinite(observed[key]):
            q_low, q_high = np.percentile(values, [2.5, 97.5])
            low = observed[key] - inflate * (q_high - plugin[key])
            high = observed[key] - inflate * (q_low - plugin[key])
        result[f"{key}_ci_low"], result[f"{key}_ci_high"] = float(low), float(high)
    return pd.Series(result, name=f"{shift[0]} -> {shift[1]} along {axis[0]} -> {axis[1]}")


def _nearest_centroid_accuracy(X, train_a, train_b, test_a, test_b, leave_one_out=False):
    correct = []
    for label, test in ((0, test_a), (1, test_b)):
        for i in test:
            a, b = train_a, train_b
            if leave_one_out:
                a = a[a != i]
                b = b[b != i]
            ca, cb = X[a].mean(axis=0), X[b].mean(axis=0)
            pred = int(np.sum((X[i] - cb) ** 2) < np.sum((X[i] - ca) ** 2))
            correct.append(pred == label)
    return float(np.mean(correct))


def group_separation(
    adata: ad.AnnData,
    *,
    groups: Mapping[str, str],
    pairs: Sequence[tuple[str, str]],
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    n_perm: int = 2000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Test whether paired groups have different centroids.

    One row per pair (a, b), plus ``combined`` when several pairs are given
    (e.g. WT vs mutant, untreated and DMSO; labels are permuted within
    each pair, never across). Columns:

    - ``distance_sq_per_feature``: debiased squared centroid distance / n_features.
      In a pooled-SD normalized layer this is the mean squared standardized
      difference (0 = no separation beyond noise).
    - ``snr``: debiased squared distance over its noise.
    - ``p_value``: one-sided permutation p-value for the squared distance.
    - ``loo_accuracy``: leave-one-well-out nearest-centroid accuracy.
    - ``transfer_accuracy``: accuracy on this pair's wells of centroids trained
      on the other pair(s), e.g. untreated -> DMSO; tests whether the
      separation generalizes to wells in other plate positions/vehicles.

    p-values describe technical (well-level) reproducibility on one plate,
    not generalization to other differentiations, clones or donors.
    """
    if not pairs:
        raise ValueError("Provide at least one (group_a, group_b) pair.")
    names = [g for pair in pairs for g in pair]
    X, index = _prepare(adata, groups, names, layer, features)
    rng = np.random.default_rng(random_state)
    F = X.shape[1]

    def distance_sq(idx_a, idx_b):
        d = X[idx_a].mean(axis=0) - X[idx_b].mean(axis=0)
        return float(d @ d)

    rows, observed_total, nulls = [], 0.0, []
    for a, b in pairs:
        ia, ib = index[a], index[b]
        _, noise = _moments(X, {a: ia, b: ib})
        raw = distance_sq(ia, ib)
        pool = np.concatenate([ia, ib])
        null = []
        for _ in range(int(n_perm)):
            perm = rng.permutation(pool)
            null.append(distance_sq(perm[: len(ia)], perm[len(ia):]))
        null = np.asarray(null)
        nulls.append(null)
        observed_total += raw
        others = [(c, d) for c, d in pairs if (c, d) != (a, b)]
        transfer = [
            _nearest_centroid_accuracy(X, index[c], index[d], ia, ib) for c, d in others
        ]
        rows.append({
            "pair": f"{a} vs {b}", "n_a": len(ia), "n_b": len(ib), "n_features": F,
            "distance_sq_per_feature": (raw - noise[a] - noise[b]) / F,
            "snr": (raw - noise[a] - noise[b]) / (noise[a] + noise[b]),
            "p_value": float((1 + np.sum(null >= raw)) / (1 + len(null))),
            "loo_accuracy": _nearest_centroid_accuracy(X, ia, ib, ia, ib, leave_one_out=True),
            "transfer_accuracy": float(np.mean(transfer)) if transfer else np.nan,
        })
    if len(pairs) > 1:
        combined = np.sum(nulls, axis=0)
        rows.append({
            "pair": "combined", "n_a": sum(r["n_a"] for r in rows), "n_b": sum(r["n_b"] for r in rows),
            "n_features": F,
            "distance_sq_per_feature": float(np.mean([r["distance_sq_per_feature"] for r in rows])),
            "snr": np.nan,
            "p_value": float((1 + np.sum(combined >= observed_total)) / (1 + len(combined))),
            "loo_accuracy": np.nan, "transfer_accuracy": np.nan,
        })
    return pd.DataFrame(rows)


def feature_contrasts(
    adata: ad.AnnData,
    *,
    groups: Mapping[str, str],
    contrasts: Mapping[str, tuple[str, str]],
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-feature Welch t-tests for named contrasts ``name: (a, b)`` (b - a).

    Returns a long table with ``difference`` (layer units), ``cohen_d`` (pooled
    within-pair SD), ``t``, ``p_value`` and Benjamini-Hochberg ``q_value``
    computed within each contrast. Wells are technical replicates; correlated
    features make the number of significant features a descriptive count.
    """
    if not contrasts:
        raise ValueError("Provide at least one contrast.")
    names = [g for pair in contrasts.values() for g in pair]
    X, index = _prepare(adata, groups, names, layer, features)
    feature_names = (adata.var_names if features is None else pd.Index(list(features))).astype(str)
    tables = []
    for name, (a, b) in contrasts.items():
        xa, xb = X[index[a]], X[index[b]]
        diff = xb.mean(axis=0) - xa.mean(axis=0)
        na, nb = len(xa), len(xb)
        pooled = np.sqrt(((na - 1) * xa.var(axis=0, ddof=1) + (nb - 1) * xb.var(axis=0, ddof=1)) / (na + nb - 2))
        t, p = stats.ttest_ind(xb, xa, axis=0, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        tables.append(pd.DataFrame({
            "contrast": name, "feature": feature_names, "group_a": a, "group_b": b,
            "difference": diff,
            "cohen_d": np.divide(diff, pooled, out=np.full_like(diff, np.nan), where=pooled > 0),
            "t": t, "p_value": p, "q_value": bh_fdr(p),
        }))
    return pd.concat(tables, ignore_index=True)
