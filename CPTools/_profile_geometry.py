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

from ._reference_normalization import _batches, _query_mask, _source
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


def _coefficients(spec, label):
    """Weights {group: w} for a (start, end) pair or a weighted combination of groups."""
    if isinstance(spec, Mapping):
        weights = {str(g): float(w) for g, w in spec.items() if float(w) != 0}
        if len(weights) < 2 or not all(np.isfinite(list(weights.values()))):
            raise ValueError(f"{label} needs at least two groups with finite, nonzero weights.")
        if abs(sum(weights.values())) > 1e-9:
            raise ValueError(f"{label} weights must sum to 0 so the vector is a difference of centroids.")
        return weights
    if len(spec) != 2 or spec[0] == spec[1]:
        raise ValueError(f"{label} must be two different group names or a {{group: weight}} mapping.")
    return {spec[1]: 1.0, spec[0]: -1.0}


def _describe(spec):
    if not isinstance(spec, Mapping):
        return f"{spec[0]} -> {spec[1]}"
    terms = [f"{'-' if w < 0 else '+'} {abs(w):g}·{g}" for g, w in spec.items() if w]
    return " ".join(terms).lstrip("+ ")


def _shift_statistics(X, index, alpha, beta, path=None, debias=True):
    """Debiased geometry of S = sum(alpha * centroids) on A = sum(beta * centroids)."""
    means, noise = _moments(X, index)
    if not debias:
        noise = dict.fromkeys(noise, 0.0)
    A = sum(w * means[g] for g, w in beta.items())
    S = sum(w * means[g] for g, w in alpha.items())
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
    if path is not None:
        start, end, toward = path

        def distance_sq(g, h):
            d = means[g] - means[h]
            return float(d @ d) - noise[g] - noise[h]
        before, after = distance_sq(start, toward), distance_sq(end, toward)
        out["remaining_distance"] = np.sqrt(max(after, 0) / before) if before > 0 else np.nan
    return out


def _projection_inference(X, index, alpha, beta, projection, axis_sq):
    """Delta-method SE and Welch-Satterthwaite df for the debiased projection.

    First order: each group's wells are projected on the gradient of S.A/|A|^2
    with respect to its centroid, so noise in the shift *and* in the axis both
    count. Second order adds the noise x noise term of S.A, which matters when
    either vector is close to its noise floor.
    """
    means = {g: X[i].mean(axis=0) for g, i in index.items()}
    centered = {g: X[i] - means[g] for g, i in index.items()}
    A = sum(w * means[g] for g, w in beta.items())
    S = sum(w * means[g] for g, w in alpha.items())
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
    axis: tuple[str, str] | Mapping[str, float],
    shift: tuple[str, str] | Mapping[str, float],
    toward: str | None = None,
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    n_boot: int = 2000,
    random_state: int = 0,
) -> pd.Series:
    """Measure how far a shift moves profiles along a reference axis.

    ``axis=(a0, a1)`` defines A = centroid(a1) - centroid(a0), e.g. WT ->
    mutant. ``shift=(s0, s1)`` defines S = centroid(s1) - centroid(s0), e.g.
    mutant + vehicle -> mutant + drug. Either can instead be a weighted
    combination ``{group: weight}`` with weights summing to 0, e.g. a genotype
    axis averaged over two vehicles, ``{"MUT_untreated": .5, "MUT_DMSO": .5,
    "WT_untreated": -.5, "WT_DMSO": -.5}``. Returned (noise-debiased) values:

    - ``projection``: S.A / |A|^2. 1 = S travels the full axis length toward
      a1; negative values move toward a0 (e.g. rescue for an a1-side shift).
      ``projection_se``, ``projection_ci_low/high`` and ``p_projection`` (H0:
      projection = 0) use a delta-method t approximation with
      Welch-Satterthwaite df; it accounts for noise in both vectors.
    - ``cosine_debiased``: directional agreement corrected for noise in both
      vectors (``cosine`` is the plain, noise-attenuated value).
    - ``shift_vs_axis_length``, ``orthogonal_vs_axis_length``: |S| and its part
      orthogonal to A, in units of |A|.
    - ``remaining_distance`` (with ``toward``; needs a ``(s0, s1)`` shift):
      |s1 - toward| / |s0 - toward|.
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
    beta = _coefficients(axis, "axis")
    alpha = _coefficients(shift, "shift")
    path = None
    if toward is not None:
        if isinstance(shift, Mapping):
            raise ValueError("toward needs a (start, end) shift, not a weighted combination.")
        if toward == shift[0]:
            raise ValueError("toward must differ from the shift start group.")
        path = (shift[0], shift[1], toward)
    names = [*beta, *alpha] + ([toward] if toward is not None else [])
    X, index = _prepare(adata, groups, names, layer, features)
    result = {
        "axis": _describe(axis), "shift": _describe(shift), "toward": toward or "",
        **{f"n_{g}": len(i) for g, i in index.items()},
        "n_features": X.shape[1],
        **_shift_row(X, index, alpha, beta, path, n_boot, np.random.default_rng(random_state)),
    }
    return pd.Series(result, name=f"{_describe(shift)} along {_describe(axis)}")


def _shift_row(X, index, alpha, beta, path, n_boot, rng):
    """Debiased statistics, projection inference and bootstrap intervals of one shift."""
    observed = _shift_statistics(X, index, alpha, beta, path)
    se = df = np.nan
    if np.isfinite(observed["projection"]):
        means, noise = _moments(X, index)
        A = sum(w * means[g] for g, w in beta.items())
        axis_sq = float(A @ A) - sum(w * w * noise[g] for g, w in beta.items())
        se, df = _projection_inference(X, index, alpha, beta, observed["projection"], axis_sq)
    result = {
        **{k: v for k, v in observed.items() if k != "dot"},
        "projection_se": se, "projection_df": df,
        "projection_ci_low": observed["projection"] - stats.t.ppf(0.975, df) * se,
        "projection_ci_high": observed["projection"] + stats.t.ppf(0.975, df) * se,
        "p_projection": float(2 * stats.t.sf(abs(observed["projection"]) / se, df)) if se > 0 else np.nan,
    }

    keys = ["cosine_debiased", "shift_vs_axis_length", "orthogonal_vs_axis_length"]
    keys += ["remaining_distance"] if path is not None else []
    boot = {key: [] for key in keys}
    for _ in range(int(n_boot)):
        sample = {g: rng.choice(i, size=len(i), replace=True) for g, i in index.items()}
        values = _shift_statistics(X, sample, alpha, beta, path)
        for key in keys:
            boot[key].append(values[key])
    plugin = _shift_statistics(X, index, alpha, beta, path, debias=False)
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
    return result


def profile_shift_screen(
    adata: ad.AnnData,
    *,
    groups: Mapping[str, str],
    axis: tuple[str, str] | Mapping[str, float],
    shifts: Mapping[str, tuple[str, str] | Mapping[str, float]],
    toward: str | None = None,
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    n_boot: int = 0,
    random_state: int = 0,
) -> pd.DataFrame:
    """Measure many shifts on one axis, e.g. every compound of a screen.

    Each value of ``shifts`` is a ``(start, end)`` pair or a ``{group: weight}``
    combination, as in ``profile_shift``, and gets the same statistics. Shifts
    typically share a baseline group (the vehicle) and may share groups with
    the axis; the debiasing handles both. Contrasts between shifts are shifts
    too, e.g. a drug's response in a mutant minus its response in WT:
    ``{"MUT_drug": 1, "MUT_DMSO": -1, "WT_drug": -1, "WT_DMSO": 1}``.

    ``toward`` adds ``remaining_distance`` for ``(start, end)`` shifts; weighted
    shifts get NaN. ``q_projection`` is the Benjamini-Hochberg q-value of
    ``p_projection`` across the shifts of this call, so call once per family
    (e.g. per genotype). Projection intervals are analytic; ``n_boot > 0``
    adds bootstrap intervals for the other metrics at n_boot x shifts cost.

    Returns one row per shift, indexed by its name, with ``min_group_wells``
    (the smallest group the shift uses) instead of per-group counts.
    """
    if not shifts:
        raise ValueError("Provide at least one shift.")
    beta = _coefficients(axis, "axis")
    alphas = {str(name): _coefficients(spec, f"shift {name!r}") for name, spec in shifts.items()}
    paths = {}
    if toward is not None:
        for name, spec in shifts.items():
            if isinstance(spec, Mapping):
                continue
            if toward == spec[0]:
                raise ValueError(f"toward must differ from the start group of shift {name!r}.")
            paths[str(name)] = (spec[0], spec[1], toward)
    names = [*beta, *(g for alpha in alphas.values() for g in alpha)] + ([toward] if toward is not None else [])
    X, index = _prepare(adata, groups, names, layer, features)
    rng = np.random.default_rng(random_state)
    rows = {}
    for (name, alpha), spec in zip(alphas.items(), shifts.values()):
        path = paths.get(name)
        used = {*beta, *alpha, *([toward] if path is not None else [])}
        subset = {g: i for g, i in index.items() if g in used}
        rows[name] = {
            "axis": _describe(axis), "shift": _describe(spec), "toward": toward if path is not None else "",
            "n_features": X.shape[1], "min_group_wells": min(len(index[g]) for g in alpha),
            **_shift_row(X, subset, alpha, beta, path, n_boot, rng),
        }
        if toward is not None and path is None:
            rows[name]["remaining_distance"] = np.nan
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "name"
    p = table["p_projection"].to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    q[finite] = bh_fdr(p[finite])
    table["q_projection"] = q
    return table


def axis_pca(
    adata: ad.AnnData,
    *,
    groups: Mapping[str, str],
    axis: tuple[str, str] | Mapping[str, float],
    origin: str | None = None,
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    fit_query: str | None = None,
    n_comps: int = 2,
) -> pd.DataFrame:
    """Coordinates along an axis and along the main directions orthogonal to it.

    A PCA whose first coordinate is fixed to an axis, e.g. WT -> mutant.
    ``axis`` is a ``(start, end)`` pair or a ``{group: weight}`` combination
    as in ``profile_shift``; ``origin`` (default: the start group of a pair)
    is placed at 0. For each well x, with A the axis vector and o the origin
    centroid:

    - ``axis`` = (x - o).A / |A|^2: 0 at the origin, 1 at the end of the axis;
    - ``orthogonal_k``: the residual (x - o) minus its axis part, projected on
      the k-th principal direction of the residuals of the ``fit_query``
      wells (all wells by default), divided by |A|.

    All coordinates share the axis-length unit, so a plot of ``axis`` against
    ``orthogonal_1`` keeps distances comparable. Residual directions are taken
    around the origin, not around the residual mean, so the origin stays at
    0 on every coordinate; each direction is signed so that the fit wells
    have a non-negative mean. ``attrs["variance_fraction"]`` gives the share
    of the fit wells' squared distance from the origin along the axis and
    along each orthogonal direction; ``attrs["axis_length"]`` is |A|.

    Centroids carry well noise; with few wells per group, the axis direction
    is approximate. Returns one row per well, indexed like ``adata.obs``.
    """
    beta = _coefficients(axis, "axis")
    if origin is None:
        if isinstance(axis, Mapping):
            raise ValueError("origin is required when axis is a weighted combination.")
        origin = axis[0]
    if not isinstance(n_comps, (int, np.integer)) or n_comps < 1:
        raise ValueError("n_comps must be a positive integer.")
    X, index = _prepare(adata, groups, [*beta, origin], layer, features)
    if not np.isfinite(X).all():
        raise ValueError("Profiles contain nonfinite values; select a common finite feature set first.")
    means = {g: X[i].mean(axis=0) for g, i in index.items()}
    A = sum(w * means[g] for g, w in beta.items())
    length = float(np.linalg.norm(A))
    if length == 0:
        raise ValueError("The axis has zero length.")
    Z = X - means[origin]
    along = Z @ A / length**2
    residual = Z - np.outer(along, A)
    fit = np.ones(adata.n_obs, dtype=bool) if fit_query is None else _query_mask(adata.obs, fit_query)
    if fit.sum() < 2:
        raise ValueError("fit_query selects fewer than 2 wells.")
    _, singular, directions = np.linalg.svd(residual[fit], full_matrices=False)
    n_comps = min(int(n_comps), len(singular))
    coords = residual @ directions[:n_comps].T
    flip = np.where(coords[fit].mean(axis=0) < 0, -1.0, 1.0)
    coords = coords * flip / length
    total = float(np.sum(Z[fit] ** 2))
    table = pd.DataFrame(np.column_stack([along, coords]), index=adata.obs_names,
                         columns=["axis", *[f"orthogonal_{k + 1}" for k in range(n_comps)]])
    table.attrs["axis_length"] = length
    table.attrs["variance_fraction"] = {
        "axis": float(np.sum((along[fit] * length) ** 2) / total) if total > 0 else np.nan,
        **{f"orthogonal_{k + 1}": float(singular[k] ** 2 / total) if total > 0 else np.nan for k in range(n_comps)},
    }
    return table


def replicate_outliers(
    adata: ad.AnnData,
    *,
    groupby: str,
    layer: str | None = "normalized",
    features: Sequence[str] | None = None,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Flag wells whose profile is far from the other wells of their condition.

    For each well, ``distance`` is the RMS over features of its difference from
    the mean of the *other* wells in its ``groupby`` group (layer units; SD units
    for a noise-scaled layer). ``relative_distance`` divides it by the median
    distance of its group, so a group whose wells all vary more is not flagged
    as a whole. ``robust_z`` is log(relative_distance) over 1.4826 x the MAD of
    that log ratio pooled across all groups; wells with ``robust_z > threshold``
    are flagged. Pooling matters with few wells: a MAD from six wells is so
    unstable that a within-group z flags a clean well on many plates. Groups
    with fewer than three wells get NaN.

    A flag means "inspect the images", not "remove". Returns one row per well
    (indexed like ``adata.obs``).
    """
    if groupby not in adata.obs:
        raise KeyError(f"Missing {groupby!r} in obs.")
    data = adata if features is None else adata[:, list(features)]
    if data.n_vars == 0:
        raise ValueError("Select at least one feature.")
    X = _source(data, layer)
    if not np.isfinite(X).all():
        raise ValueError("Profiles contain nonfinite values; select a common finite feature set first.")
    labels = data.obs[groupby].astype(str).to_numpy()
    distance = np.full(data.n_obs, np.nan)
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group)
        if len(idx) < 3:
            continue
        total = X[idx].sum(axis=0)
        for i in idx:
            others = (total - X[i]) / (len(idx) - 1)
            distance[i] = np.sqrt(np.mean((X[i] - others) ** 2))
    frame = pd.DataFrame({"group": labels, "distance": distance}, index=data.obs_names)
    median = frame.groupby("group")["distance"].transform("median")
    frame["relative_distance"] = frame["distance"] / median.where(median > 0)
    log_ratio = np.log(frame["relative_distance"])
    mad = 1.4826 * np.nanmedian(np.abs(log_ratio - np.nanmedian(log_ratio))) if log_ratio.notna().any() else np.nan
    frame["robust_z"] = log_ratio / mad if mad > 0 else np.nan
    frame["outlier"] = frame["robust_z"] > threshold
    return frame


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
    variance: str = "welch",
) -> pd.DataFrame:
    """Per-feature t-tests for named contrasts ``name: (a, b)`` (b - a).

    Returns a long table with ``difference`` (layer units), its
    ``standard_error``, ``cohen_d`` (pooled within-pair SD), ``t``,
    ``p_value`` and Benjamini-Hochberg ``q_value`` computed within each
    contrast. ``variance`` sets the standard error:

    - ``"welch"``: sqrt(var_a/n_a + var_b/n_b) from the two groups (Welch test);
    - ``"pooled"``: one variance per feature pooled over every group named in
      ``contrasts`` (df = sum of n_g - 1), sqrt(s^2 (1/n_a + 1/n_b)), with a
      t test on that df. It assumes equal well noise across groups, as in a
      noise-scaled layer, and is far more stable when groups have few wells.

    Wells are technical replicates; correlated features make the number of
    significant features a descriptive count. ``feature_meta_analysis``
    combines the contrasts per feature.
    """
    if not contrasts:
        raise ValueError("Provide at least one contrast.")
    if variance not in {"welch", "pooled"}:
        raise ValueError("variance must be 'welch' or 'pooled'.")
    names = [g for pair in contrasts.values() for g in pair]
    X, index = _prepare(adata, groups, names, layer, features)
    feature_names = (adata.var_names if features is None else pd.Index(list(features))).astype(str)
    if variance == "pooled":
        unique = list(dict.fromkeys(names))
        df_pooled = sum(len(index[g]) - 1 for g in unique)
        if df_pooled < 1:
            raise ValueError("variance='pooled' needs groups with at least two wells.")
        s2 = sum(((X[index[g]] - X[index[g]].mean(axis=0)) ** 2).sum(axis=0) for g in unique) / df_pooled
    tables = []
    for name, (a, b) in contrasts.items():
        xa, xb = X[index[a]], X[index[b]]
        diff = xb.mean(axis=0) - xa.mean(axis=0)
        na, nb = len(xa), len(xb)
        va, vb = xa.var(axis=0, ddof=1), xb.var(axis=0, ddof=1)
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        if variance == "pooled":
            se = np.sqrt(s2 * (1.0 / na + 1.0 / nb))
            t = np.divide(diff, se, out=np.full_like(diff, np.nan), where=se > 0)
            p = 2.0 * stats.t.sf(np.abs(t), df_pooled)
        else:
            se = np.sqrt(va / na + vb / nb)
            t, p = stats.ttest_ind(xb, xa, axis=0, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        tables.append(pd.DataFrame({
            "contrast": name, "feature": feature_names, "group_a": a, "group_b": b,
            "difference": diff, "standard_error": se,
            "cohen_d": np.divide(diff, pooled, out=np.full_like(diff, np.nan), where=pooled > 0),
            "t": t, "p_value": p, "q_value": bh_fdr(p),
        }))
    return pd.concat(tables, ignore_index=True)


def feature_meta_analysis(
    table: pd.DataFrame,
    *,
    effect: str = "difference",
    standard_error: str = "standard_error",
    by: str = "feature",
    study: str = "contrast",
    method: str = "fixed",
) -> pd.DataFrame:
    """Inverse-variance meta-analysis of per-feature effects across contrasts.

    Combines a long table with one row per ``by`` (feature) and ``study``
    (contrast), such as the output of ``feature_contrasts``, into one row per
    feature, like metafor's ``rma(yi, sei, method="FE")``. With effects y_i,
    standard errors s_i and weights w_i = 1 / s_i^2 over the k contrasts:

    - ``method="fixed"``: estimate = sum(w_i y_i) / sum(w_i), standard error
      1 / sqrt(sum(w_i)); it tests a common effect shared by all contrasts.
    - ``method="random"``: DerSimonian-Laird; between-contrast variance
      tau2 = max(0, (Q - (k - 1)) / (sum(w_i) - sum(w_i^2) / sum(w_i))) and
      weights 1 / (s_i^2 + tau2).

    Columns: ``estimate``, ``standard_error``, 95% ``ci_low``/``ci_high``,
    ``z``, two-sided ``p_value``, Benjamini-Hochberg ``q_value`` across
    features, Cochran's ``Q`` (sum(w_i (y_i - fixed estimate)^2)) with its
    chi-square ``Q_p_value`` on k - 1 df, ``I2`` = max(0, (Q - (k - 1)) / Q),
    ``tau2`` (0 for the fixed model), ``k`` and ``n_positive`` (contrasts with
    y_i > 0). Rows with a missing effect or a nonpositive or missing standard
    error are skipped. Standard errors from a few wells are themselves noisy,
    so p-values are approximate. Returns a table indexed by ``by``.
    """
    if method not in {"fixed", "random"}:
        raise ValueError("method must be 'fixed' or 'random'.")
    missing = [c for c in (effect, standard_error, by, study) if c not in table.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if table.duplicated([by, study]).any():
        raise ValueError(f"Each ({by}, {study}) pair must appear once.")
    y = table[effect].to_numpy(dtype=float)
    s = table[standard_error].to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(s) & (s > 0)
    if not keep.any():
        raise ValueError("No rows with a finite effect and a positive standard error.")
    data = pd.DataFrame({"key": table[by].to_numpy()[keep], "y": y[keep], "w": 1.0 / s[keep] ** 2})
    data["wy"], data["wyy"], data["ww"] = data.w * data.y, data.w * data.y ** 2, data.w ** 2
    sums = data.groupby("key", sort=False).agg(k=("y", "size"), n_positive=("y", lambda v: int((v > 0).sum())),
                                              w=("w", "sum"), wy=("wy", "sum"), wyy=("wyy", "sum"), ww=("ww", "sum"))
    fixed = sums.wy / sums.w
    Q = (sums.wyy - sums.wy ** 2 / sums.w).clip(lower=0.0)
    df = sums.k - 1
    if method == "fixed":
        tau2 = pd.Series(0.0, index=sums.index)
        estimate, se = fixed, 1.0 / np.sqrt(sums.w)
    else:
        c = sums.w - sums.ww / sums.w
        tau2 = ((Q - df) / c.where(c > 0)).clip(lower=0.0).fillna(0.0)
        data["w_re"] = 1.0 / (1.0 / data.w + data.key.map(tau2))
        data["wy_re"] = data.w_re * data.y
        re = data.groupby("key", sort=False)[["w_re", "wy_re"]].sum()
        estimate, se = re.wy_re / re.w_re, 1.0 / np.sqrt(re.w_re)
    z = estimate / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    out = pd.DataFrame({
        "estimate": estimate, "standard_error": se,
        "ci_low": estimate - 1.959964 * se, "ci_high": estimate + 1.959964 * se,
        "z": z, "p_value": p, "q_value": bh_fdr(p),
        "Q": Q, "Q_p_value": np.where(df > 0, stats.chi2.sf(Q, df.clip(lower=1)), np.nan),
        "I2": np.where(Q > 0, ((Q - df) / Q.where(Q > 0)).clip(lower=0.0), 0.0),
        "tau2": tau2, "k": sums.k, "n_positive": sums.n_positive,
    })
    out.index.name = by
    return out


def effect_size(
    adata: ad.AnnData,
    *,
    treatment_key: str = "Treatment",
    batch_key: str = "Batch",
    noise_key: str | None = None,
    layer: str | None = None,
    features: Sequence[str] | None = None,
) -> pd.Series:
    """Distance of each treatment's mean profile from the origin, corrected for replicate noise.

    For profiles centered on a reference (0 = vehicle, e.g. after
    ``robust_reference_normalize``), the root mean square over features of a
    treatment's mean overstates its effect: averaging n noisy wells leaves
    noise variance s^2 / n in every feature. s^2 is estimated per batch and
    feature as the pooled variance of the replicate residuals of the
    ``noise_key`` groups on that batch (default ``treatment_key``), so
    treatments on noisier plates are corrected more. Returns, per treatment,
    sqrt(max(0, mean over features of (mean^2 - sum over wells of s^2 / n^2))),
    in the units of the data.

    Pass ``noise_key`` when the scored groups are not replicates, e.g.
    ``scrambled_treatments`` labels: their spread reflects different
    treatments, so take s^2 from the real ones. Wells with a missing
    ``treatment_key`` then still inform the noise but are not scored.
    """
    for key in (treatment_key, noise_key or treatment_key):
        if key not in adata.obs:
            raise KeyError(f"Missing {key!r} in obs.")
    noise_labels = adata.obs[noise_key or treatment_key]
    if noise_labels.isna().any():
        raise ValueError(f"{noise_key or treatment_key!r} must not be missing.")
    data = adata if features is None else adata[:, list(features)]
    if data.n_vars == 0:
        raise ValueError("Select at least one feature.")
    X = np.asarray(_source(data, layer), dtype=np.float64)
    batch = _batches(adata, batch_key)
    noise_group = noise_labels.astype(str).to_numpy()

    variance = np.empty_like(X)
    for b in pd.unique(batch):
        wells = batch == b
        residual = X[wells] - pd.DataFrame(X[wells]).groupby(noise_group[wells]).transform("mean").to_numpy()
        dof = int(wells.sum()) - len(np.unique(noise_group[wells]))
        if dof < 1:
            raise ValueError(f"Batch {b!r} has no replicate wells to estimate noise from.")
        variance[wells] = (residual ** 2).sum(axis=0) / dof

    label = adata.obs[treatment_key]
    scored = label.notna().to_numpy()
    if not scored.any():
        raise ValueError(f"No wells with a {treatment_key!r} value.")
    label = label[scored].astype(str).to_numpy()
    grouped = pd.DataFrame(X[scored]).groupby(label)
    means, counts = grouped.mean(), grouped.size()
    noise = pd.DataFrame(variance[scored]).groupby(label).sum().loc[means.index].to_numpy() / (counts.to_numpy()[:, None] ** 2)
    debiased = ((means.to_numpy() ** 2) - noise).mean(axis=1)
    return pd.Series(np.sqrt(np.clip(debiased, 0, None)), index=means.index.rename(treatment_key), name="effect_size")
