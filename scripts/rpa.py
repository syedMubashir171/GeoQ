"""Riemannian Procrustes Analysis, and whether its extra steps scale with displacement.

What this tests
---------------
Section 6.4 of the manuscript derives a prediction it does not test. If the
discrepancy surviving re-centring is rotation-reducible, and if the
rotation-reducible part grows with inter-subject displacement, then a method
that adds a rotation step should outperform plain re-centring by an amount
that also grows with displacement. Riemannian Procrustes Analysis
(Rodrigues et al. 2019) is that method, and this script runs it.

The comparison is between re-centring alone and re-centring followed by the
stretching and rotation steps, on identical folds. What is correlated with
displacement is the difference between them, not the performance of either.

The supervision cost is real and is charged honestly
----------------------------------------------------
The rotation is estimated from labelled trials in both domains, so RPA is
supervised where re-centring is not. To make that cost visible rather than
hidden, each held-out subject's trials are split chronologically: the first
half supplies the labelled calibration data the rotation needs, and only the
second half is classified. Re-centring is given the same split, so the two
methods are compared on identical evaluation trials and differ only in what
they do with the calibration half.

This is stricter than the usual reporting of RPA, which often evaluates on
all target trials while estimating the rotation from a subset of them.

Cost and scope
--------------
The rotation is an optimisation over SO(n), with n(n-1)/2 free parameters:
28 at eight electrodes and 153 at eighteen. Measured here, a single fit takes
about half a second at eight channels and one and a half at eighteen, which
is affordable at both. Three random restarts are used because the objective
is not convex on the rotation group, and the best is kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from geoq.datasets import load_dataset
from geoq.features.alignment import recenter
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import distance_airm, frechet_mean
from geoq.models.classical.mdm import MDM

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper1"
)

BANDS = [
    (4, 8, "theta"),
    (8, 13, "mu"),
    (13, 30, "beta"),
    (30, 45, "low_gamma"),
    (8, 30, "mu_beta"),
    (4, 40, "broad"),
]
COUNTS = (8, 18)
N_SUBSETS = 3
N_RESTARTS = 3


def dispersion(covariances: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared geodesic distance from a set to its reference."""
    return float(
        np.mean([float(distance_airm(c, reference)) ** 2 for c in covariances])
    )


def stretch(covariances: np.ndarray, scale: float) -> np.ndarray:
    """Rescale dispersion about the identity by a power map.

    After re-centring the reference is the identity, and moving each point
    along its geodesic from the identity by a factor ``scale`` multiplies
    every distance to the identity by that factor. For an SPD matrix this is
    the matrix power.

    Args:
        covariances: Re-centred SPD stack.
        scale: Multiplicative factor applied to distances.

    Returns:
        The rescaled stack.
    """
    out = np.empty_like(covariances)
    for index, matrix in enumerate(covariances):
        values, vectors = np.linalg.eigh(matrix)
        out[index] = vectors @ np.diag(values**scale) @ vectors.T
    return out


def procrustes_rotation(source_means, target_means, seed: int = 0):
    """Find the rotation best aligning target class means to source ones.

    Solves ``min_Q sum_k d(Q T_k Q^T, S_k)^2`` over the rotation group, with
    Q parametrised as the exponential of a skew-symmetric matrix so that the
    constraint is satisfied by construction rather than enforced.

    Args:
        source_means: Class-conditional means of the source domain.
        target_means: Class-conditional means of the target domain.
        seed: Seed for the random restarts.

    Returns:
        The rotation matrix, and the objective before and after.
    """
    n = source_means[0].shape[0]
    upper = np.triu_indices(n, 1)
    rng = np.random.default_rng(seed)

    def objective(vector: np.ndarray) -> float:
        skew = np.zeros((n, n))
        skew[upper] = vector
        skew = skew - skew.T
        rotation = expm(skew)
        return sum(
            float(distance_airm(rotation @ t @ rotation.T, s)) ** 2
            for s, t in zip(source_means, target_means, strict=True)
        )

    before = objective(np.zeros(upper[0].size))
    best_value, best_vector = np.inf, np.zeros(upper[0].size)
    for restart in range(N_RESTARTS):
        #  The first start is the identity rotation, so the result can never
        #  be worse than doing nothing; the others explore, since the
        #  objective is not convex on the rotation group.
        start = (
            np.zeros(upper[0].size)
            if restart == 0
            else rng.normal(0, 0.3, upper[0].size)
        )
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 200, "maxfun": 3000},
        )
        if result.fun < best_value:
            best_value, best_vector = result.fun, result.x

    skew = np.zeros((n, n))
    skew[upper] = best_vector
    skew = skew - skew.T
    return expm(skew), before, best_value


def split_halves(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each subject's trials chronologically into two halves."""
    first = np.zeros(subjects.shape[0], dtype=bool)
    for subject in np.unique(subjects):
        positions = np.flatnonzero(subjects == subject)
        first[positions[: positions.size // 2]] = True
    return first, ~first


def run_fold(covariances, labels, subjects, target, classes):
    """Compare re-centring against RPA for one held-out subject.

    Returns:
        Mapping with the kappa of each method and the rotation's objective
        reduction, or None if the fold cannot be evaluated.
    """
    from sklearn.metrics import cohen_kappa_score

    calib_mask, eval_mask = split_halves(subjects)
    is_target = subjects == target

    #  Source domain: every other subject, each re-centred on its own mean.
    source = np.empty_like(covariances)
    for subject in np.unique(subjects):
        cell = subjects == subject
        source[cell] = recenter(
            covariances[cell], frechet_mean(covariances[cell]), validate=False
        )

    #  Target: the reference comes from the calibration half only, so no
    #  evaluated trial contributes to the transform applied to it.
    target_ref = frechet_mean(covariances[is_target & calib_mask])
    target_all = recenter(covariances[is_target], target_ref, validate=False)
    target_calib = recenter(
        covariances[is_target & calib_mask], target_ref, validate=False
    )
    target_eval = recenter(
        covariances[is_target & eval_mask], target_ref, validate=False
    )

    train = source[~is_target]
    train_labels = labels[~is_target]
    eval_labels = labels[is_target & eval_mask]
    calib_labels = labels[is_target & calib_mask]

    if len(set(eval_labels)) < 2 or len(set(calib_labels)) < 2:
        return None

    identity = np.eye(covariances.shape[-1])
    scale = np.sqrt(dispersion(train, identity) / dispersion(target_all, identity))
    stretched_calib = stretch(target_calib, scale)
    stretched_eval = stretch(target_eval, scale)

    source_means = [frechet_mean(train[train_labels == k]) for k in classes]
    calib_means = [frechet_mean(stretched_calib[calib_labels == k]) for k in classes]
    rotation, before, after = procrustes_rotation(source_means, calib_means)
    rotated_eval = np.stack([rotation @ c @ rotation.T for c in stretched_eval])

    scores = {}
    for name, model in (
        ("mdm", MDM()),
        ("ts_lda", make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())),
    ):
        for method, features in (
            ("rct", target_eval),
            ("rpa", rotated_eval),
        ):
            from sklearn.base import clone

            fitted = clone(model).fit(train, train_labels)
            scores[f"{name}_{method}"] = float(
                cohen_kappa_score(eval_labels, fitted.predict(features))
            )
    scores["rotation_before"] = before
    scores["rotation_after"] = after
    scores["stretch_scale"] = float(scale)
    return scores


def sweep(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Run RPA against re-centring across bands and channel counts."""
    for low, high, band in BANDS:
        out = RESULTS / f"I_rpa_{dataset_name}_{band}.csv"
        if out.exists():
            print(f"{band}: cached", flush=True)
            continue

        data = load_dataset(
            dataset_name, cache_dir=CACHE, low_freq=float(low), high_freq=float(high)
        )
        classes = sorted(set(data.labels.tolist()))
        rows = []
        for n_channels in COUNTS:
            rng = np.random.default_rng(0)
            for seed in range(N_SUBSETS):
                channels = np.sort(
                    rng.choice(data.n_channels, n_channels, replace=False)
                )
                covariances = Covariances(
                    estimator="oas", audit_conditioning=False
                ).fit_transform(data.epochs[:, channels, :])

                #  Displacement is computed on the unaligned covariances, as
                #  everywhere else in the paper.
                references = {
                    s: frechet_mean(covariances[data.subjects == s])
                    for s in np.unique(data.subjects)
                }
                keys = sorted(references)
                displacement = float(
                    np.mean(
                        [
                            float(distance_airm(references[a], references[b]))
                            for i, a in enumerate(keys)
                            for b in keys[i + 1 :]
                        ]
                    )
                )

                for target in keys:
                    scores = run_fold(
                        covariances, data.labels, data.subjects, target, classes
                    )
                    if scores is None:
                        continue
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "band": band,
                            "n_channels": n_channels,
                            "subset_seed": seed,
                            "subject": target,
                            "displacement": displacement,
                            **scores,
                        }
                    )
                print(f"  {band} {n_channels}ch subset {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{band}: saved", flush=True)


def analyse() -> None:
    """Does the value of the rotation step scale with displacement?"""
    import warnings

    import statsmodels.formula.api as smf
    from scipy import stats

    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

    files = sorted(RESULTS.glob("I_rpa_*.csv"))
    if not files:
        sys.exit(f"No I_rpa_*.csv under {RESULTS}; run the sweep first.")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame.to_csv(RESULTS / "I_rpa_all.csv", index=False)

    for model in ("mdm", "ts_lda"):
        frame[f"{model}_extra"] = frame[f"{model}_rpa"] - frame[f"{model}_rct"]

    collapsed = (
        frame.groupby(["band", "n_channels", "subject"])
        .mean(numeric_only=True)
        .reset_index()
    )
    usable = collapsed[collapsed.band != "theta"].copy()

    print(
        f"\n{len(frame)} folds, {len(collapsed)} subject-condition cells "
        f"({len(usable)} excluding theta)\n"
    )

    print("=" * 72)
    print("Does the rotation step help at all?")
    print("=" * 72)
    print(
        collapsed.groupby("n_channels")[
            [
                "mdm_rct",
                "mdm_rpa",
                "mdm_extra",
                "ts_lda_rct",
                "ts_lda_rpa",
                "ts_lda_extra",
            ]
        ]
        .mean()
        .round(3)
        .to_string()
    )
    print("\nThe rotation is estimated from labelled target trials, so a")
    print("positive 'extra' column is bought with supervision that")
    print("re-centring does not require.")

    print("\n" + "=" * 72)
    print("Does its value scale with displacement?")
    print("=" * 72)
    for model in ("mdm", "ts_lda"):
        column = f"{model}_extra"
        rho, p = stats.spearmanr(usable.displacement, usable[column])
        print(f"\n{model}: extra benefit from rotation ~ displacement")
        print(f"  Spearman rho={rho:+.3f} p={p:.4g} n={len(usable)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(
                f"{column} ~ displacement", usable, groups=usable["subject"]
            ).fit(reml=True)
        low, high = fit.conf_int().loc["displacement"]
        print(
            f"  mixed model slope {fit.params['displacement']:+.4f}  "
            f"[{low:+.4f}, {high:+.4f}]  p={fit.pvalues['displacement']:.4g}"
        )

    print("\n" + "=" * 72)
    print("Rotation objective: how much discrepancy it removes")
    print("=" * 72)
    frame["objective_reduction"] = 1.0 - frame.rotation_after / frame.rotation_before
    print(
        frame.groupby("n_channels")
        .objective_reduction.describe()[["mean", "std", "min", "max"]]
        .round(3)
        .to_string()
    )
    rho, p = stats.spearmanr(frame.displacement, frame.objective_reduction)
    print(f"\n  reduction ~ displacement: rho={rho:+.3f} p={p:.4g} n={len(frame)}")


if __name__ == "__main__":
    if "--analyse" not in sys.argv:
        sweep()
    analyse()
