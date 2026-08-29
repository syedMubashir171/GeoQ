"""Split-sample robustness check for the displacement--benefit relationship.

The concern this addresses
--------------------------
In the main analysis, a subject's displacement and that subject's alignment
benefit are computed from the same trials. A participant whose recording is
unusual will therefore tend to have both an unusual reference matrix, and so
a large displacement, and an unusual fold score. Part of the measured
relationship could be shared estimation noise rather than a dependence
between the two quantities.

The design
----------
Each subject's trials are split in half chronologically. The first half
supplies the reference matrix used for re-centring and the displacement
value; the second half supplies every trial that is classified. No trial
contributes to both the estimate of displacement and the measurement of
benefit.

The split is chronological rather than random because a random split would
place adjacent trials, which share drift and alertness, on both sides of the
partition. That would leak precisely the within-subject autocorrelation the
split is meant to exclude.

This is a robustness check and not a replacement for the main results. It
halves the data available for both alignment and evaluation, so its absolute
kappa values are lower and are not comparable with the tables reported for
the full data. What is comparable, and what this script tests, is whether
the sign and approximate magnitude of the displacement relationship survive
when the two quantities no longer share any trial.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
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


def split_indices(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each subject's trials in half, chronologically.

    Args:
        subjects: Subject identifier per trial, in recording order.

    Returns:
        Boolean masks selecting the reference half and the evaluation half.

    Raises:
        ValueError: If any subject has fewer than four trials, which leaves
            too few on either side to estimate a reference or to score.
    """
    reference = np.zeros(subjects.shape[0], dtype=bool)
    for subject in np.unique(subjects):
        positions = np.flatnonzero(subjects == subject)
        if positions.size < 4:
            raise ValueError(
                f"Subject {subject!r} has {positions.size} trials, too few to "
                f"split. A reference estimated from one or two covariances is "
                f"not a reference."
            )
        reference[positions[: positions.size // 2]] = True
    return reference, ~reference


def split_sample_condition(epochs, labels, subjects, channels):
    """Measure displacement and benefit on disjoint halves of each subject.

    Args:
        epochs: Raw epochs of shape ``(n_trials, n_channels, n_times)``.
        labels: Class labels.
        subjects: Subject identifier per trial, in recording order.
        channels: Indices of the electrodes to use.

    Returns:
        Tuple of per-subject displacement, the aligned evaluation
        covariances, the unaligned evaluation covariances, and the labels and
        subjects restricted to the evaluation half.
    """
    covariances = Covariances(estimator="oas", audit_conditioning=False).fit_transform(
        epochs[:, channels, :]
    )
    ref_mask, eval_mask = split_indices(subjects)

    #  One reference per subject, from that subject's first half only.
    references = {
        subject: frechet_mean(covariances[ref_mask & (subjects == subject)])
        for subject in np.unique(subjects)
    }

    keys = sorted(references)
    displacement = {
        s: float(
            np.mean(
                [
                    float(distance_airm(references[s], references[o]))
                    for o in keys
                    if o != s
                ]
            )
        )
        for s in keys
    }

    #  The evaluation half is re-centred using the references estimated from
    #  the reference half, so no trial that is classified contributed to the
    #  transform applied to it.
    raw = covariances[eval_mask]
    aligned = np.empty_like(raw)
    eval_subjects = subjects[eval_mask]
    for subject in keys:
        cell = eval_subjects == subject
        aligned[cell] = recenter(raw[cell], references[subject], validate=False)

    return displacement, aligned, raw, labels[eval_mask], eval_subjects


def per_subject_scores(model, features, labels, subjects) -> dict:
    """Return each subject's kappa on its own leave-one-subject-out fold."""
    result = evaluate(
        model,
        features,
        labels,
        groups=subjects,
        splitter=LeaveOneSubjectOut(),
        metrics=("kappa",),
    )
    return {f.test_groups[0]: f.scores["kappa"] for f in result.folds}


def sweep(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Run every band and channel count under the split-sample design."""
    for low, high, band in BANDS:
        out = RESULTS / f"H_split_{dataset_name}_{band}.csv"
        if out.exists():
            print(f"{band}: cached", flush=True)
            continue

        data = load_dataset(
            dataset_name, cache_dir=CACHE, low_freq=float(low), high_freq=float(high)
        )
        rows = []
        for n_channels in COUNTS:
            rng = np.random.default_rng(0)
            for seed in range(N_SUBSETS):
                channels = np.sort(
                    rng.choice(data.n_channels, n_channels, replace=False)
                )
                displacement, aligned, raw, labels, subjects = split_sample_condition(
                    data.epochs, data.labels, data.subjects, channels
                )
                for name, model in (
                    ("mdm", MDM()),
                    (
                        "ts_lda",
                        make_pipeline(TangentSpace(), LinearDiscriminantAnalysis()),
                    ),
                ):
                    before = per_subject_scores(model, raw, labels, subjects)
                    after = per_subject_scores(model, aligned, labels, subjects)
                    for subject in sorted(before):
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "band": band,
                                "n_channels": n_channels,
                                "subset_seed": seed,
                                "subject": subject,
                                "classifier": name,
                                "displacement": displacement[subject],
                                "kappa_raw": before[subject],
                                "kappa_aligned": after[subject],
                                "gain": after[subject] - before[subject],
                            }
                        )
                print(f"  {band} {n_channels}ch subset {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{band}: saved", flush=True)


def analyse() -> None:
    """Fit the same mixed model as the main analysis, on the split data."""
    import warnings

    import statsmodels.formula.api as smf

    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

    files = sorted(RESULTS.glob("H_split_*.csv"))
    if not files:
        sys.exit(f"No H_split_*.csv under {RESULTS}; run the sweep first.")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame.to_csv(RESULTS / "H_split_all.csv", index=False)

    collapsed = (
        frame.groupby(["dataset", "band", "n_channels", "subject", "classifier"])[
            ["displacement", "kappa_raw", "kappa_aligned", "gain"]
        ]
        .mean()
        .reset_index()
    )
    condition_raw = (
        collapsed.groupby(["band", "n_channels", "classifier"])
        .kappa_raw.mean()
        .rename("condition_raw")
        .reset_index()
    )
    collapsed = collapsed.merge(condition_raw, on=["band", "n_channels", "classifier"])
    usable = collapsed[collapsed.condition_raw > 0.02].copy()

    print(
        f"\n{len(collapsed)} subject-condition observations "
        f"({len(usable)} after excluding undecodable conditions)"
    )
    print(
        "\nHalving the data lowers absolute performance, so these kappa "
        "values\nare not comparable with the full-data tables. What is "
        "comparable is\nthe sign and magnitude of the displacement slopes.\n"
    )
    print(
        usable.groupby("classifier")[["kappa_raw", "kappa_aligned", "gain"]]
        .mean()
        .round(3)
        .to_string()
    )

    print("\n" + "=" * 70)
    print("Mixed model on disjoint halves: gain ~ displacement * classifier")
    print("=" * 70)
    model = None
    for formula, label in (
        ("~displacement", "random intercept and slope"),
        (None, "random intercept only"),
    ):
        try:
            fit = smf.mixedlm(
                "gain ~ displacement * classifier",
                usable,
                groups=usable["subject"],
                re_formula=formula,
            ).fit(reml=True)
            if fit.converged:
                model, used = fit, label
                break
        except Exception as error:
            print(f"  {label} failed: {error}")
    if model is None:
        sys.exit("Neither random-effects structure converged.")

    print(f"random-effects structure used: {used}\n")
    print(model.summary().tables[1].to_string())

    key = "displacement:classifier[T.ts_lda]"
    conf = model.conf_int()
    print(
        f"\nMDM slope    {model.params['displacement']:+.4f}  "
        f"[{conf.loc['displacement', 0]:+.4f}, "
        f"{conf.loc['displacement', 1]:+.4f}]"
    )
    print(
        f"interaction  {model.params[key]:+.4f}  "
        f"[{conf.loc[key, 0]:+.4f}, {conf.loc[key, 1]:+.4f}]  "
        f"p={model.pvalues[key]:.4g}"
    )

    print("\n" + "=" * 70)
    print("Separate models, for comparison with the full-data analysis")
    print("=" * 70)
    for classifier in ("mdm", "ts_lda"):
        cell = usable[usable.classifier == classifier]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm("gain ~ displacement", cell, groups=cell["subject"]).fit(
                reml=True
            )
        low, high = fit.conf_int().loc["displacement"]
        print(
            f"  {classifier:7s} slope {fit.params['displacement']:+.4f}  "
            f"[{low:+.4f}, {high:+.4f}]  "
            f"p={fit.pvalues['displacement']:.4g}  n={len(cell)}"
        )
    print(
        "\nFull data gave +0.0406 [+0.0173, +0.0639] for MDM and "
        "-0.0208\n[-0.0351, -0.0065] for TS+LDA. Overlapping intervals "
        "here would\nindicate the relationship does not depend on the two "
        "quantities\nsharing trials."
    )


if __name__ == "__main__":
    if "--analyse" not in sys.argv:
        sweep()
    analyse()
