"""Subject-level analysis of the displacement--benefit relationship.

What this adds to the condition-level results
---------------------------------------------
Every analysis in the manuscript so far treats a band and channel count as
one observation: displacement is averaged over all subject pairs, and
benefit over all folds. That tests whether a preprocessing choice with more
inter-subject spread yields more benefit, which is a statement about
conditions rather than about people.

The question a clinician asks is different. For a particular participant,
does sitting further from the rest of the group predict a larger gain from
re-centring? The two can come apart: benefit might track the group's overall
spread while being unrelated to any individual's position within it.

Two quantities are therefore defined per subject and condition. The
subject's displacement is the mean geodesic distance from that subject's
reference matrix to every other subject's, rather than the mean over all
pairs. The subject's benefit is the change in kappa on that subject's own
leave-one-subject-out fold. Both are computed within the condition, so the
subject-level values reduce to the condition-level ones when averaged.

Why a mixed model rather than a correlation
-------------------------------------------
Each subject appears in all twelve conditions, so the 108 observations are
not independent, and a Spearman correlation over them would treat repeated
measurements of the same nine people as 108 separate facts. A linear mixed
model with a random intercept and slope per subject accounts for that
dependence directly.

The fixed effect of interest is the displacement by classifier interaction.
The mechanism predicts opposite signs for the two classifier families, so
the interaction is the term that carries the prediction, and its sign and
confidence interval are what the manuscript should report.

The three random electrode subsets are averaged before modelling, as the
review specified, giving twelve conditions of six bands by two channel
counts and 108 subject-condition cells per classifier.
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
from geoq.features.alignment import RiemannianAlignment, domain_references
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import distance_airm
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


def subject_displacements(covariances, subjects) -> dict:
    """Mean geodesic distance from each subject's reference to the others.

    Args:
        covariances: SPD stack of shape ``(n_trials, n, n)``.
        subjects: Subject identifier per trial.

    Returns:
        Mapping from subject to its mean distance from the rest.
    """
    references = domain_references(covariances, subjects)
    keys = sorted(references)
    return {
        s: float(
            np.mean(
                [
                    float(distance_airm(references[s], references[other]))
                    for other in keys
                    if other != s
                ]
            )
        )
        for s in keys
    }


def per_subject_scores(model, features, labels, subjects) -> dict:
    """Return each subject's kappa on its own leave-one-subject-out fold.

    Args:
        model: Unfitted estimator.
        features: Feature array.
        labels: Class labels.
        subjects: Subject identifier per trial.

    Returns:
        Mapping from subject to kappa.
    """
    result = evaluate(
        model,
        features,
        labels,
        groups=subjects,
        splitter=LeaveOneSubjectOut(),
        metrics=("kappa",),
    )
    scores = {}
    for fold in result.folds:
        #  Each leave-one-subject-out fold holds out exactly one subject, so
        #  test_groups has one element and the fold score is that subject's.
        scores[fold.test_groups[0]] = fold.scores["kappa"]
    return scores


def sweep(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Record per-subject displacement and benefit for every condition."""
    for low, high, band in BANDS:
        out = RESULTS / f"G_subject_{dataset_name}_{band}.csv"
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
                covariances = Covariances(
                    estimator="oas", audit_conditioning=False
                ).fit_transform(data.epochs[:, channels, :])
                aligned = RiemannianAlignment(
                    assume_calibration_data=True
                ).fit_transform(covariances, domains=data.subjects)

                displacement = subject_displacements(covariances, data.subjects)

                for name, model in (
                    ("mdm", MDM()),
                    (
                        "ts_lda",
                        make_pipeline(TangentSpace(), LinearDiscriminantAnalysis()),
                    ),
                ):
                    raw = per_subject_scores(
                        model, covariances, data.labels, data.subjects
                    )
                    ali = per_subject_scores(model, aligned, data.labels, data.subjects)
                    for subject in sorted(raw):
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "band": band,
                                "n_channels": n_channels,
                                "subset_seed": seed,
                                "subject": subject,
                                "classifier": name,
                                "displacement": displacement[subject],
                                "kappa_raw": raw[subject],
                                "kappa_aligned": ali[subject],
                                "gain": ali[subject] - raw[subject],
                            }
                        )
                print(f"  {band} {n_channels}ch subset {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{band}: saved", flush=True)


def analyse() -> None:
    """Collapse subsets and fit the mixed model."""
    import warnings

    import statsmodels.formula.api as smf

    #  Convergence warnings are expected and are not failures. They arise
    #  when a variance component is estimated at or near zero, which happens
    #  whenever between-subject heterogeneity is small; the fit is still
    #  usable and the estimate is reported with its interval. Suppressing
    #  them keeps the output readable, and the variance itself is printed
    #  below so a reader can see when this occurred.
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

    files = sorted(RESULTS.glob("G_subject_*.csv"))
    if not files:
        sys.exit(f"No G_subject_*.csv under {RESULTS}; run the sweep first.")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame.to_csv(RESULTS / "G_subject_all.csv", index=False)

    #  The three electrode subsets are averaged, as specified, leaving one
    #  observation per subject, band, channel count and classifier.
    collapsed = (
        frame.groupby(["dataset", "band", "n_channels", "subject", "classifier"])[
            ["displacement", "kappa_raw", "kappa_aligned", "gain"]
        ]
        .mean()
        .reset_index()
    )

    #  Theta is excluded on the condition mean rather than on each subject's
    #  own raw score. Excluding by the subject's own kappa_raw would condition
    #  on part of the outcome, since gain is defined as aligned minus raw, and
    #  would bias the relationship being tested.
    condition_raw = (
        collapsed.groupby(["band", "n_channels", "classifier"])
        .kappa_raw.mean()
        .rename("condition_raw")
        .reset_index()
    )
    collapsed = collapsed.merge(condition_raw, on=["band", "n_channels", "classifier"])
    usable = collapsed[collapsed.condition_raw > 0.02].copy()

    n_cond = usable.groupby("classifier").apply(
        lambda d: d[["band", "n_channels"]].drop_duplicates().shape[0]
    )
    print(
        f"\n{len(collapsed)} subject-condition observations "
        f"({len(usable)} after excluding undecodable conditions)"
    )
    print(f"conditions retained per classifier:\n{n_cond.to_string()}\n")

    print("=" * 70)
    print("Mixed model: gain ~ displacement * classifier, random by subject")
    print("=" * 70)

    #  A random slope is attempted first. Where the slope variance is
    #  estimated at zero the fit does not converge, and an intercept-only
    #  random effect is used instead; which was used is reported, since the
    #  two make different assumptions about between-subject heterogeneity.
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
    between = (
        float(model.cov_re.iloc[0, 0])
        if hasattr(model.cov_re, "iloc")
        else float(np.atleast_2d(model.cov_re)[0, 0])
    )
    residual = float(model.scale)
    print(
        f"\nbetween-subject variance {between:.5f}, residual {residual:.5f}"
        f"  (ICC {between / (between + residual):.3f})"
    )

    slope_mdm = model.params["displacement"]
    interaction = model.params["displacement:classifier[T.ts_lda]"]
    conf = model.conf_int()
    print(
        f"\nMDM slope            {slope_mdm:+.4f}  "
        f"[{conf.loc['displacement', 0]:+.4f}, "
        f"{conf.loc['displacement', 1]:+.4f}]"
    )
    print(
        f"interaction          {interaction:+.4f}  "
        f"[{conf.loc['displacement:classifier[T.ts_lda]', 0]:+.4f}, "
        f"{conf.loc['displacement:classifier[T.ts_lda]', 1]:+.4f}]  "
        f"p={model.pvalues['displacement:classifier[T.ts_lda]']:.4g}"
    )
    print(f"implied TS+LDA slope {slope_mdm + interaction:+.4f}")
    print(
        "\nThe interaction is the term carrying the prediction: a negative"
        "\nvalue larger in magnitude than the MDM slope means the two"
        "\nclassifiers respond to displacement with opposite sign."
    )

    #  A per-classifier model is also reported, since a reader will want the
    #  simple slope for each rather than only the contrast between them.
    print("\n" + "=" * 70)
    print("Separate models, one per classifier")
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


if __name__ == "__main__":
    if "--analyse" not in sys.argv:
        sweep()
    analyse()
