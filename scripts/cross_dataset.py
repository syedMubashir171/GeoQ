"""Does inter-subject displacement predict alignment benefit across datasets?

The question this answers
-------------------------
The manuscript's Limitations section names this as the most important open
question about the predictor: every manipulation reported there is confined
to BCI Competition IV dataset 2a, so the predictor is shown to work across
conditions within one dataset and not across datasets.

Why matched channel counts are the whole design
-----------------------------------------------
Three datasets give three points, which supports nothing. The useful design
holds the channel count fixed across datasets and lets displacement vary
naturally with montage, paradigm and participant population.

At 8 and 18 electrodes, subsets drawn from dataset 2a and from PhysioNet MI
have identical SPD dimension (36 and 171 respectively). If benefit still
tracks displacement when conditions from both datasets are pooled at a
matched dimension, the relationship cannot be a dimension effect and cannot
be a property of one recording setup. That is a considerably stronger claim
than the within-dataset result, and it is the one a reviewer will ask for.

Dataset 2b contributes at three electrodes only, so it enters as a
low-displacement anchor rather than as a matched condition.

Three questions are reported separately
---------------------------------------
1. Pooled across datasets at matched dimension, does displacement predict
   benefit?
2. Does the relationship hold within each dataset considered alone?
3. Do the datasets fall on a common line, or does each have its own offset?

The third matters most. A predictor that ranks conditions correctly within a
dataset but needs a different intercept per dataset is useful for tuning a
pipeline and useless for deciding whether to align a new recording, which is
the use the manuscript claims for it.

Cost
----
PhysioNet MI is the expensive part: leave-one-subject-out over N subjects
means N model fits per condition, and the Frechet means dominate. Start with
N_SUBJECTS = 12 to confirm the pipeline runs end to end, then raise it. Each
condition is written to its own file and completed files are skipped, so an
interrupted session resumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.alignment import RiemannianAlignment, domain_references
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import distance_airm
from geoq.geometry.spd import condition_number
from geoq.models.classical.mdm import MDM

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path("/content/drive/MyDrive/GeoQ_workspace/results/paper1")

#  PhysioNet has 109 participants; a leave-one-subject-out sweep over all of
#  them is not affordable here and adds little, since the quantity of
#  interest is one displacement value per condition rather than a precise
#  per-subject score. Twelve keeps the fold count comparable to the other
#  two datasets, which also avoids confounding displacement with the number
#  of training subjects.
N_SUBJECTS = 12

BANDS = [
    (4, 8, "theta"),
    (8, 13, "mu"),
    (13, 30, "beta"),
    (30, 45, "low_gamma"),
    (8, 30, "mu_beta"),
    (4, 40, "broad"),
]

#  Channel counts at which each dataset can contribute. 2a and PhysioNet
#  share 8 and 18; 2b has only three electrodes.
DATASETS = {
    "bci_iv_2a_lr": (8, 18),
    "physionet_mi": (8, 18),
    "bci_iv_2b": (3,),
}

N_SUBSETS = 3


def displacement(covariances: np.ndarray, subjects: np.ndarray) -> float:
    """Mean geodesic distance between subjects' reference means."""
    references = domain_references(covariances, subjects)
    keys = sorted(references)
    return float(
        np.mean(
            [
                float(distance_airm(references[a], references[b]))
                for index, a in enumerate(keys)
                for b in keys[index + 1 :]
            ]
        )
    )


def run_condition(dataset, channels, subset_seed: int) -> list[dict]:
    """Measure displacement and alignment benefit for one condition.

    Args:
        dataset: A loaded :class:`geoq.datasets.base.EEGDataset`.
        channels: Indices of the electrodes to use.
        subset_seed: Identifier of the electrode subset, recorded for
            traceability.

    Returns:
        One record per classifier.
    """
    covariances = Covariances(estimator="oas", audit_conditioning=False).fit_transform(
        dataset.epochs[:, channels, :]
    )
    aligned = RiemannianAlignment(assume_calibration_data=True).fit_transform(
        covariances, domains=dataset.subjects
    )

    n_channels = len(channels)
    shared = {
        "n_channels": n_channels,
        "spd_dimension": n_channels * (n_channels + 1) // 2,
        "median_condition_number": float(np.median(condition_number(covariances))),
        "displacement": displacement(covariances, dataset.subjects),
        "subset_seed": subset_seed,
        "n_subjects": int(dataset.n_subjects),
        "n_trials": int(dataset.n_trials),
    }

    splitter = LeaveOneSubjectOut()
    records = []
    for name, model in (
        ("mdm", MDM()),
        ("ts_lda", make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())),
    ):
        scores = {}
        for label, features in (("raw", covariances), ("aligned", aligned)):
            scores[label] = evaluate(
                model,
                features,
                dataset.labels,
                groups=dataset.subjects,
                splitter=splitter,
                metrics=("kappa",),
            ).mean("kappa")
        records.append(
            {
                **shared,
                "model": name,
                "kappa_raw": scores["raw"],
                "kappa_aligned": scores["aligned"],
                "gain": scores["aligned"] - scores["raw"],
            }
        )
    return records


def sweep() -> None:
    """Run every dataset, band and channel count, saving as it goes."""
    for name, counts in DATASETS.items():
        for low, high, band in BANDS:
            out = RESULTS / f"F_cross_{name}_{band}.csv"
            if out.exists():
                print(f"{name} {band}: cached", flush=True)
                continue

            kwargs = {
                "cache_dir": CACHE,
                "low_freq": float(low),
                "high_freq": float(high),
            }
            if name == "physionet_mi":
                kwargs["subjects"] = list(range(1, N_SUBJECTS + 1))
            data = load_dataset(name, **kwargs)

            rows = []
            for n_channels in counts:
                if n_channels > data.n_channels:
                    continue
                rng = np.random.default_rng(0)
                repeats = 1 if n_channels == data.n_channels else N_SUBSETS
                for seed in range(repeats):
                    channels = (
                        np.arange(data.n_channels)
                        if n_channels == data.n_channels
                        else np.sort(
                            rng.choice(data.n_channels, n_channels, replace=False)
                        )
                    )
                    for record in run_condition(data, channels, seed):
                        rows.append(
                            {
                                **record,
                                "dataset": name,
                                "band": band,
                                "low_freq": low,
                                "high_freq": high,
                            }
                        )
                    print(
                        f"  {name} {band} {n_channels}ch subset {seed} done", flush=True
                    )
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{name} {band}: saved", flush=True)


def analyse() -> None:
    """Report the three questions separately."""
    files = sorted(RESULTS.glob("F_cross_*.csv"))
    if not files:
        sys.exit(f"No F_cross_*.csv under {RESULTS}; run the sweep first.")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame.to_csv(RESULTS / "F_cross_all.csv", index=False)

    #  Theta is excluded wherever raw performance is at or below chance, for
    #  the reason given in the manuscript: a benefit is undefined when there
    #  is no performance to improve. The exclusion is applied by measured
    #  performance rather than by band name, so it adapts to datasets where
    #  a different band happens to be undecodable.
    usable = frame[frame.kappa_raw > 0.02]

    print(f"\n{len(frame)} conditions, {len(usable)} with raw kappa > 0.02\n")
    print(
        frame.groupby(["dataset", "band", "n_channels"])[
            ["displacement", "median_condition_number", "kappa_raw", "gain"]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    def rho(x, y) -> str:
        r, p = stats.spearmanr(np.asarray(x, float), np.asarray(y, float))
        return f"rho={r:+.3f} p={p:.4f} n={len(x)}"

    for model in ("mdm", "ts_lda"):
        sub = usable[usable.model == model]
        print(f"\n{'=' * 66}\n{model}\n{'=' * 66}")

        print("\nQ1  pooled across datasets, at matched channel counts")
        for n in sorted(set(sub.n_channels)):
            cell = sub[sub.n_channels == n]
            if cell.dataset.nunique() < 2:
                continue
            print(
                f"  {n:2d} ch ({cell.dataset.nunique()} datasets): "
                f"{rho(cell.displacement, cell.gain)}"
            )
        print(f"  all conditions:  {rho(sub.displacement, sub.gain)}")

        print("\nQ2  within each dataset alone")
        for name in sorted(set(sub.dataset)):
            cell = sub[sub.dataset == name]
            if len(cell) >= 4:
                print(f"  {name:14s} {rho(cell.displacement, cell.gain)}")

        print("\nQ3  common line, or a per-dataset offset?")
        for name in sorted(set(sub.dataset)):
            cell = sub[sub.dataset == name]
            if len(cell) < 4:
                continue
            slope, intercept = np.polyfit(cell.displacement, cell.gain, 1)
            print(
                f"  {name:14s} slope {slope:+.4f}  intercept {intercept:+.4f}"
                f"  displacement {cell.displacement.min():.2f}"
                f"-{cell.displacement.max():.2f}"
            )
        print(
            "\n  Similar slopes with different intercepts would mean the"
            "\n  predictor ranks conditions within a dataset but does not"
            "\n  transfer between them, which is weaker than the manuscript"
            "\n  currently claims."
        )


if __name__ == "__main__":
    if "--analyse" not in sys.argv:
        sweep()
    analyse()
