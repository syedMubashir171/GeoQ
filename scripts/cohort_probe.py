"""Leave-one-subject-out feasibility check for a larger motor-imagery cohort.

Why
---
The first probe trained on one subject and tested on another. With 64
channels the tangent vector has 2080 dimensions, so a single subject's 200
trials cannot fit a transferable decoder, and a chance result there says
little. The question that matters for the paper is whether a decoder
trained on many subjects transfers to a new one, which is what
leave-one-subject-out measures.

This probe loads the first N subjects of a MOABB cohort with the paper's
preprocessing (8-30 Hz, 0.5-2.5 s after the cue, OAS covariances) and runs
leave-one-subject-out among them with two standard decoders:

ts_slda  tangent space + shrinkage linear discriminant analysis
ts_lr    tangent space + standardisation + logistic regression (C = 0.01)

The reference point is fitted on the training subjects of each fold.

Decision rule
-------------
If the mean kappa of the better decoder stays below about 0.10, the cohort
sits too close to chance for paired tests between quantum and classical
models to gain power from its size, and we do not run it.

Usage (repository root not required)
------------------------------------
    !python scripts/cohort_probe.py Cho2017 10
    !python scripts/cohort_probe.py Lee2019_MI 10

The first argument is the MOABB dataset class, the second the number of
subjects to load (default 10).
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

DATASET = sys.argv[1] if len(sys.argv) > 1 else "Cho2017"
N_SUBJECTS = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def folder_size(path: Path) -> float:
    """Size of a directory tree in gigabytes."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    """Load N subjects, report costs, and run leave-one-subject-out."""
    import mne
    import moabb
    import moabb.datasets as moabb_datasets
    from moabb.paradigms import LeftRightImagery
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import cohen_kappa_score
    from sklearn.preprocessing import StandardScaler

    print(f"MOABB {moabb.__version__}, MNE {mne.__version__}")
    dataset = getattr(moabb_datasets, DATASET)()
    available = list(dataset.subject_list)
    chosen = available[:N_SUBJECTS]
    print(f"{DATASET}: {len(available)} subjects available; probing {chosen}")

    data_root = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")
    before = folder_size(data_root)
    paradigm = LeftRightImagery(fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)
    started = time.perf_counter()
    epochs, labels, meta = paradigm.get_data(dataset=dataset, subjects=chosen)
    seconds = time.perf_counter() - started
    used = folder_size(data_root) - before

    names, counts = np.unique(labels, return_counts=True)
    print(
        f"\nLoaded in {seconds / 60:.1f} min; epochs {epochs.shape} "
        f"(trials, channels, samples)"
    )
    print(f"  classes      : {dict(zip(names, counts, strict=True))}")
    print(
        f"  disk used    : {used:.2f} GB; extrapolated to all "
        f"{len(available)} subjects: ~{used / len(chosen) * len(available):.1f} GB"
    )
    free = shutil.disk_usage(Path.home()).free / 1e9
    print(f"  free disk    : {free:.1f} GB")

    covariances = Covariances("oas").fit_transform(epochs)
    subjects = meta.subject.to_numpy()
    scores = {"ts_slda": [], "ts_lr": []}
    print("\nLeave-one-subject-out kappa")
    for held_out in chosen:
        train, test = subjects != held_out, subjects == held_out
        reference = TangentSpace(metric="riemann").fit(covariances[train])
        x_train = reference.transform(covariances[train])
        x_test = reference.transform(covariances[test])

        slda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        slda.fit(x_train, labels[train])
        scores["ts_slda"].append(cohen_kappa_score(labels[test], slda.predict(x_test)))

        scaler = StandardScaler().fit(x_train)
        logistic = LogisticRegression(C=0.01, max_iter=5000)
        logistic.fit(scaler.transform(x_train), labels[train])
        scores["ts_lr"].append(
            cohen_kappa_score(labels[test], logistic.predict(scaler.transform(x_test)))
        )
        print(
            f"  subject {held_out:3d}: ts_slda {scores['ts_slda'][-1]:+.3f}  "
            f"ts_lr {scores['ts_lr'][-1]:+.3f}"
        )

    print("\nMean (sd) over held-out subjects")
    for name, values in scores.items():
        values = np.asarray(values)
        print(
            f"  {name:8s} {values.mean():+.3f} ({values.std(ddof=1):.3f}); "
            f"above 0.10 for {(values > 0.10).sum()} of {values.size}"
        )
    best = max(np.mean(v) for v in scores.values())
    verdict = "worth running" if best >= 0.10 else "too close to chance to add power"
    print(f"\nVerdict by the 0.10 rule: {verdict} (best mean {best:+.3f})")
    print("Done. Paste this output in full.")


if __name__ == "__main__":
    main()
