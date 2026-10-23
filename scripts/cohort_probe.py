"""Feasibility check for a third, larger motor-imagery cohort.

Why
---
Nine subject records per dataset is the paper's main limitation: the
primary comparison can only bound a circuit advantage, and the evidence for
entanglement beyond routing rests on a pooled secondary test. A cohort with
around fifty subjects would let the same paired tests resolve differences
of a few hundredths in kappa.

Cho et al. (2017, GigaScience) recorded left- and right-hand motor imagery
from 52 subjects with 64 EEG channels, and MOABB distributes it as
``Cho2017``. Before committing a day of compute to it, this script checks
what it costs on Colab and whether there is cross-subject signal under our
exact preprocessing (8-30 Hz, 0.5-2.5 s, OAS covariances, tangent space).
The last cohort we tried, PhysioNet, gave chance for every model, so the
signal check matters.

What it does (minutes, plus the download of two subjects)
-----------------------------------------------------------
1. Loads two subjects through MOABB with the paper's preprocessing.
2. Reports channels, trials per subject, class balance, sampling rate,
   download time and the disk used per subject, and extrapolates to the
   full cohort.
3. Trains tangent-space LDA on one subject and tests on the other, in
   both directions, as a crude check that a decoder transfers.

Usage
-----
    !python scripts/cohort_probe.py

Paste the whole output.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

PROBE_SUBJECTS = (1, 2)
FULL_COHORT = 52


def folder_size(path: Path) -> float:
    """Size of a directory tree in gigabytes."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    """Load two subjects, report costs, and test cross-subject transfer."""
    import mne
    import moabb
    from moabb.datasets import Cho2017
    from moabb.paradigms import LeftRightImagery
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    from sklearn.pipeline import make_pipeline

    print(f"MOABB {moabb.__version__}, MNE {mne.__version__}")
    dataset = Cho2017()
    print(
        f"Cho2017 subjects available in MOABB: {len(dataset.subject_list)} "
        f"(of {FULL_COHORT} recorded)"
    )

    data_root = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")
    before = folder_size(data_root)
    paradigm = LeftRightImagery(fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)

    started = time.perf_counter()
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=list(PROBE_SUBJECTS)
    )
    seconds = time.perf_counter() - started
    used = folder_size(data_root) - before

    print(f"\nLoaded subjects {PROBE_SUBJECTS} in {seconds / 60:.1f} min")
    print(f"  epochs shape   : {epochs.shape} (trials, channels, samples)")
    names, counts = np.unique(labels, return_counts=True)
    print(f"  classes        : {dict(zip(names, counts, strict=True))}")
    for subject in PROBE_SUBJECTS:
        mask = (meta.subject == subject).to_numpy()
        print(f"  subject {subject}: {mask.sum()} trials")
    print(f"  disk used      : {used:.2f} GB for {len(PROBE_SUBJECTS)} subjects")
    per_subject = used / len(PROBE_SUBJECTS)
    cohort = len(dataset.subject_list)
    print(
        f"  extrapolated   : ~{per_subject * cohort:.1f} GB and "
        f"~{seconds / len(PROBE_SUBJECTS) * cohort / 60:.0f} min to download "
        f"all {cohort} subjects"
    )
    free = (
        shutil.disk_usage(data_root if data_root.exists() else Path.home()).free / 1e9
    )
    print(f"  free disk now  : {free:.1f} GB")

    #  Crude transfer check: one subject trains, the other tests.
    model = make_pipeline(
        Covariances("oas"),
        TangentSpace(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    subjects = meta.subject.to_numpy()
    print("\nCross-subject tangent-space LDA (train -> test)")
    for train_subject, test_subject in (PROBE_SUBJECTS, PROBE_SUBJECTS[::-1]):
        train, test = subjects == train_subject, subjects == test_subject
        model.fit(epochs[train], labels[train])
        kappa = cohen_kappa_score(labels[test], model.predict(epochs[test]))
        print(f"  {train_subject} -> {test_subject}: kappa {kappa:+.3f}")
    print("\nDone. Paste this output in full.")


if __name__ == "__main__":
    main()
