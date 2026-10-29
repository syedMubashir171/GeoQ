"""Stream Lee2019_MI into per-subject covariance files on Drive.

Why streaming
-------------
Lee et al. (2019, GigaScience) recorded left- and right-hand motor imagery
from 54 subjects with 62 EEG channels at 1000 Hz. The raw files take about
130 GB, more than a Colab disk holds. The experiment never needs the raw
signal, only each trial's spatial covariance matrix, which is 62 x 62. So
each subject is downloaded, epoched with the paper's preprocessing,
reduced to OAS covariances, saved to Drive, and its raw files are deleted
before the next subject is fetched. Peak disk use stays near one subject.

Preprocessing (identical to the main datasets)
----------------------------------------------
MOABB LeftRightImagery paradigm, 8-30 Hz band-pass, 0.5-2.5 s after the
cue, EEG channels only, resampled to 250 Hz to match BCI IV 2a and 2b,
covariances by the same OAS estimator used for the main datasets.

Resumable: a subject whose file already exists on Drive is skipped.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/lee2019_prepare.py

Output
------
{CACHE}/lee2019/subject_{NN}.npz    covariances (trials, 62, 62), labels
{CACHE}/lee2019/summary.csv         trials per subject and class
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seeds_stats import CACHE

from geoq.features.covariance import Covariances

OUT = Path(CACHE) / "lee2019"
RESAMPLE = 250.0
EXPECTED_CHANNELS = 62
#: Stop before the disk fills if raw files are not being deleted.
MIN_FREE_GB = 5.0


#: Folders searched for files that appear while a subject downloads. The
#: mounted Drive is excluded: it is slow to scan and holds only our outputs.
SCAN_ROOTS = (Path.home(), Path("/tmp"))
#: Smaller files (configs, manifests) are left alone.
MIN_DELETE_BYTES = 1_000_000


def snapshot() -> set[Path]:
    """Every file currently under the scan roots, outside the Drive mount."""
    files = set()
    for root in SCAN_ROOTS:
        if root.exists():
            for path in root.rglob("*"):
                try:
                    if path.is_file() and "/content/drive" not in str(path):
                        files.add(path)
                except OSError:
                    continue
    return files


def delete_new(before: set[Path]) -> float:
    """Delete large files created since ``before``; return gigabytes freed.

    MOABB's reported paths do not always point at the files its download
    backend writes, so the files to remove are found by difference: whatever
    appeared on disk while this subject was loaded.
    """
    freed = 0.0
    for path in snapshot() - before:
        try:
            size = path.stat().st_size
            if size >= MIN_DELETE_BYTES:
                path.unlink()
                freed += size / 1e9
        except OSError:
            continue
    return freed


def main() -> None:
    """Download, reduce and delete one subject at a time."""
    import moabb.datasets as moabb_datasets
    from moabb.paradigms import LeftRightImagery

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = moabb_datasets.Lee2019_MI()
    paradigm = LeftRightImagery(
        fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5, resample=RESAMPLE
    )
    subjects = list(dataset.subject_list)
    print(f"Lee2019_MI: {len(subjects)} subjects -> {OUT}", flush=True)

    for subject in subjects:
        target = OUT / f"subject_{subject:02d}.npz"
        if target.exists():
            print(f"  subject {subject:2d} cached", flush=True)
            continue
        started = time.perf_counter()
        before = snapshot()
        epochs, labels, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        if epochs.shape[1] != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"Subject {subject}: {epochs.shape[1]} channels, expected "
                f"{EXPECTED_CHANNELS}; the paradigm's channel selection changed."
            )
        covariances = Covariances(estimator="oas").fit_transform(epochs)
        #  Written to a temporary name first so an interrupted save cannot
        #  leave a truncated file that would later be treated as complete.
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(
            partial, covariances=covariances, labels=np.asarray(labels).astype(str)
        )
        partial.replace(target)
        freed = delete_new(before)
        free = shutil.disk_usage(Path.home()).free / 1e9
        if free < MIN_FREE_GB:
            raise SystemExit(
                f"Only {free:.1f} GB free after subject {subject}; raw files are "
                "not being removed. Stop and report this output."
            )
        print(
            f"  subject {subject:2d}: {len(labels)} trials, "
            f"{epochs.shape[2]} samples, freed {freed:.2f} GB, free {free:.1f} GB "
            f"({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )

    rows = []
    for subject in subjects:
        with np.load(OUT / f"subject_{subject:02d}.npz") as archive:
            labels = archive["labels"]
            names, counts = np.unique(labels, return_counts=True)
            rows.append(
                {
                    "subject": subject,
                    "trials": len(labels),
                    **dict(zip(names, counts, strict=True)),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(f"\nDone: {len(summary)} subjects, {summary.trials.sum()} trials")
    print(summary.describe().loc[["min", "mean", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
