"""Record exactly how the data were loaded, filtered and epoched.

Why
---
A reviewer must be able to tell, from the paper alone, which channels
entered the covariance estimate and what happened to artifact-marked
trials. Both datasets ship EOG channels (three in BCI IV-2a, three in 2b)
and trial-level artifact annotations, and a covariance pipeline that
silently includes EOG would be measuring eye movement as well as cortical
activity.

This script answers that question from the code and the loaded data
rather than from memory. It prints the loader's configuration, the
channel names and types actually present after loading, the filter and
epoch settings, the per-subject and per-class trial counts, and the
library versions. Nothing is recomputed and no model is trained; it takes
seconds.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/provenance.py

Paste the whole output into the manuscript discussion; the Methods
section quotes it.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seeds_stats import CACHE

from geoq.datasets import load_dataset

DATASETS = ("bci_iv_2a_lr", "bci_iv_2b")


def show_versions() -> None:
    """Print the versions of every library that touches the raw data."""
    print("=" * 70)
    print("LIBRARY VERSIONS")
    print("=" * 70)
    for name in ("mne", "moabb", "numpy", "scipy", "sklearn", "pennylane"):
        try:
            module = __import__(name)
            print(f"  {name:12s} {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            print(f"  {name:12s} not installed")


def show_loader_source() -> None:
    """Print the adapter's specification table and preprocessing code.

    The specification states which channels a dataset is expected to have;
    the preprocessing function states what is done to them. Printing the
    source removes any doubt about which branch ran.
    """
    print("\n" + "=" * 70)
    print("LOADER SOURCE")
    print("=" * 70)
    try:
        from geoq.datasets import moabb_adapter as adapter
    except ImportError as error:  # pragma: no cover - depends on layout
        print(f"  could not import moabb_adapter: {error}")
        return

    specs = getattr(adapter, "MOABB_SPECS", None)
    if specs is not None:
        print("\n--- MOABB_SPECS ---")
        for key, spec in specs.items():
            print(f"  {key}: {spec}")

    for name, member in vars(adapter).items():
        if name.startswith("_") and not callable(member):
            continue
        if (
            not callable(member)
            or getattr(member, "__module__", "") != adapter.__name__
        ):
            continue
        try:
            source = inspect.getsource(member)
        except (OSError, TypeError):  # pragma: no cover
            continue
        #  Only the functions that can touch channels, filtering or epochs.
        if any(
            word in source
            for word in (
                "pick",
                "filter",
                "eog",
                "EOG",
                "tmin",
                "reject",
                "resample",
                "Epochs",
            )
        ):
            print(f"\n--- {name} ---")
            print(source)


def describe(dataset: str) -> None:
    """Print what the loaded dataset actually contains."""
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset}")
    print("=" * 70)
    data = load_dataset(dataset, cache_dir=CACHE)

    print(f"  repr           : {data}")
    for attribute in (
        "channels",
        "ch_names",
        "sfreq",
        "sampling_rate",
        "tmin",
        "tmax",
        "band",
        "filter",
        "info",
        "spec",
        "metadata",
        "description",
    ):
        if hasattr(data, attribute):
            print(f"  {attribute:15s}: {getattr(data, attribute)}")

    epochs = np.asarray(data.epochs)
    labels = np.asarray(data.labels)
    subjects = np.asarray(data.subjects)
    print(f"  epochs shape   : {epochs.shape} (trials, channels, samples)")
    print(f"  dtype          : {epochs.dtype}")
    print(
        f"  amplitude range: {np.abs(epochs).max():.3e} "
        f"(volts if ~1e-4, microvolts if ~1e2)"
    )
    print(f"  any NaN        : {bool(np.isnan(epochs).any())}")

    classes, counts = np.unique(labels, return_counts=True)
    print(f"  classes        : {dict(zip(classes, counts, strict=True))}")
    print("  trials per subject and class:")
    for subject in np.unique(subjects):
        mask = subjects == subject
        per_class = {str(c): int(((labels == c) & mask).sum()) for c in classes}
        print(f"    subject {subject}: {int(mask.sum())} trials  {per_class}")

    #  A channel-count check that names the likely explanation. 2a has 22
    #  EEG and 3 EOG channels; 2b has 3 EEG and 3 EOG.
    n_channels = epochs.shape[1]
    expected = {"bci_iv_2a_lr": 22, "bci_iv_2b": 3}[dataset]
    if n_channels == expected:
        print(
            f"  CHANNELS       : {n_channels}, matches the EEG-only count, "
            "so EOG channels are not in the covariance estimate."
        )
    elif n_channels == expected + 3:
        print(
            f"  CHANNELS       : {n_channels} = {expected} EEG + 3 EOG. "
            "EOG IS INCLUDED IN THE COVARIANCE ESTIMATE."
        )
    else:
        print(
            f"  CHANNELS       : {n_channels}, expected {expected} "
            f"EEG or {expected + 3} with EOG; check the adapter."
        )


def main() -> None:
    """Print everything needed to document the preprocessing."""
    show_versions()
    show_loader_source()
    for dataset in DATASETS:
        describe(dataset)
    print("\nDone. Paste this output in full.")


if __name__ == "__main__":
    main()
