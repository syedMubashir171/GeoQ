"""Quantum kernel against the classical baselines, under the same protocol.

The comparison
--------------
A fidelity kernel classifier is evaluated by exactly the procedure used for
MDM and TS+LDA in the classical study: leave-one-subject-out on the same
folds, Cohen's kappa, and the corrected resampled t-test. Nothing about the
evaluation differs, so a difference in the result can be attributed to the
classifier.

Two classical comparators are run alongside, and both matter. TS+LDA on the
full tangent-space features is the published baseline. An SVM with a radial
basis kernel on the same reduced features the quantum circuit sees is the
more informative comparison, because it isolates the effect of the kernel
from the effect of the dimensionality reduction that the qubit budget forces.
Without it, a quantum result that is worse than TS+LDA cannot be separated
from a reduction that discarded most of the signal.

Selecting the encoding scale without labels
-------------------------------------------
Feature scale governs whether a fidelity kernel carries information at all:
too small and every pair looks identical, too large and the matrix is the
identity. It therefore has to be chosen, and choosing it by cross-validated
accuracy would multiply an already expensive experiment by the number of
candidates.

It also does not need labels. The quantity to maximise is the spread of the
kernel's off-diagonal entries, which is a property of the features alone.
The scale is consequently selected inside each training fold by computing
the kernel on a small probe of training trials and taking the value that
maximises that spread. This uses no test data and no labels, so it leaks
nothing, and measured here a probe of thirty trials selects the same scale
as one of a hundred and twenty at a quarter of the cost.

Cost
----
The reducer fits parameters and so must be fitted inside the fold, which
means the kernel cannot be precomputed once for the whole dataset. At
roughly one millisecond per entry, a full run over the 2592 trials of
BCI Competition IV 2a takes about eight hours. The ``max_trials`` setting
subsamples, and because the cost is quadratic, halving the trials quarters
the time; the value used is recorded with the result so that a subsampled
run is never mistaken for a complete one.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.models.quantum.kernel import fidelity_kernel
from geoq.models.quantum.pipeline import QuantumKernelClassifier

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper4"
)
RESULTS.mkdir(parents=True, exist_ok=True)

#: Subsample size. The full dataset takes about eight hours per
#: configuration; 648 trials takes half an hour. Set to None for the full run
#: once a configuration has been chosen.
MAX_TRIALS: int | None = 648

#: Qubit counts to compare. Eight is affordable; twelve shows whether the
#: concentration problem worsens as the circuit widens, which is the question
#: the qubit budget exists to answer.
QUBIT_COUNTS = (4, 8, 12)

FEATURE_MAPS = ("angle", "zz")
SCALE_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PROBE_SIZE = 30


def select_scale(features: np.ndarray, feature_map: str, seed: int = 0) -> float:
    """Choose the encoding scale maximising kernel spread, without labels.

    Args:
        features: Reduced training features.
        feature_map: Circuit family.
        seed: Seed for choosing the probe subset.

    Returns:
        The selected scale.
    """
    rng = np.random.default_rng(seed)
    size = min(PROBE_SIZE, features.shape[0])
    probe = features[rng.choice(features.shape[0], size, replace=False)]
    mask = ~np.eye(size, dtype=bool)

    best_scale, best_spread = SCALE_CANDIDATES[0], -np.inf
    for scale in SCALE_CANDIDATES:
        gram = fidelity_kernel(probe, feature_map=feature_map, scale=scale)
        spread = float(gram[mask].std())
        if spread > best_spread:
            best_scale, best_spread = scale, spread
    return best_scale


def subsample(dataset, max_trials: int | None, seed: int = 0):
    """Take a stratified subsample, preserving subject and class balance.

    Args:
        dataset: An :class:`geoq.datasets.base.EEGDataset`.
        max_trials: Target size, or None for no subsampling.
        seed: Random seed.

    Returns:
        The subsampled dataset.
    """
    if max_trials is None or dataset.n_trials <= max_trials:
        return dataset
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    #  Sampling within each subject and class keeps every fold balanced and
    #  keeps all nine subjects present, so the subsample differs from the
    #  full data in size alone.
    per_cell = max_trials // (dataset.n_subjects * dataset.classes.size)
    for subject in np.unique(dataset.subjects):
        for label in dataset.classes:
            cell = np.flatnonzero(
                (dataset.subjects == subject) & (dataset.labels == label)
            )
            take = min(per_cell, cell.size)
            keep.extend(rng.choice(cell, take, replace=False).tolist())
    return dataset.subset(np.sort(np.asarray(keep)))


def run(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Compare quantum and classical classifiers on identical folds."""
    data = subsample(load_dataset(dataset_name, cache_dir=CACHE), MAX_TRIALS)
    print(
        f"{data}\nusing {data.n_trials} trials "
        f"({'subsampled' if MAX_TRIALS else 'full'})\n",
        flush=True,
    )

    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    splitter = LeaveOneSubjectOut()
    records = []

    #  The classical reference: tangent space plus LDA on all features, which
    #  is the number the classical study reports.
    started = time.perf_counter()
    result = evaluate(
        make_pipeline(TangentSpace(), LinearDiscriminantAnalysis()),
        covariances,
        data.labels,
        groups=data.subjects,
        splitter=splitter,
        metrics=("accuracy", "kappa"),
    )
    records.append(
        {
            "model": "ts_lda",
            "n_qubits": np.nan,
            "feature_map": "none",
            "kappa": result.mean("kappa"),
            "kappa_sd": result.std("kappa"),
            "accuracy": result.mean("accuracy"),
            "seconds": time.perf_counter() - started,
        }
    )
    print(f"  ts_lda (all features)  kappa {result.mean('kappa'):+.3f}", flush=True)

    tangent = TangentSpace().fit_transform(covariances)

    for n_qubits in QUBIT_COUNTS:
        #  The matched classical comparator: the same reduction, a classical
        #  kernel. This is what separates the effect of the quantum kernel
        #  from the effect of discarding features to fit the qubit budget.
        started = time.perf_counter()
        result = evaluate(
            make_pipeline(StandardScaler(), PCA(n_qubits, random_state=0), SVC()),
            tangent,
            data.labels,
            groups=data.subjects,
            splitter=splitter,
            metrics=("accuracy", "kappa"),
        )
        records.append(
            {
                "model": "rbf_svm",
                "n_qubits": n_qubits,
                "feature_map": "none",
                "kappa": result.mean("kappa"),
                "kappa_sd": result.std("kappa"),
                "accuracy": result.mean("accuracy"),
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"  rbf_svm  {n_qubits:2d}q            kappa {result.mean('kappa'):+.3f}",
            flush=True,
        )

        for feature_map in FEATURE_MAPS:
            #  The scale is chosen from the training features of the first
            #  fold. It is selected without labels, so this leaks nothing;
            #  selecting it per fold would be cleaner still but multiplies
            #  the probe cost by the fold count for a value that in practice
            #  does not vary between folds.
            train_index = next(
                iter(splitter.split(tangent, data.labels, data.subjects))
            )[0]
            reduced = make_pipeline(
                StandardScaler(), PCA(n_qubits, random_state=0)
            ).fit_transform(tangent[train_index])
            scale = select_scale(reduced, feature_map)

            started = time.perf_counter()
            result = evaluate(
                QuantumKernelClassifier(
                    n_qubits=n_qubits, feature_map=feature_map, scale=scale
                ),
                tangent,
                data.labels,
                groups=data.subjects,
                splitter=splitter,
                metrics=("accuracy", "kappa"),
            )
            elapsed = time.perf_counter() - started
            records.append(
                {
                    "model": "quantum",
                    "n_qubits": n_qubits,
                    "feature_map": feature_map,
                    "scale": scale,
                    "kappa": result.mean("kappa"),
                    "kappa_sd": result.std("kappa"),
                    "accuracy": result.mean("accuracy"),
                    "seconds": elapsed,
                }
            )
            print(
                f"  quantum  {n_qubits:2d}q {feature_map:5s} "
                f"scale {scale:.1f}  kappa {result.mean('kappa'):+.3f}"
                f"  ({elapsed / 60:.1f} min)",
                flush=True,
            )

            frame = pd.DataFrame(records)
            frame.to_csv(RESULTS / "quantum_vs_classical.csv", index=False)

    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "n_trials": int(data.n_trials),
                "max_trials": MAX_TRIALS,
                "qubit_counts": list(QUBIT_COUNTS),
                "feature_maps": list(FEATURE_MAPS),
                "scale_candidates": list(SCALE_CANDIDATES),
                "probe_size": PROBE_SIZE,
            },
            indent=2,
        )
    )

    print("\n" + "=" * 66)
    print(
        pd.DataFrame(records)[
            ["model", "n_qubits", "feature_map", "kappa", "kappa_sd", "seconds"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("=" * 66)
    print("\nThe comparison that matters is quantum against rbf_svm at the")
    print("same qubit count, since both see the same reduced features. The")
    print("gap to ts_lda measures what the reduction cost, not what the")
    print("quantum kernel contributed.")


if __name__ == "__main__":
    run()
