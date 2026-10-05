"""Fidelity kernels with per-fold bandwidth and tuned SVM hyperparameters.

Two problems this fixes
-----------------------
First, the earlier kernel experiment selected the encoding bandwidth once,
from the training features of the first fold, and reused it for every
fold. The criterion is label-free, so nothing leaked from the labels, but
the value was still chosen with the whole subject set in view. Here the
bandwidth is selected inside each outer training fold, from a probe of
that fold's training trials only.

Second, both support vector machines ran with scikit-learn's defaults
($C=1$, and ``gamma='scale'`` for the RBF kernel). An untuned classical
comparator is not a comparator. Here $C$ is tuned for the quantum kernel,
and $C$ together with the RBF bandwidth for the classical one, by grouped
inner cross-validation over the training subjects of each fold.

What this does not change
-------------------------
The kernel matrices were already computed inside each fold, because the
reduction (standardise, PCA, MaxAbs) is fitted per fold. Moving the
bandwidth selection inside the fold therefore costs only the probe
evaluations, which are 30 by 30, so this rerun costs about what the
original did.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/kernel_nested.py

Outputs (paper2 results folder)
-------------------------------
kernel_nested.csv            per-fold kappa, selected bandwidth and C
kernel_nested_summary.csv    mean kappa per model and width
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seeds_stats import CACHE, RESULTS

from geoq.datasets import load_dataset
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.models.quantum.kernel import fidelity_kernel

DATASET = "bci_iv_2a_lr"
MAX_TRIALS = 648
QUBITS = (4, 8, 12)
FEATURE_MAPS = ("angle", "zz")
BANDWIDTHS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
PROBE_SIZE = 30
C_GRID = (0.1, 1.0, 10.0, 100.0)
RBF_GAMMAS = ("scale", 0.01, 0.1, 1.0)
INNER_FOLDS = 3


def subsample(epochs, labels, subjects, max_trials: int, seed: int = 0):
    """Stratified subsample preserving the subject and class balance."""
    rng = np.random.default_rng(seed)
    keep = []
    per_cell = max_trials // (np.unique(subjects).size * np.unique(labels).size)
    for subject, label in itertools.product(np.unique(subjects), np.unique(labels)):
        index = np.flatnonzero((subjects == subject) & (labels == label))
        keep.append(rng.choice(index, min(per_cell, index.size), replace=False))
    keep = np.sort(np.concatenate(keep))
    return epochs[keep], labels[keep], subjects[keep]


def select_bandwidth(features: np.ndarray, feature_map: str, seed: int = 0) -> float:
    """Choose the bandwidth without labels, from training trials only.

    A kernel that has concentrated towards the identity, or saturated
    near one, has little spread among its off-diagonal entries. Maximising
    that spread rejects both failure modes and uses no labels.

    Args:
        features: Reduced training features of this fold.
        feature_map: Name of the feature map.
        seed: Seed for drawing the probe.

    Returns:
        The selected bandwidth.
    """
    rng = np.random.default_rng(seed)
    size = min(PROBE_SIZE, features.shape[0])
    probe = features[rng.choice(features.shape[0], size, replace=False)]
    mask = ~np.eye(size, dtype=bool)
    best, best_spread = BANDWIDTHS[0], -np.inf
    for bandwidth in BANDWIDTHS:
        gram = fidelity_kernel(probe, feature_map=feature_map, scale=bandwidth)
        spread = float(gram[mask].std())
        if spread > best_spread:
            best, best_spread = bandwidth, spread
    return best


def tune_precomputed(gram: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Choose $C$ by grouped inner cross-validation on a training kernel."""
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    best, best_score = C_GRID[0], -np.inf
    for value in C_GRID:
        scores = []
        for inner_train, inner_test in splitter.split(gram, labels, groups):
            model = SVC(kernel="precomputed", C=value)
            model.fit(gram[np.ix_(inner_train, inner_train)], labels[inner_train])
            predicted = model.predict(gram[np.ix_(inner_test, inner_train)])
            scores.append(cohen_kappa_score(labels[inner_test], predicted))
        if np.mean(scores) > best_score:
            best, best_score = value, float(np.mean(scores))
    return best


def tune_rbf(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[float, object]:
    """Choose $C$ and the RBF bandwidth by grouped inner cross-validation."""
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    best, best_score = (C_GRID[0], RBF_GAMMAS[0]), -np.inf
    for value, gamma in itertools.product(C_GRID, RBF_GAMMAS):
        scores = []
        for inner_train, inner_test in splitter.split(features, labels, groups):
            model = SVC(C=value, gamma=gamma)
            model.fit(features[inner_train], labels[inner_train])
            predicted = model.predict(features[inner_test])
            scores.append(cohen_kappa_score(labels[inner_test], predicted))
        if np.mean(scores) > best_score:
            best, best_score = (value, gamma), float(np.mean(scores))
    return best


def main() -> None:
    """Run every model on every fold, resuming after an interruption."""
    output = RESULTS / "kernel_nested.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {(r["model"], r["n_qubits"], r["subject"]) for r in records}

    data = load_dataset(DATASET, cache_dir=CACHE)
    epochs, labels, subjects = subsample(
        np.asarray(data.epochs),
        np.asarray(data.labels),
        np.asarray(data.subjects),
        MAX_TRIALS,
    )
    labels = LabelEncoder().fit_transform(labels)
    covariances = Covariances(estimator="oas").fit_transform(epochs)
    print(
        f"{DATASET}: {len(labels)} trials, {np.unique(subjects).size} subjects",
        flush=True,
    )

    def store(model, n_qubits, subject, kappa, **extra):
        records.append(
            {
                "model": model,
                "n_qubits": n_qubits,
                "subject": subject,
                "kappa": float(kappa),
                **extra,
            }
        )
        pd.DataFrame(records).to_csv(output, index=False)

    for held_out in np.unique(subjects):
        train, test = subjects != held_out, subjects == held_out
        reference = TangentSpace().fit(covariances[train])
        tangent_train = reference.transform(covariances[train])
        tangent_test = reference.transform(covariances[test])
        y_train, y_test = labels[train], labels[test]
        groups = subjects[train]

        #  Reference outside the matched protocol: all 253 features.
        if ("ts_lda", 0, held_out) not in done:
            model = LinearDiscriminantAnalysis().fit(tangent_train, y_train)
            store(
                "ts_lda",
                0,
                held_out,
                cohen_kappa_score(y_test, model.predict(tangent_test)),
            )

        for n_qubits in QUBITS:
            reducer = make_pipeline(
                StandardScaler(), PCA(n_qubits, random_state=0), MaxAbsScaler()
            )
            x_train = reducer.fit_transform(tangent_train)
            x_test = reducer.transform(tangent_test)

            if ("rbf_svm", n_qubits, held_out) not in done:
                started = time.perf_counter()
                (value, gamma) = tune_rbf(x_train, y_train, groups)
                model = SVC(C=value, gamma=gamma).fit(x_train, y_train)
                kappa = cohen_kappa_score(y_test, model.predict(x_test))
                store(
                    "rbf_svm",
                    n_qubits,
                    held_out,
                    kappa,
                    bandwidth=str(gamma),
                    C=value,
                    minutes=(time.perf_counter() - started) / 60,
                )
                print(
                    f"  subject {held_out} {n_qubits:2d}q rbf_svm    "
                    f"kappa {kappa:+.3f}  C={value:g} gamma={gamma}",
                    flush=True,
                )

            for feature_map in FEATURE_MAPS:
                if (feature_map, n_qubits, held_out) in done:
                    continue
                started = time.perf_counter()
                #  Bandwidth from this fold's training trials only.
                bandwidth = select_bandwidth(x_train, feature_map)
                gram_train = fidelity_kernel(
                    x_train, feature_map=feature_map, scale=bandwidth
                )
                gram_test = fidelity_kernel(
                    x_test, x_train, feature_map=feature_map, scale=bandwidth
                )
                value = tune_precomputed(gram_train, y_train, groups)
                model = SVC(kernel="precomputed", C=value)
                model.fit(gram_train, y_train)
                kappa = cohen_kappa_score(y_test, model.predict(gram_test))
                store(
                    feature_map,
                    n_qubits,
                    held_out,
                    kappa,
                    bandwidth=bandwidth,
                    C=value,
                    minutes=(time.perf_counter() - started) / 60,
                )
                print(
                    f"  subject {held_out} {n_qubits:2d}q "
                    f"{feature_map:8s} kappa {kappa:+.3f}  "
                    f"bandwidth={bandwidth:g} C={value:g} "
                    f"({(time.perf_counter() - started) / 60:.1f} min)",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    summary = (
        frame.groupby(["model", "n_qubits"])
        .agg(
            kappa=("kappa", "mean"),
            kappa_sd=("kappa", "std"),
            minutes=("minutes", "sum"),
        )
        .round(4)
    )
    summary.to_csv(RESULTS / "kernel_nested_summary.csv")
    print("\nMean LOSO kappa")
    print(summary.to_string())
    print("\nSelected bandwidth per fold (counts)")
    print(
        frame[frame.model.isin(FEATURE_MAPS)]
        .groupby(["model", "n_qubits", "bandwidth"])
        .size()
        .rename("folds")
        .to_string()
    )
    print("\nSelected C per fold (counts)")
    print(
        frame.dropna(subset=["C"])
        .groupby(["model", "n_qubits", "C"])
        .size()
        .rename("folds")
        .to_string()
    )


if __name__ == "__main__":
    main()
