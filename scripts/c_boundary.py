"""Check whether the edge of the C grid limited the angle and RBF kernels.

Why
---
In ``kernel_nested.py`` the inner loop often chose C = 0.1, the smallest
value offered, for both the angle kernel and the RBF kernel. A selected
value on the boundary of a grid means the grid may have stopped the search
early. This script repeats the selection with the grid extended downward
to 0.001 and reports whether the outer kappa changes.

Why this is cheap
-----------------
The angle feature map produces a product state, so its fidelity kernel has
a closed form,

    k(x, x') = prod_j cos^2( gamma * (x_j - x'_j) / 2 ),

which the simulated kernel matched to 3.9e-16. The bandwidth gamma is read
from the per-fold choices already stored in ``kernel_nested.csv``, so the
selection procedure is identical and no circuit is simulated except for a
small agreement check. The RBF kernel is classical. The whole script runs
in minutes.

Two checks are built in
-----------------------
1. The closed-form angle kernel is compared with the simulated one on a
   probe of every fold; any disagreement above 1e-10 stops the run.
2. With the original grid, every fold must reproduce the kappa stored in
   ``kernel_nested.csv``. The largest discrepancy is printed; it should be
   zero.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/c_boundary.py

Output
------
c_boundary.csv   per-fold kappa and chosen C under both grids
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.kernel_nested import (
    C_GRID,
    DATASET,
    INNER_FOLDS,
    MAX_TRIALS,
    QUBITS,
    RBF_GAMMAS,
    subsample,
)
from scripts.seeds_stats import CACHE, RESULTS

from geoq.datasets import load_dataset
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.models.quantum.kernel import fidelity_kernel

EXTENDED_GRID = (0.001, 0.01, *C_GRID)
GRIDS = {"original": C_GRID, "extended": EXTENDED_GRID}
AGREEMENT_PROBE = 20


def angle_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Closed-form fidelity kernel of the product-state angle encoding."""
    difference = bandwidth * (x[:, None, :] - y[None, :, :]) / 2.0
    return np.prod(np.cos(difference) ** 2, axis=-1)


def check_agreement(x: np.ndarray, bandwidth: float) -> float:
    """Largest gap between the closed form and the simulated kernel."""
    probe = x[:AGREEMENT_PROBE]
    simulated = fidelity_kernel(probe, feature_map="angle", scale=bandwidth)
    gap = float(np.abs(simulated - angle_kernel(probe, probe, bandwidth)).max())
    if gap > 1e-10:
        raise RuntimeError(f"Closed form disagrees with simulation by {gap:.2e}.")
    return gap


def tune_precomputed(gram, labels, groups, grid) -> float:
    """Choose C by grouped inner cross-validation, as in kernel_nested."""
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    best, best_score = grid[0], -np.inf
    for value in grid:
        scores = []
        for inner_train, inner_test in splitter.split(gram, labels, groups):
            model = SVC(kernel="precomputed", C=value)
            model.fit(gram[np.ix_(inner_train, inner_train)], labels[inner_train])
            predicted = model.predict(gram[np.ix_(inner_test, inner_train)])
            scores.append(cohen_kappa_score(labels[inner_test], predicted))
        if np.mean(scores) > best_score:
            best, best_score = value, float(np.mean(scores))
    return best


def tune_rbf(features, labels, groups, grid) -> tuple[float, object]:
    """Choose C and the RBF bandwidth by grouped inner cross-validation."""
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    best, best_score = (grid[0], RBF_GAMMAS[0]), -np.inf
    for value, gamma in itertools.product(grid, RBF_GAMMAS):
        scores = []
        for inner_train, inner_test in splitter.split(features, labels, groups):
            model = SVC(C=value, gamma=gamma)
            model.fit(features[inner_train], labels[inner_train])
            scores.append(
                cohen_kappa_score(
                    labels[inner_test], model.predict(features[inner_test])
                )
            )
        if np.mean(scores) > best_score:
            best, best_score = (value, gamma), float(np.mean(scores))
    return best


def main() -> None:
    """Rerun angle and RBF selection under both grids and compare."""
    previous = pd.read_csv(RESULTS / "kernel_nested.csv")
    bandwidths = previous[previous.model == "angle"].set_index(["n_qubits", "subject"])[
        "bandwidth"
    ]
    stored = previous.set_index(["model", "n_qubits", "subject"])["kappa"]

    data = load_dataset(DATASET, cache_dir=CACHE)
    epochs, labels, subjects = subsample(
        np.asarray(data.epochs),
        np.asarray(data.labels),
        np.asarray(data.subjects),
        MAX_TRIALS,
    )
    labels = LabelEncoder().fit_transform(labels)
    covariances = Covariances(estimator="oas").fit_transform(epochs)

    records, worst_gap = [], 0.0
    for held_out in np.unique(subjects):
        train, test = subjects != held_out, subjects == held_out
        reference = TangentSpace().fit(covariances[train])
        tangent_train = reference.transform(covariances[train])
        tangent_test = reference.transform(covariances[test])
        y_train, y_test = labels[train], labels[test]
        groups = subjects[train]

        for n_qubits in QUBITS:
            reducer = make_pipeline(
                StandardScaler(), PCA(n_qubits, random_state=0), MaxAbsScaler()
            )
            x_train = reducer.fit_transform(tangent_train)
            x_test = reducer.transform(tangent_test)

            bandwidth = float(bandwidths.loc[(n_qubits, held_out)])
            worst_gap = max(worst_gap, check_agreement(x_train, bandwidth))
            gram_train = angle_kernel(x_train, x_train, bandwidth)
            gram_test = angle_kernel(x_test, x_train, bandwidth)

            for grid_name, grid in GRIDS.items():
                value = tune_precomputed(gram_train, y_train, groups, grid)
                model = SVC(kernel="precomputed", C=value).fit(gram_train, y_train)
                kappa = cohen_kappa_score(y_test, model.predict(gram_test))
                records.append(
                    {
                        "model": "angle",
                        "n_qubits": n_qubits,
                        "subject": held_out,
                        "grid": grid_name,
                        "C": value,
                        "kappa": float(kappa),
                    }
                )

                value_rbf, gamma = tune_rbf(x_train, y_train, groups, grid)
                model = SVC(C=value_rbf, gamma=gamma).fit(x_train, y_train)
                kappa = cohen_kappa_score(y_test, model.predict(x_test))
                records.append(
                    {
                        "model": "rbf_svm",
                        "n_qubits": n_qubits,
                        "subject": held_out,
                        "grid": grid_name,
                        "C": value_rbf,
                        "kappa": float(kappa),
                    }
                )
        print(f"  subject {held_out} done", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(RESULTS / "c_boundary.csv", index=False)

    original = frame[frame.grid == "original"].set_index(
        ["model", "n_qubits", "subject"]
    )
    reproduce = (original["kappa"] - stored.reindex(original.index)).abs().max()
    print(f"\nClosed form vs simulated angle kernel: max gap {worst_gap:.2e}")
    print(
        f"Original grid vs kernel_nested.csv:     max kappa gap {reproduce:.2e} "
        "(should be 0)"
    )

    table = frame.pivot_table(
        index=["model", "n_qubits"], columns="grid", values="kappa", aggfunc="mean"
    )
    table["change"] = table["extended"] - table["original"]
    print("\nMean LOSO kappa under the original and extended C grids")
    print(table.round(4).to_string())
    print("\nC chosen under the extended grid (counts over folds)")
    print(
        frame[frame.grid == "extended"]
        .groupby(["model", "n_qubits", "C"])
        .size()
        .rename("folds")
        .to_string()
    )


if __name__ == "__main__":
    main()
