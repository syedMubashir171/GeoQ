"""Standard Riemannian EEG baselines under the same fold-wise protocol.

Why
---
The paper's classical comparators are matched to the circuit: they see the
same eight (or six) principal components. That isolates the effect of the
model, but it leaves open how the whole pipeline compares with the decoders
the BCI literature actually uses. This script adds the three standard
Riemannian decoders on the full feature set, with every fitted step inside
the fold:

ts_lr    tangent space + standardisation + logistic regression, with the
         inverse regularisation C chosen by grouped inner cross-validation
ts_slda  tangent space + linear discriminant analysis with Ledoit-Wolf
         shrinkage (no free hyperparameter)
mdm      minimum distance to the Riemannian class means on the covariance
         matrices themselves (no free hyperparameter)

The reference point is fitted on the training subjects of each fold, as in
the main experiments. All three models are deterministic, so no seeds.

The circuit and the 24-unit MLP are read from inductive_{dataset}.csv (five
seeds, fold-wise reference) and compared with each baseline by the same
exact paired test used in the paper.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/standard_baselines.py

Outputs (paper2 results folder)
-------------------------------
standard_baselines.csv        per-fold kappa and selected C
standard_baselines_tests.csv  circuit and MLP against each baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.readout_control import paired
from scripts.seeds_stats import CACHE, RESULTS, WIDTHS, holm

from geoq.datasets import load_dataset
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
INNER_FOLDS = 3
BASELINES = ("ts_lr", "ts_slda", "mdm")


def logistic(value: float):
    """Standardised logistic regression with inverse penalty ``value``."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=value, max_iter=5000))


def tune_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Choose C by cross-validation grouped by training subject."""
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    best, best_score = C_GRID[0], -np.inf
    for value in C_GRID:
        scores = []
        for inner_train, inner_test in splitter.split(x, y, groups):
            model = logistic(value).fit(x[inner_train], y[inner_train])
            scores.append(
                cohen_kappa_score(y[inner_test], model.predict(x[inner_test]))
            )
        if np.mean(scores) > best_score:
            best, best_score = value, float(np.mean(scores))
    return best


def run(dataset: str) -> pd.DataFrame:
    """Evaluate the three baselines with leave-one-subject-out."""
    try:
        import pyriemann.classification as riemann_classifiers
    except ImportError:  # pragma: no cover - pyriemann ships with MOABB
        riemann_classifiers = None
        print("  pyriemann not installed; MDM skipped", flush=True)

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    labels = LabelEncoder().fit_transform(data.labels)
    subjects = np.asarray(data.subjects)

    rows = []
    for held_out in np.unique(subjects):
        train, test = subjects != held_out, subjects == held_out
        reference = TangentSpace().fit(covariances[train])
        x_train = reference.transform(covariances[train])
        x_test = reference.transform(covariances[test])
        y_train, y_test = labels[train], labels[test]

        value = tune_logistic(x_train, y_train, subjects[train])
        model = logistic(value).fit(x_train, y_train)
        rows.append(
            {
                "dataset": dataset,
                "model": "ts_lr",
                "subject": held_out,
                "C": value,
                "kappa": cohen_kappa_score(y_test, model.predict(x_test)),
            }
        )

        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(x_train, y_train)
        rows.append(
            {
                "dataset": dataset,
                "model": "ts_slda",
                "subject": held_out,
                "C": np.nan,
                "kappa": cohen_kappa_score(y_test, model.predict(x_test)),
            }
        )

        if riemann_classifiers is not None:
            model = riemann_classifiers.MDM(metric="riemann").fit(
                covariances[train], y_train
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": "mdm",
                    "subject": held_out,
                    "C": np.nan,
                    "kappa": cohen_kappa_score(
                        y_test, model.predict(covariances[test])
                    ),
                }
            )
        done = [r for r in rows if r["subject"] == held_out]
        print(
            f"    subject {held_out}: "
            + "  ".join(f"{r['model']} {r['kappa']:+.3f}" for r in done),
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run both datasets and compare with the circuit and the MLP."""
    frames, tests = [], []
    for dataset in WIDTHS:
        print(dataset, flush=True)
        frame = run(dataset)
        frames.append(frame)
        base = frame.pivot_table(index="subject", columns="model", values="kappa")

        main_runs = RESULTS / f"inductive_{dataset}.csv"
        if not main_runs.exists():
            print(f"  {main_runs.name} not found; paired tests skipped", flush=True)
            continue
        ours = (
            pd.read_csv(main_runs)
            .groupby(["model", "subject"])
            .kappa.mean()
            .unstack("model")
        )
        for family in ("hybrid", "mlp_24"):
            if family not in ours.columns:
                continue
            batch = [
                {
                    "dataset": dataset,
                    "comparison": f"{family} - {name}",
                    **paired(
                        ours[family].reindex(base.index).to_numpy(),
                        base[name].to_numpy(),
                    ),
                }
                for name in BASELINES
                if name in base.columns
            ]
            for test, adjusted in zip(
                batch, holm([b["p_perm"] for b in batch]), strict=True
            ):
                test["p_holm"] = adjusted
            tests.extend(batch)

    table = pd.concat(frames, ignore_index=True)
    table.to_csv(RESULTS / "standard_baselines.csv", index=False)
    print("\nMean LOSO kappa (fold-wise reference, all features)")
    print(
        table.groupby(["dataset", "model"])
        .kappa.agg(["mean", "std"])
        .round(4)
        .to_string()
    )
    print("\nC chosen for ts_lr (counts over folds)")
    print(
        table[table.model == "ts_lr"]
        .groupby(["dataset", "C"])
        .size()
        .rename("folds")
        .to_string()
    )
    if tests:
        result = pd.DataFrame(tests)
        result.to_csv(RESULTS / "standard_baselines_tests.csv", index=False)
        print("\nPaired tests (Holm over the three baselines, per dataset and model)")
        print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
