"""Replication on Lee2019_MI: 54 subjects, the same models and tests.

Why
---
With nine subject records per dataset, the paper's primary comparison
can only bound a circuit advantage, and the evidence that entanglement adds
interaction beyond routing rests on a pooled secondary test. Lee2019_MI
has 54 subjects, enough for the same paired tests to resolve differences
of a few hundredths in kappa within one dataset.

Protocol
--------
Covariances come from lee2019_prepare.py (identical preprocessing to the
main datasets, 62 EEG channels, 250 Hz). Subjects are split into six
groups; each group is held out once while the model trains on the other
45 subjects. Every subject is therefore tested exactly once by a model that
never saw it, which gives one kappa per subject and paired tests over 54
records. Six folds rather than 54 keep the circuit's cost near 4 h per
seed. The Riemannian reference and the PCA reduction are fitted inside
each fold. The circuit is fixed at nine layers and eight qubits, as in the
entanglement experiments.

Stages (run in order; resumable per model, seed and fold)
---------------------------------------------------------
core      ts_lr, mlp_24, mlp_8, hybrid                 3 seeds
controls  product_readout (3 seeds), hybrid_no_entangle (1 seed)
summary   tables and paired tests

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/lee2019_experiment.py core
    !python scripts/lee2019_experiment.py controls
    !python scripts/lee2019_experiment.py summary

Outputs (paper2 results folder)
-------------------------------
lee2019_runs.csv    one row per model, seed and held-out subject
lee2019_tests.csv   paired tests over the 54 subjects
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

STAGE = sys.argv[1] if len(sys.argv) > 1 else "core"
sys.argv = sys.argv[:1]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.readout_control import ProductReadoutReuploading, paired  # noqa: E402
from scripts.seeds_stats import CACHE, LAYERS, RESULTS, holm, models  # noqa: E402
from scripts.standard_baselines import logistic, tune_logistic  # noqa: E402

from geoq.features.tangent_space import TangentSpace  # noqa: E402

DATA = Path(CACHE) / "lee2019"
N_QUBITS = 8
N_FOLDS = 6
SEEDS = (0, 1, 2)
PLAN = {
    "core": [("ts_lr", (0,)), ("mlp_24", SEEDS), ("mlp_8", SEEDS), ("hybrid", SEEDS)],
    "controls": [("product_readout", SEEDS), ("hybrid_no_entangle", (0,))],
}
PLANNED = (
    ("hybrid", "hybrid_no_entangle"),
    ("hybrid", "product_readout"),
    ("hybrid", "mlp_24"),
    ("hybrid", "mlp_8"),
)
REFERENCE = (("hybrid", "ts_lr"), ("mlp_24", "ts_lr"))


def load():
    """Covariances, encoded labels and subject ids for the whole cohort."""
    files = sorted(DATA.glob("subject_*.npz"))
    if not files:
        raise SystemExit(f"No covariance files in {DATA}; run lee2019_prepare.py.")
    covariances, labels, subjects = [], [], []
    for path in files:
        with np.load(path) as archive:
            covariances.append(archive["covariances"])
            labels.append(archive["labels"])
            subjects.append(np.full(len(archive["labels"]), int(path.stem[-2:])))
    return (
        np.concatenate(covariances),
        LabelEncoder().fit_transform(np.concatenate(labels)),
        np.concatenate(subjects),
    )


def fold_of(subjects: np.ndarray) -> dict:
    """Assign subjects to folds round-robin in sorted order."""
    return {s: i % N_FOLDS for i, s in enumerate(np.unique(subjects))}


def build(name: str, n_features: int, seed: int):
    """A fresh, unfitted model by name."""
    if name == "product_readout":
        return ProductReadoutReuploading(n_qubits=N_QUBITS, n_layers=LAYERS, seed=seed)
    return models(N_QUBITS, n_features, seed)[name]


def run(stage: str) -> None:
    """Fit every model of the stage on every fold, resuming at fold level."""
    output = RESULTS / "lee2019_runs.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {(r["model"], r["seed"], r["fold"]) for r in records}

    covariances, labels, subjects = load()
    folds = fold_of(subjects)
    fold_index = np.array([folds[s] for s in subjects])
    n_features = covariances.shape[1] * (covariances.shape[1] + 1) // 2
    print(
        f"Lee2019_MI: {len(labels)} trials, {len(folds)} subjects, "
        f"{covariances.shape[1]} channels, {N_FOLDS} folds",
        flush=True,
    )

    projected = {}
    for name, seeds in PLAN[stage]:
        for seed in seeds:
            for fold in range(N_FOLDS):
                if not (fold_index == fold).any():
                    continue
                if (name, seed, fold) in done:
                    print(f"  {name:20s} seed {seed} fold {fold} cached", flush=True)
                    continue
                if fold not in projected:
                    train = fold_index != fold
                    reference = TangentSpace().fit(covariances[train])
                    projected[fold] = (
                        train,
                        reference.transform(covariances[train]),
                        reference.transform(covariances[~train]),
                    )
                train, x_train, x_test = projected[fold]
                started = time.perf_counter()
                if name == "ts_lr":
                    value = tune_logistic(x_train, labels[train], subjects[train])
                    model = logistic(value).fit(x_train, labels[train])
                else:
                    model = build(name, n_features, seed)
                    model.fit(x_train, labels[train])
                predicted = model.predict(x_test)
                test_subjects = subjects[~train]
                for subject in np.unique(test_subjects):
                    mask = test_subjects == subject
                    records.append(
                        {
                            "model": name,
                            "seed": seed,
                            "fold": fold,
                            "subject": int(subject),
                            "kappa": float(
                                cohen_kappa_score(labels[~train][mask], predicted[mask])
                            ),
                        }
                    )
                pd.DataFrame(records).to_csv(output, index=False)
                fold_kappa = np.mean(
                    [
                        r["kappa"]
                        for r in records
                        if r["model"] == name
                        and r["seed"] == seed
                        and r["fold"] == fold
                    ]
                )
                print(
                    f"  {name:20s} seed {seed} fold {fold}: mean kappa "
                    f"{fold_kappa:+.3f} ({(time.perf_counter() - started) / 60:.1f}"
                    " min)",
                    flush=True,
                )


def summary() -> None:
    """Means, seed spread and paired tests over the 54 subjects."""
    output = RESULTS / "lee2019_runs.csv"
    if not output.exists():
        raise SystemExit("No runs yet.")
    frame = pd.read_csv(output)
    means = frame.groupby(["model", "subject"]).kappa.mean().unstack("model")
    seed_sd = frame.groupby(["model", "seed"]).kappa.mean().groupby("model").std()
    table = pd.DataFrame(
        {
            "subjects": means.notna().sum(),
            "kappa": means.mean(),
            "seed_sd": seed_sd,
            "seeds": frame.groupby("model").seed.nunique(),
        }
    )
    print("Lee2019_MI, mean kappa over held-out subjects")
    print(table.round(4).to_string())

    rows = []
    for family, pairs in (("planned", PLANNED), ("reference", REFERENCE)):
        available = [
            (a, b) for a, b in pairs if a in means.columns and b in means.columns
        ]
        batch = []
        for a, b in available:
            both = means[[a, b]].dropna()
            batch.append(
                {
                    "family": family,
                    "comparison": f"{a} - {b}",
                    "n": len(both),
                    **paired(both[a].to_numpy(), both[b].to_numpy()),
                }
            )
        if batch:
            for test, adjusted in zip(
                batch, holm([t["p_perm"] for t in batch]), strict=True
            ):
                test["p_holm"] = adjusted
            rows.extend(batch)
    if rows:
        tests = pd.DataFrame(rows)
        tests.to_csv(RESULTS / "lee2019_tests.csv", index=False)
        print(
            "\nPaired tests over held-out subjects (Holm within family; "
            "p from 20,000 random sign flips when n > 20)"
        )
        print(tests.round(4).to_string(index=False))


def main() -> None:
    """Run one stage, then summarise."""
    if STAGE != "summary":
        if STAGE not in PLAN:
            raise SystemExit("Use: core, controls or summary.")
        run(STAGE)
    summary()


if __name__ == "__main__":
    main()
