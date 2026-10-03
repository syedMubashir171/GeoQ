"""Finite-shot evaluation of the trained re-uploading circuit.

Why
---
Every result so far comes from exact state-vector simulation, so the
readout expectation carries no measurement noise. On hardware the
expectation is estimated from a finite number of shots, and the class is
the sign of that estimate, so a trial whose true expectation lies near
zero can flip. This script measures how much accuracy that costs, without
retraining: the circuit is trained once per fold exactly as in
``inductive_reference.py``, and the same trained parameters are then
evaluated analytically and at several shot budgets.

The result is the shot budget at which the circuit keeps the accuracy
reported in the paper, which is the quantity a hardware experiment needs.

Cost
----
One seed per dataset, so roughly the cost of a single entry in the main
table (about three hours in total). The shot evaluations themselves are
cheap; training dominates.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/shot_noise.py

Output
------
shot_noise.csv   one row per (dataset, subject, shot budget)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hybrid import ReuploadingClassifier
from scripts.seeds_stats import CACHE, LAYERS, RESULTS, WIDTHS

from geoq.datasets import load_dataset
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

#: Shot budgets per prediction. ``None`` is the exact simulation used
#: everywhere else in the paper and is the reference row.
SHOTS = (None, 100, 1000, 10000)

#: One seed is enough to measure a degradation curve, and the training
#: cost is the same as one entry of the main table.
SEED = 0


def shot_predictions(model, features, shots: int, seed: int) -> np.ndarray:
    """Predict with the trained parameters and a finite shot budget.

    Args:
        model: A fitted ``ReuploadingClassifier``.
        features: Unreduced features for the held-out trials.
        shots: Number of shots per prediction, or ``None`` for exact.
        seed: Seed for the measurement sampler.

    Returns:
        Predicted labels.
    """
    import pennylane as qml

    reduced = model.reducer_.transform(np.asarray(features, dtype=np.float64))
    n_qubits, n_layers = model.n_qubits, model.n_layers
    try:
        device = qml.device("default.qubit", wires=n_qubits, shots=shots, seed=seed)
    except TypeError:
        #  Older PennyLane releases do not accept a device seed.
        device = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(device)
    def circuit(params, x):
        #  The same gates as ReuploadingClassifier._build; the agreement
        #  check below fails loudly if the two ever drift apart.
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.RY(x[..., wire % x.shape[-1]], wires=wire)
            for wire in range(n_qubits):
                qml.Rot(*params[layer, wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return qml.expval(qml.PauliZ(0))

    outputs = np.asarray(circuit(model.params_, reduced))
    return np.where(outputs > 0, model.classes_[1], model.classes_[0])


def check_circuit_matches(model, features) -> None:
    """Confirm the re-implemented circuit equals the trained one exactly."""
    reduced = model.reducer_.transform(np.asarray(features, dtype=np.float64))
    trained = np.asarray(model.circuit_(model.params_, reduced))
    mine = np.where(
        shot_predictions(model, features, None, SEED) == model.classes_[1],
        1.0,
        -1.0,
    )
    agree = float(np.mean(np.sign(trained) == mine))
    if agree < 1.0:
        raise RuntimeError(
            f"Shot circuit disagrees with the trained circuit on "
            f"{(1 - agree) * 100:.1f}% of trials."
        )


def run(dataset: str, records: list[dict]) -> list[dict]:
    """Train once per fold, then evaluate at every shot budget."""
    output = RESULTS / "shot_noise.csv"
    done = {(r["dataset"], r["subject"]) for r in records}

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    labels = LabelEncoder().fit_transform(data.labels)
    subjects = np.asarray(data.subjects)
    n_qubits = WIDTHS[dataset]

    for subject in np.unique(subjects):
        if (dataset, subject) in done:
            print(f"    subject {subject} cached", flush=True)
            continue
        started = time.perf_counter()
        train, test = subjects != subject, subjects == subject
        reference = TangentSpace().fit(covariances[train])
        x_train = reference.transform(covariances[train])
        x_test = reference.transform(covariances[test])

        model = ReuploadingClassifier(n_qubits=n_qubits, n_layers=LAYERS, seed=SEED)
        model.fit(x_train, labels[train])
        check_circuit_matches(model, x_test)

        scores = {}
        for shots in SHOTS:
            predicted = shot_predictions(model, x_test, shots, SEED)
            kappa = float(cohen_kappa_score(labels[test], predicted))
            scores[shots] = kappa
            records.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "shots": "exact" if shots is None else shots,
                    "kappa": kappa,
                }
            )
        pd.DataFrame(records).to_csv(output, index=False)
        summary = "  ".join(
            f"{'exact' if s is None else s}: {scores[s]:+.3f}" for s in SHOTS
        )
        print(
            f"    subject {subject}  {summary}  "
            f"({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )
    return records


def main() -> None:
    """Run both datasets and print the degradation curve."""
    output = RESULTS / "shot_noise.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    for dataset in WIDTHS:
        print(f"{dataset}  (seed {SEED}, {LAYERS} layers)", flush=True)
        records = run(dataset, records)

    table = pd.DataFrame(records)
    order = ["exact", 100, 1000, 10000]
    pivot = (
        table.assign(shots=table.shots.astype(str))
        .pivot_table(index="dataset", columns="shots", values="kappa")
        .reindex(columns=[str(s) for s in order])
    )
    print("\nMean kappa by shot budget")
    print(pivot.round(4).to_string())
    print("\nLoss relative to exact simulation")
    print(pivot.sub(pivot[str(order[0])], axis=0).round(4).to_string())


if __name__ == "__main__":
    main()
