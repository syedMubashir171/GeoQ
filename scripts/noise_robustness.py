"""Robustness of the trained circuit to hardware-like noise.

Why
---
All results in the paper come from exact state-vector simulation. The
finite-shot experiment covers measurement statistics only. This script
adds the two other error sources of current hardware, gate errors and
readout errors, and measures how much accuracy the trained circuit keeps.

What is done
------------
For every fold the nine-layer circuit is trained once (seed 0, fold-wise
reference, exactly as in the main experiments) and its parameters are
saved to Drive, so further noise levels can be evaluated later without
retraining. The same parameters are then evaluated with a density-matrix
simulator under a noise model applied after every gate:

    single-qubit gates   depolarising channel with probability p1
    CNOT                 depolarising channel with probability p2 on both
                         the control and the target qubit
    readout              bit-flip with probability r on the measured qubit

The levels span roughly an order of magnitude either side of the error
rates reported for current superconducting and trapped-ion devices
(single-qubit ~1e-4 to 1e-3, two-qubit ~1e-3 to 1e-2, readout ~1e-2).
Applying local depolarising channels after a CNOT approximates, but is not
identical to, a two-qubit depolarising channel; the paper says so.

Unital noise such as depolarising and bit-flip channels mostly shrinks
<Z_1> towards zero rather than flipping its sign, and the class is the
sign, so noise alone changes few predictions. Its practical cost is that a
shrunken expectation needs more measurement shots to resolve. Every noise
level is therefore also evaluated with 100 and 1000 shots, sampled from the
exact noisy expectation; this combination is what a hardware run measures.
The mean |<Z_1>| per level is reported to show the shrinkage directly.

The circuit is evaluated, not retrained, under noise. This measures how
far a model trained in simulation degrades when deployed on noisy
hardware, which is the relevant question for a classifier whose training
would in practice also happen in simulation.

A check on every fold confirms that the noisy circuit with all
probabilities set to zero reproduces the trained circuit's predictions
exactly.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/noise_robustness.py

Outputs (paper2 results folder)
-------------------------------
noise_params/{dataset}_s{subject}.npz   trained parameters per fold
noise_robustness.csv                    kappa per dataset, subject, level and shots
"""

from __future__ import annotations

import pickle
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

SEED = 0

#: (name, single-qubit p1, two-qubit p2, readout r)
LEVELS = (
    ("ideal", 0.0, 0.0, 0.0),
    ("low", 1e-4, 1e-3, 0.005),
    ("moderate", 5e-4, 5e-3, 0.01),
    ("high", 1e-3, 1e-2, 0.02),
    ("very_high", 5e-3, 5e-2, 0.05),
)

#: Trials per density-matrix batch; bounds memory at eight qubits.
CHUNK = 32

#: Shot budgets sampled from each exact noisy expectation; None is exact.
SHOTS = (None, 1000, 100)


def noisy_expectations(
    model, reduced: np.ndarray, p1: float, p2: float, readout: float
) -> np.ndarray:
    """<Z_1> of the trained circuit under the noise model, per trial."""
    import pennylane as qml

    n_qubits, n_layers, params = model.n_qubits, model.n_layers, model.params_
    device = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(device)
    def circuit(x):
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.RY(x[wire % x.shape[0]], wires=wire)
                if p1:
                    qml.DepolarizingChannel(p1, wires=wire)
            for wire in range(n_qubits):
                qml.Rot(*params[layer, wire], wires=wire)
                if p1:
                    qml.DepolarizingChannel(p1, wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
                if p2:
                    qml.DepolarizingChannel(p2, wires=wire)
                    qml.DepolarizingChannel(p2, wires=wire + 1)
        if readout:
            qml.BitFlip(readout, wires=0)
        return qml.expval(qml.PauliZ(0))

    values = [float(circuit(row)) for row in reduced]
    return np.asarray(values)


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> list[dict]:
    """Kappa at every noise level and shot budget, with an agreement check."""
    reduced = model.reducer_.transform(np.asarray(x_test, dtype=np.float64))
    trained = np.asarray(model.circuit_(model.params_, reduced))
    rng = np.random.default_rng(SEED)
    rows = []
    for name, p1, p2, readout in LEVELS:
        outputs = np.concatenate(
            [
                noisy_expectations(
                    model, reduced[start : start + CHUNK], p1, p2, readout
                )
                for start in range(0, len(reduced), CHUNK)
            ]
        )
        if name == "ideal":
            agreement = float(np.mean(np.sign(outputs) == np.sign(trained)))
            if agreement < 1.0:
                raise RuntimeError(
                    f"Noise-free density-matrix circuit disagrees with the "
                    f"trained circuit on {(1 - agreement) * 100:.1f}% of trials."
                )
        for shots in SHOTS:
            if shots is None:
                estimate = outputs
            else:
                #  Each shot returns +1 with probability (1 + <Z>) / 2.
                ups = rng.binomial(shots, np.clip((1 + outputs) / 2, 0, 1))
                estimate = 2 * ups / shots - 1
            predicted = np.where(estimate > 0, model.classes_[1], model.classes_[0])
            rows.append(
                {
                    "level": name,
                    "shots": "exact" if shots is None else shots,
                    "kappa": float(cohen_kappa_score(y_test, predicted)),
                    "mean_abs_z": float(np.mean(np.abs(outputs))),
                }
            )
    return rows


def run(dataset: str, records: list[dict]) -> list[dict]:
    """Train (or reload) the circuit per fold and evaluate every level."""
    output = RESULTS / "noise_robustness.csv"
    store = RESULTS / "noise_params"
    store.mkdir(parents=True, exist_ok=True)
    done = {(r["dataset"], r["subject"]) for r in records}

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    labels = LabelEncoder().fit_transform(data.labels)
    subjects = np.asarray(data.subjects)

    for subject in np.unique(subjects):
        if (dataset, subject) in done:
            print(f"    subject {subject} cached", flush=True)
            continue
        started = time.perf_counter()
        train, test = subjects != subject, subjects == subject
        reference = TangentSpace().fit(covariances[train])
        x_train = reference.transform(covariances[train])
        x_test = reference.transform(covariances[test])

        saved = store / f"{dataset}_s{subject}.pkl"
        if saved.exists():
            with saved.open("rb") as handle:
                model = pickle.load(handle)
        else:
            model = ReuploadingClassifier(
                n_qubits=WIDTHS[dataset], n_layers=LAYERS, seed=SEED
            )
            model.fit(x_train, labels[train])
            #  The QNode is rebuilt on load; only the fitted state is kept.
            circuit = model.circuit_
            model.circuit_ = None
            with saved.open("wb") as handle:
                pickle.dump(model, handle)
            model.circuit_ = circuit
        if getattr(model, "circuit_", None) is None:
            model.circuit_ = model._build()

        rows = evaluate(model, x_test, labels[test])
        for row in rows:
            records.append({"dataset": dataset, "subject": subject, **row})
        pd.DataFrame(records).to_csv(output, index=False)
        exact = [r for r in rows if r["shots"] == "exact"]
        print(
            f"    subject {subject}  "
            + "  ".join(f"{r['level']}: {r['kappa']:+.3f}" for r in exact)
            + f"  ({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )
    return records


def main() -> None:
    """Run both datasets and print the degradation table."""
    output = RESULTS / "noise_robustness.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    for dataset in WIDTHS:
        print(f"{dataset}  (seed {SEED}, {LAYERS} layers)", flush=True)
        records = run(dataset, records)

    table = pd.DataFrame(records)
    table["shots"] = table["shots"].astype(str)
    order = [name for name, *_ in LEVELS]
    pivot = table.pivot_table(
        index=["dataset", "shots"], columns="level", values="kappa"
    ).reindex(columns=order)
    print("\nMean kappa by noise level (rows: shots per prediction)")
    print(pivot.round(4).to_string())
    print("\nLoss relative to the noise-free circuit with exact readout")
    reference = pivot.xs("exact", level="shots")["ideal"]
    print(pivot.sub(reference, axis=0, level="dataset").round(4).to_string())
    print("\nMean |<Z_1>| by noise level (shrinkage of the readout)")
    print(
        table[table.shots == "exact"]
        .pivot_table(index="dataset", columns="level", values="mean_abs_z")
        .reindex(columns=order)
        .round(4)
        .to_string()
    )
    print(
        "\nLevels (p1, p2, readout): "
        + ", ".join(f"{n}=({a:g}, {b:g}, {c:g})" for n, a, b, c in LEVELS)
    )


if __name__ == "__main__":
    main()
