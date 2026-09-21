"""Product-state control with an all-qubit readout.

Why this control is needed
--------------------------
The entanglement ablation in ``ablations.py`` removes the CNOT chain but
keeps the readout ``<Z_0>``. Without entangling gates, qubit 0 evolves
independently of the others, so that model sees only the first principal
component. Its collapse to chance therefore shows that the CNOTs are the
circuit's only route for combining features. It does not show that
entanglement contributes anything beyond that routing.

This control closes the gap. Every qubit keeps its own re-uploading block
with no entangling gates, and the readout is the mean of ``<Z_i>`` over all
qubits. The model sees every feature, and its state is a product state at
all times. It is an additive model,

    f(x) = (1/n) * sum_i f_i(x_i),

where each ``f_i`` is a trigonometric series in one feature. If the
entangled circuit still beats it, entanglement contributes more than
feature access: it provides the interactions between features.

Because the state never entangles, each qubit is simulated on its own. That
costs n single-qubit circuits rather than one 2^n-dimensional state, so this
run is much cheaper than the entangled circuit.

Outputs (in the paper2 results folder)
--------------------------------------
readout_{dataset}.csv        per-fold kappa for every seed
readout_tests.csv            paired tests against the seeded circuit,
                             Holm-corrected together with the three
                             comparisons already reported
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hybrid import ReuploadingClassifier
from scripts.seeds_stats import (
    CACHE,
    COMPARISONS,
    LAYERS,
    RESULTS,
    SEEDS,
    WIDTHS,
    exact_permutation_p,
    holm,
)

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

N_BOOTSTRAP = 10_000


class ProductReadoutReuploading(ReuploadingClassifier):
    """Unentangled re-uploading circuit read out on every qubit.

    The trainable parameters, the encoding, the optimiser and the training
    schedule are inherited unchanged. Only the circuit differs: no CNOTs,
    and the prediction is the mean of the single-qubit Z expectations.
    """

    def _build(self):
        """Return a callable with the parent's ``circuit(params, x)`` API."""
        import pennylane as qml
        from pennylane import numpy as pnp

        device = qml.device("default.qubit", wires=1)

        @qml.qnode(device, interface="autograd", diff_method="backprop")
        def single(wire_params, feature):
            for layer in range(self.n_layers):
                qml.RY(feature, wires=0)
                qml.Rot(*wire_params[layer], wires=0)
            return qml.expval(qml.PauliZ(0))

        def circuit(params, x):
            n_inputs = x.shape[-1]
            outputs = [
                single(params[:, wire, :], x[..., wire % n_inputs])
                for wire in range(self.n_qubits)
            ]
            return pnp.mean(pnp.stack(outputs), axis=0)

        return circuit


def run(dataset: str) -> pd.DataFrame:
    """Evaluate the control from every seed, resuming if interrupted."""
    output = RESULTS / f"readout_{dataset}.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {r["seed"] for r in records}

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)
    n_qubits = WIDTHS[dataset]

    for seed in SEEDS:
        if seed in done:
            print(f"    seed {seed} cached", flush=True)
            continue
        started = time.perf_counter()
        model = ProductReadoutReuploading(n_qubits=n_qubits, n_layers=LAYERS, seed=seed)
        result = evaluate(
            model,
            tangent,
            labels,
            groups=data.subjects,
            splitter=LeaveOneSubjectOut(),
            metrics=("kappa",),
        )
        for fold in result.folds:
            records.append(
                {
                    "dataset": dataset,
                    "model": "product_readout",
                    "seed": seed,
                    "subject": fold.test_groups[0],
                    "kappa": fold.scores["kappa"],
                }
            )
        pd.DataFrame(records).to_csv(output, index=False)
        print(
            f"    seed {seed} product_readout kappa "
            f"{result.mean('kappa'):+.3f} "
            f"({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )
    return pd.DataFrame(records)


def paired(first: np.ndarray, second: np.ndarray) -> dict:
    """Difference, bootstrap interval, Hedges' g and exact p for one pair."""
    difference = first - second
    rng = np.random.default_rng(0)
    boot = difference[
        rng.integers(0, difference.size, (N_BOOTSTRAP, difference.size))
    ].mean(axis=1)
    n = difference.size
    correction = 1 - 3 / (4 * (n - 1) - 1)
    return {
        "difference": difference.mean(),
        "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
        "hedges_g": difference.mean() / difference.std(ddof=1) * correction,
        "p_perm": exact_permutation_p(difference),
        "wins": int((difference > 0).sum()),
    }


def main() -> None:
    """Run the control and test it alongside the reported comparisons."""
    rows = []
    for dataset in WIDTHS:
        print(f"{dataset}", flush=True)
        control = run(dataset)
        seeds = pd.read_csv(RESULTS / f"seeds_{dataset}.csv")
        frame = pd.concat([seeds, control], ignore_index=True)
        means = frame.groupby(["model", "subject"]).kappa.mean().unstack("model")
        pairs = [*COMPARISONS, ("hybrid", "product_readout")]
        tests = [
            {
                "dataset": dataset,
                "comparison": f"{a} - {b}",
                **paired(means[a].to_numpy(), means[b].to_numpy()),
            }
            for a, b in pairs
        ]
        for test, adjusted in zip(
            tests, holm([t["p_perm"] for t in tests]), strict=False
        ):
            test["p_holm"] = adjusted
        rows.extend(tests)
        print(means.mean().round(3).to_string(), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "readout_tests.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
