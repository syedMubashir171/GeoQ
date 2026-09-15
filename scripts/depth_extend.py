"""Is the hybrid still climbing at eleven layers, and does QNG help where it matters?

Two corrections to the earlier investigation
---------------------------------------------
The depth sweep found kappa rising from 0.197 at five layers to 0.403 at
nine, against 0.409 for the matched classical model. The increments were
+0.120 then +0.086: decelerating, but not flat. A plateau and a slow climb
look alike over three points, and they support different claims, so the
sweep is extended to eleven layers. Rising past 0.409 would mean the circuit
exceeds the classical model given identical inputs; converging on it would
mean parity is the ceiling.

The optimiser comparison was run on four qubits and three layers, a
configuration the prototype showed sits at chance, where both optimisers
moved the cost only from 1.08 to about 0.97. Comparing optimisers on a model
that barely learns says little about either. It is repeated here on the
nine-layer circuit that reaches parity, which is the landscape the question
is actually about.

Why wall-clock time is reported alongside step count
-----------------------------------------------------
Quantum natural gradient rescales updates by the Fubini-Study metric tensor,
which costs several times more per step than a plain gradient. Measured
here, 0.287 seconds against 0.075. A method that converges in fewer steps
can therefore still lose on time, and reporting steps alone would flatter
whichever method has the more expensive step. Both are given.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hybrid import ReuploadingClassifier

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper2"
)
RESULTS.mkdir(parents=True, exist_ok=True)

CLASSICAL_REFERENCE = 0.409


def extend_depth(tangent, labels, subjects) -> None:
    """Add eleven layers to the existing sweep."""
    output = RESULTS / "depth_sweep.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {r["n_layers"] for r in records}

    for n_layers in (11,):
        if n_layers in done:
            print(f"  {n_layers} layers: cached", flush=True)
            continue
        started = time.perf_counter()
        result = evaluate(
            ReuploadingClassifier(n_qubits=8, n_layers=n_layers),
            tangent,
            labels,
            groups=subjects,
            splitter=LeaveOneSubjectOut(),
            metrics=("accuracy", "kappa"),
        )
        elapsed = (time.perf_counter() - started) / 60
        records.append(
            {
                "n_qubits": 8,
                "n_layers": n_layers,
                "n_parameters": 8 * n_layers * 3,
                "kappa": result.mean("kappa"),
                "kappa_sd": result.std("kappa"),
                "minutes": elapsed,
            }
        )
        pd.DataFrame(records).to_csv(output, index=False)
        print(
            f"  {n_layers} layers ({8 * n_layers * 3} parameters): "
            f"kappa {result.mean('kappa'):+.3f}  ({elapsed:.1f} min)",
            flush=True,
        )

    frame = pd.DataFrame(records).sort_values("n_layers")
    frame["gain"] = frame.kappa.diff()
    print(
        "\n"
        + frame[["n_layers", "n_parameters", "kappa", "gain", "minutes"]]
        .round(3)
        .to_string(index=False)
    )

    best = frame.kappa.max()
    print(
        f"\n  matched classical model {CLASSICAL_REFERENCE:.3f}, "
        f"best hybrid {best:.3f}, gap {best - CLASSICAL_REFERENCE:+.3f}"
    )
    if len(frame) >= 2:
        last_gain = frame.gain.iloc[-1]
        print(f"  final increment {last_gain:+.3f}")
        if abs(last_gain) < 0.02:
            print(
                "  The increments have flattened, so parity rather than"
                "\n  advantage is the ceiling at this width."
            )
        else:
            print(
                "  The increments have not flattened, so the sweep has not"
                "\n  yet found the ceiling and the claim should stay open."
            )


def optimiser_on_working_circuit(tangent, labels, subjects) -> None:
    """Compare optimisers on the circuit that actually learns.

    The earlier comparison used four qubits and three layers, which sits at
    chance. This uses eight qubits and nine layers, the configuration that
    reaches parity with the classical model, so the landscape being compared
    is the one the result depends on.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    train_index, _ = next(iter(LeaveOneSubjectOut().split(tangent, labels, subjects)))
    n_qubits, n_layers = 8, 9
    reducer = make_pipeline(
        StandardScaler(), PCA(n_qubits, random_state=0), MaxAbsScaler()
    )
    features = pnp.array(
        reducer.fit_transform(tangent[train_index]), requires_grad=False
    )
    targets = pnp.array(
        np.where(labels[train_index] == 1, 1.0, -1.0), requires_grad=False
    )

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="autograd")
    def circuit(params, x):
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.RY(x[..., wire % x.shape[-1]], wires=wire)
            for wire in range(n_qubits):
                qml.Rot(*params[layer, wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(0)
    start = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
    #  A fixed subset keeps the cost comparable between optimisers; the
    #  question is the shape of the descent, not generalisation.
    subset = slice(0, 256)

    def cost(p):
        return pnp.mean((circuit(p, features[subset]) - targets[subset]) ** 2)

    rows = []
    for name in ("vanilla", "qng"):
        optimiser = (
            qml.AdamOptimizer(stepsize=0.05)
            if name == "vanilla"
            else qml.QNGOptimizer(stepsize=0.05)
        )
        params = pnp.array(start.copy(), requires_grad=True)
        began = time.perf_counter()
        for step in range(30):
            if name == "qng":
                params, value = optimiser.step_and_cost(
                    cost,
                    params,
                    metric_tensor_fn=lambda p, _c=circuit: qml.metric_tensor(
                        _c, approx="block-diag"
                    )(p, features[subset][:1]),
                )
            else:
                params, value = optimiser.step_and_cost(cost, params)
            rows.append(
                {
                    "optimiser": name,
                    "step": step,
                    "cost": float(value),
                    "seconds": time.perf_counter() - began,
                }
            )
            if step % 10 == 0:
                print(
                    f"    {name:8s} step {step:2d}  cost {float(value):.4f}", flush=True
                )
        print(
            f"  {name:8s} final {rows[-1]['cost']:.4f} after "
            f"{rows[-1]['seconds']:.1f} s",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "optimiser_9layer.csv", index=False)

    print("\n  cost by step:")
    print(
        frame[frame.step.isin([0, 9, 19, 29])]
        .pivot(index="step", columns="optimiser", values="cost")
        .round(4)
        .to_string()
    )

    final = frame.groupby("optimiser").last()
    vanilla, qng = final.loc["vanilla"], final.loc["qng"]
    print(
        f"\n  per step:  vanilla {vanilla.seconds / 30:.3f} s, "
        f"qng {qng.seconds / 30:.3f} s"
    )
    print(f"  final cost: vanilla {vanilla.cost:.4f}, qng {qng.cost:.4f}")

    #  The fair question is what each optimiser achieves in equal time, not
    #  in equal steps.
    budget = min(vanilla.seconds, qng.seconds)
    for name in ("vanilla", "qng"):
        cell = frame[(frame.optimiser == name) & (frame.seconds <= budget)]
        print(
            f"  cost after {budget:.1f} s of {name:8s}: "
            f"{cell.cost.iloc[-1]:.4f} at step {int(cell.step.iloc[-1])}"
        )


def main(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Run both investigations."""
    data = load_dataset(dataset_name, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)

    print("=" * 68)
    print("Extending the depth sweep to eleven layers")
    print("=" * 68)
    extend_depth(tangent, labels, data.subjects)

    print("\n" + "=" * 68)
    print("Optimisers on the nine-layer circuit that reaches parity")
    print("=" * 68)
    optimiser_on_working_circuit(tangent, labels, data.subjects)


if __name__ == "__main__":
    main()
