"""Does depth rescue the hybrid, and does the optimiser limit it?

Two questions left open by the hybrid prototype
------------------------------------------------
The prototype found the hybrid level with a matched classical model at four
and six qubits and well behind it at eight, where the classical model kept
improving and the circuit did not. It also found three layers at chance
regardless of width while five layers worked. Two explanations survive that
result, and they call for different responses.

The first is that the circuit is underparametrised: five layers is simply
not deep enough to use eight components, and more depth would close the gap.
The second is that the ceiling is real and depth will not help, in which case
the limit is the encoding rather than the parameter count.

Sweeping depth at a fixed width separates them. Accuracy that keeps rising
with layers supports the first; a plateau supports the second, and a plateau
together with a training cost that stops falling would point to an
optimisation failure rather than a representational one.

The optimiser question
----------------------
Vanilla gradient descent follows the steepest direction in parameter space,
which need not be the steepest direction in the space of quantum states the
parameters describe. Quantum natural gradient rescales the update by the
Fubini-Study metric tensor, which is the natural geometry of that space, and
can converge in far fewer steps on landscapes where the two disagree
(Stokes et al., 2020).

Whether it helps here is worth knowing either way. If it does, the hybrid's
ceiling was partly an optimisation artefact and there is a geometric
optimisation result for the paper. If it does not, the ceiling is a property
of the model, which is a cleaner claim than leaving the question open.

The metric tensor costs more per step than a plain gradient, so the honest
comparison is convergence against wall-clock time as well as against step
count. Reporting only steps would flatter whichever method has the more
expensive step.

References
----------
Stokes, J. et al. (2020). Quantum natural gradient. *Quantum*, 4, 269.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.hybrid import ReuploadingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper2"
)
RESULTS.mkdir(parents=True, exist_ok=True)


def depth_sweep(tangent, labels, subjects) -> None:
    """Does more depth close the gap at eight qubits?

    Eight qubits is where the prototype's gap opened, so it is the width at
    which the question has to be asked. Five layers is repeated from the
    prototype as a consistency check: a different value here would mean
    something had drifted.
    """
    output = RESULTS / "depth_sweep.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {r["n_layers"] for r in records}

    splitter = LeaveOneSubjectOut()
    for n_layers in (5, 7, 9):
        if n_layers in done:
            print(f"  {n_layers} layers: cached", flush=True)
            continue
        started = time.perf_counter()
        result = evaluate(
            ReuploadingClassifier(n_qubits=8, n_layers=n_layers),
            tangent,
            labels,
            groups=subjects,
            splitter=splitter,
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
    print(
        "\n"
        + frame[["n_layers", "n_parameters", "kappa", "minutes"]]
        .round(3)
        .to_string(index=False)
    )
    print("\n  The matched classical model reached 0.409 at this width. A")
    print("  hybrid that plateaus below it as depth grows is limited by its")
    print("  encoding rather than by its parameter count.")


def optimiser_comparison(tangent, labels, subjects) -> None:
    """Vanilla gradient descent against quantum natural gradient.

    Run on a single held-out subject rather than the full protocol, because
    the question is about convergence rather than about generalisation, and
    the metric tensor makes each step several times more expensive.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    splitter = LeaveOneSubjectOut()
    train_index, _ = next(iter(splitter.split(tangent, labels, subjects)))

    reducer = make_pipeline(StandardScaler(), PCA(4, random_state=0), MaxAbsScaler())
    features = pnp.array(
        reducer.fit_transform(tangent[train_index]), requires_grad=False
    )
    targets = pnp.array(
        np.where(labels[train_index] == 1, 1.0, -1.0), requires_grad=False
    )

    n_qubits, n_layers = 4, 3
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
    subset = slice(0, 256)

    def cost(p):
        return pnp.mean((circuit(p, features[subset]) - targets[subset]) ** 2)

    rows = []
    for name, optimiser in (
        ("vanilla", qml.AdamOptimizer(stepsize=0.05)),
        #  The natural-gradient optimiser needs the QNode to compute the
        #  metric tensor, which is why it is passed at step time rather than
        #  at construction.
        ("qng", qml.QNGOptimizer(stepsize=0.05)),
    ):
        params = pnp.array(start.copy(), requires_grad=True)
        began = time.perf_counter()
        for step in range(40):
            if name == "qng":
                params, value = optimiser.step_and_cost(
                    cost,
                    params,
                    metric_tensor_fn=lambda p, _c=circuit: (
                        qml.metric_tensor(_c, approx="block-diag")(
                            p, features[subset][:1]
                        )
                    ),
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
        print(
            f"  {name:8s} final cost {rows[-1]['cost']:.4f} after "
            f"{rows[-1]['seconds']:.1f} s",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "optimiser_comparison.csv", index=False)

    print("\n  cost by step, sampled:")
    pivot = frame[frame.step.isin([0, 9, 19, 29, 39])].pivot(
        index="step", columns="optimiser", values="cost"
    )
    print(pivot.round(4).to_string())

    final = frame.groupby("optimiser").last()
    print("\n  seconds per step:")
    for name in ("vanilla", "qng"):
        print(f"    {name:8s} {final.loc[name, 'seconds'] / 40:.3f}")
    print("\n  Natural gradient is worth its cost only if it reaches a lower")
    print("  cost in less wall-clock time, not merely in fewer steps.")


def main(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Run both investigations."""
    data = load_dataset(dataset_name, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)

    print("=" * 68)
    print("Does depth rescue the hybrid at eight qubits?")
    print("=" * 68)
    depth_sweep(tangent, labels, data.subjects)

    print("\n" + "=" * 68)
    print("Vanilla gradient descent against quantum natural gradient")
    print("=" * 68)
    optimiser_comparison(tangent, labels, data.subjects)


if __name__ == "__main__":
    main()
