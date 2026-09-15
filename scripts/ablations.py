"""Three ablations that decide what the hybrid result means.

Why these three
---------------
The prototype found a re-uploading circuit reaching kappa 0.403 against 0.409
for a classical model on identical inputs, with depth ruled out as the limit
and the optimiser ruled out as well. Three questions remain, and each can
overturn or strengthen the interpretation.

Does entanglement matter? The circuit contains CNOT gates, but nothing so
far shows they contribute. This is not an idle question in this project: the
quantum kernel experiment found its best-performing feature map to be one
with no entangling gates, whose kernel turned out to equal a product of
cosines and to be computable classically in closed form. If removing the
CNOTs here leaves accuracy unchanged, the model is a product-state
classifier and the word quantum is doing no work.

Is the comparison parameter-fair? The classical model reaching 0.409 uses
706 parameters; the circuit reaching 0.403 uses 216. If a classical model
with about 216 parameters also reaches 0.409, the circuit has no efficiency
advantage. If it does not, the circuit matches a larger classical model with
a third of the parameters, which is a positive result and the strongest
claim available from this data.

Was the classical baseline tuned as hard? It was not. The hybrid received a
depth sweep across four values; the classical model used library defaults. A
reviewer will notice, and the asymmetry favours the quantum side. Giving the
classical model a search of comparable size removes the objection whichever
way it resolves.

Each experiment uses the same folds, features, metric and protocol as the
prototype, so the numbers are directly comparable with those already
reported.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
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

#: The configuration that reached parity in the prototype.
BEST_QUBITS, BEST_LAYERS = 8, 9

#: Reference values from the prototype, on identical folds and features.
CLASSICAL_KAPPA = 0.409
HYBRID_KAPPA = 0.403


class UnentangledReuploading(ReuploadingClassifier):
    """The same circuit with the entangling gates removed.

    Without CNOTs the state remains a product across qubits, so the model
    factorises into independent single-qubit functions of the inputs. Any
    accuracy it retains is therefore achievable without entanglement, and
    the comparison isolates what the entangling layer contributes.
    """

    def _build(self):
        """Return the QNode with no CNOT gates."""
        import pennylane as qml

        device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(device, interface="autograd", diff_method="backprop")
        def circuit(params, x):
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.RY(x[..., wire % x.shape[-1]], wires=wire)
                for wire in range(self.n_qubits):
                    qml.Rot(*params[layer, wire], wires=wire)
                #  The entangling loop present in the parent class is
                #  omitted here; everything else is identical, so the
                #  difference in accuracy is attributable to it alone.
            return qml.expval(qml.PauliZ(0))

        return circuit


def mlp_parameters(n_inputs: int, n_hidden: int, n_classes: int = 2) -> int:
    """Count the trainable parameters of a one-hidden-layer network."""
    return n_inputs * n_hidden + n_hidden + n_hidden * n_classes + n_classes


def build_mlp(n_hidden: int, n_qubits: int, **kwargs):
    """A classical pipeline seeing exactly the features the circuit sees."""
    return make_pipeline(
        StandardScaler(),
        PCA(n_qubits, random_state=0),
        MaxAbsScaler(),
        MLPClassifier(
            hidden_layer_sizes=(n_hidden,),
            max_iter=800,
            early_stopping=True,
            random_state=0,
            **kwargs,
        ),
    )


def main(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Run the three ablations and report them against the prototype."""
    data = load_dataset(dataset_name, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)
    splitter = LeaveOneSubjectOut()

    output = RESULTS / "ablations.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {r["name"] for r in records}

    def record(name: str, model, n_parameters: int, note: str) -> None:
        """Evaluate one model unless it is already archived."""
        if name in done:
            print(f"  {name}: cached", flush=True)
            return
        started = time.perf_counter()
        result = evaluate(
            model,
            tangent,
            labels,
            groups=data.subjects,
            splitter=splitter,
            metrics=("accuracy", "kappa"),
        )
        elapsed = (time.perf_counter() - started) / 60
        records.append(
            {
                "name": name,
                "n_parameters": n_parameters,
                "note": note,
                "kappa": result.mean("kappa"),
                "kappa_sd": result.std("kappa"),
                "accuracy": result.mean("accuracy"),
                "minutes": elapsed,
            }
        )
        pd.DataFrame(records).to_csv(output, index=False)
        print(
            f"  {name:24s} kappa {result.mean('kappa'):+.3f} "
            f"± {result.std('kappa'):.3f}  ({n_parameters} params, "
            f"{elapsed:.1f} min)",
            flush=True,
        )

    print("=" * 72)
    print("1. Does entanglement contribute?")
    print("=" * 72)
    record(
        "hybrid_no_entangle",
        UnentangledReuploading(n_qubits=BEST_QUBITS, n_layers=BEST_LAYERS),
        BEST_QUBITS * BEST_LAYERS * 3,
        "CNOT gates removed; everything else identical",
    )
    print(f"\n  The entangled circuit reached {HYBRID_KAPPA:+.3f}. A similar")
    print("  value here would mean the model is a product-state classifier")
    print("  and that its accuracy requires no entanglement.")

    print("\n" + "=" * 72)
    print("2. Is the comparison parameter-fair?")
    print("=" * 72)
    print(
        f"\n  The circuit uses {BEST_QUBITS * BEST_LAYERS * 3} parameters; "
        f"the classical model reaching {CLASSICAL_KAPPA:+.3f} uses "
        f"{mlp_parameters(BEST_QUBITS, 64)}.\n"
    )
    for n_hidden in (8, 16, 24, 64):
        count = mlp_parameters(BEST_QUBITS, n_hidden)
        record(
            f"mlp_{n_hidden}_hidden",
            build_mlp(n_hidden, BEST_QUBITS),
            count,
            f"{n_hidden} hidden units",
        )
    print("\n  A classical model of comparable size failing to reach")
    print(f"  {CLASSICAL_KAPPA:+.3f} would mean the circuit matches a larger")
    print("  classical model with a third of the parameters.")

    print("\n" + "=" * 72)
    print("3. Was the classical baseline tuned as hard as the circuit?")
    print("=" * 72)
    print("\n  The circuit received a sweep over four depths. The classical")
    print("  model is given a search of comparable size over regularisation")
    print("  and learning rate, since the asymmetry otherwise favours the")
    print("  quantum side.\n")
    for alpha in (1e-4, 1e-2):
        for rate in (1e-3, 1e-2):
            record(
                f"mlp_tuned_a{alpha:g}_lr{rate:g}",
                build_mlp(64, BEST_QUBITS, alpha=alpha, learning_rate_init=rate),
                mlp_parameters(BEST_QUBITS, 64),
                f"alpha={alpha:g}, lr={rate:g}",
            )

    frame = pd.DataFrame(records)
    print("\n" + "=" * 72)
    print(
        frame[["name", "n_parameters", "kappa", "kappa_sd", "minutes"]]
        .round(3)
        .to_string(index=False)
    )
    print("=" * 72)

    entangled_note = frame[frame.name == "hybrid_no_entangle"]
    if not entangled_note.empty:
        without = float(entangled_note.kappa.iloc[0])
        print(
            f"\nEntanglement: {HYBRID_KAPPA:+.3f} with, {without:+.3f} "
            f"without, difference {HYBRID_KAPPA - without:+.3f}"
        )

    matched = frame[frame.name.str.startswith("mlp_") & (frame.n_parameters <= 300)]
    if not matched.empty:
        best = matched.kappa.max()
        print(
            f"Parameter-matched classical best: {best:+.3f} at "
            f"{int(matched.loc[matched.kappa.idxmax(), 'n_parameters'])} "
            f"parameters, against {HYBRID_KAPPA:+.3f} for the circuit at "
            f"{BEST_QUBITS * BEST_LAYERS * 3}"
        )

    tuned = frame[frame.name.str.startswith("mlp_tuned")]
    if not tuned.empty:
        print(
            f"Best tuned classical: {tuned.kappa.max():+.3f} against "
            f"{CLASSICAL_KAPPA:+.3f} with defaults"
        )


if __name__ == "__main__":
    main()
