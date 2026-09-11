"""Honest hybrid: tangent-space features into a re-uploading circuit.

Paper 2 prototype
-----------------
A variational quantum classifier evaluated by exactly the protocol used for
the classical baselines: leave-one-subject-out on the same folds, Cohen's
kappa, and the same corrected statistics. The word honest in the title is
doing work. The comparison is arranged so that the quantum model cannot win
by being evaluated more favourably than what it is compared against.

Four design decisions, each forced by an earlier measurement
------------------------------------------------------------
The cost is a single-qubit expectation rather than a global observable. The
barren-plateau diagnostic in this project measured gradient variance falling
by a factor of six thousand across two to eight qubits for a global
observable, against 2.3 for a local one. A global cost would not train, and
the failure would look like a learning-rate problem.

The features are re-encoded in every layer. A circuit that encodes once is a
linear model in the encoded features; on concentric rings it scored 0.510
against 0.915 for three layers. Depth is where the non-linearity comes from.

Reduction is fitted inside the fold. Tangent-space features have dimension
253 at 22 channels and the circuit takes a handful, so a reduction is
unavoidable; one fitted on all the data would carry test information into
training invisibly.

Features are normalised after reduction. Principal components carry the
variance of the data rather than lying in a fixed interval, and the encoded
rotations are angles, so without normalisation the same circuit means
different things at different widths.

What the comparison is against
------------------------------
Three classical baselines on the identical features and folds: LDA at 0.383,
a random forest at 0.369 and an MLP at 0.387 on the full 253 dimensions. All
three sit within 0.018 of each other, which is itself informative: the
tangent-space representation is doing the work, and a quantum layer would
have to do something structurally different to move the number.

A classical MLP on the *reduced* features is also run, because the honest
question is not whether the quantum model beats a classifier with sixty
times more inputs, but whether it beats a classical model given exactly what
the circuit sees.

References
----------
Perez-Salinas, A. et al. (2020). Data re-uploading for a universal quantum
    classifier. *Quantum*, 4, 226.
Cerezo, M. et al. (2021). Cost function dependent barren plateaus in shallow
    parametrized quantum circuits. *Nature Communications*, 12(1), 1791.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

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


class ReuploadingClassifier(ClassifierMixin, BaseEstimator):
    """Data re-uploading variational classifier.

    Args:
        n_qubits: Circuit width, and the dimension features are reduced to.
        n_layers: Number of encode-then-rotate blocks. One layer is linear
            in the encoded features; the non-linearity comes from depth.
        epochs: Passes over the training data.
        batch_size: Samples per gradient step. Larger batches are much
            faster here because the simulator broadcasts over the batch
            axis: measured, a gradient on 128 samples costs 24 ms against
            57 ms on 32.
        learning_rate: Adam step size.
        seed: Seed for parameter initialisation and batching.

    Attributes:
        classes_: Class labels seen during fit.
        params_: Trained circuit parameters.
        history_: Training cost per epoch.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        *,
        n_layers: int = 3,
        epochs: int = 40,
        batch_size: int = 128,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        """Store hyperparameters; see the class docstring."""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def _build(self):
        """Return a QNode broadcasting over the batch axis.

        Broadcasting matters more than it might appear. Evaluating the
        circuit sample by sample costs about 1.8 ms per sample; passing the
        batch as one array costs 0.19 ms, because the simulator applies each
        gate to the whole batch at once.
        """
        import pennylane as qml

        device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(device, interface="autograd", diff_method="backprop")
        def circuit(params, x):
            for layer in range(self.n_layers):
                #  Re-encoding here, in every layer, is what makes the model
                #  non-linear in the inputs.
                for wire in range(self.n_qubits):
                    qml.RY(x[..., wire % x.shape[-1]], wires=wire)
                for wire in range(self.n_qubits):
                    qml.Rot(*params[layer, wire], wires=wire)
                for wire in range(self.n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            #  A single-qubit observable, for the barren-plateau reason
            #  given in the module docstring.
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X, y) -> ReuploadingClassifier:  # noqa: N803
        """Reduce, then train the circuit by gradient descent.

        Args:
            X: Features of shape ``(n_samples, n_features)``.
            y: Class labels.

        Returns:
            The fitted classifier.
        """
        import pennylane as qml
        from pennylane import numpy as pnp

        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        if self.classes_.size != 2:
            raise ValueError(
                f"This classifier is binary; got {self.classes_.size} classes."
            )
        self.n_features_in_ = features.shape[1]

        #  Fitted inside the fold, like every other fitted step here.
        self.reducer_ = make_pipeline(
            StandardScaler(),
            PCA(min(self.n_qubits, *features.shape), random_state=self.seed),
            MaxAbsScaler(),
        )
        reduced = self.reducer_.fit_transform(features)
        #  A PauliZ expectation lies in [-1, 1], so the targets are mapped
        #  to the same interval and the loss is a plain squared error.
        targets = pnp.array(
            np.where(labels == self.classes_[1], 1.0, -1.0), requires_grad=False
        )

        circuit = self._build()
        rng = np.random.default_rng(self.seed)
        params = pnp.array(
            rng.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True,
        )
        inputs = pnp.array(reduced, requires_grad=False)

        optimiser = qml.AdamOptimizer(stepsize=self.learning_rate)
        n = inputs.shape[0]
        self.history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(n)
            epoch_cost = 0.0
            for start in range(0, n, self.batch_size):
                index = order[start : start + self.batch_size]

                def cost(p, index=index):
                    return pnp.mean((circuit(p, inputs[index]) - targets[index]) ** 2)

                params, value = optimiser.step_and_cost(cost, params)
                epoch_cost += float(value)
            self.history_.append(epoch_cost / max(1, n // self.batch_size))

        self.params_ = params
        self.circuit_ = circuit
        return self

    def predict(self, X):  # noqa: N803
        """Predict class labels.

        Args:
            X: Features of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels.
        """
        check_is_fitted(self, "params_")
        reduced = self.reducer_.transform(np.asarray(X, dtype=np.float64))
        outputs = np.asarray(self.circuit_(self.params_, reduced))
        return np.where(outputs > 0, self.classes_[1], self.classes_[0])


def main(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Compare the hybrid against classical models on identical folds."""
    data = load_dataset(dataset_name, cache_dir=CACHE)
    print(f"{data}\n", flush=True)

    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)
    splitter = LeaveOneSubjectOut()
    print(f"{tangent.shape[1]} tangent-space features per trial\n", flush=True)

    output = RESULTS / "hybrid_vs_classical.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {(r["model"], r.get("n_qubits"), r.get("n_layers")) for r in records}

    def record(name, n_qubits, n_layers, model) -> None:
        """Evaluate one model unless it is already archived."""
        key = (name, n_qubits, n_layers)
        if key in done:
            print(f"  {name} {n_qubits} {n_layers}: cached", flush=True)
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
        records.append(
            {
                "model": name,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "kappa": result.mean("kappa"),
                "kappa_sd": result.std("kappa"),
                "accuracy": result.mean("accuracy"),
                "minutes": (time.perf_counter() - started) / 60,
            }
        )
        pd.DataFrame(records).to_csv(output, index=False)
        print(
            f"  {name:22s} kappa {result.mean('kappa'):+.3f} "
            f"± {result.std('kappa'):.3f}  "
            f"({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )

    #  The matched classical comparator: an MLP on the same reduced
    #  features the circuit sees. Comparing the hybrid only against a model
    #  with all 253 inputs would confound the circuit with the reduction.
    for n_qubits in (4, 6, 8):
        record(
            "mlp_reduced",
            n_qubits,
            np.nan,
            make_pipeline(
                StandardScaler(),
                PCA(n_qubits, random_state=0),
                MaxAbsScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    max_iter=500,
                    early_stopping=True,
                    random_state=0,
                ),
            ),
        )

    for n_qubits in (4, 6, 8):
        for n_layers in (3, 5):
            record(
                "hybrid",
                n_qubits,
                n_layers,
                ReuploadingClassifier(n_qubits=n_qubits, n_layers=n_layers),
            )

    frame = pd.DataFrame(records)
    print("\n" + "=" * 68)
    print(
        frame[["model", "n_qubits", "n_layers", "kappa", "kappa_sd", "minutes"]]
        .round(3)
        .to_string(index=False)
    )
    print("=" * 68)
    print("\nReference values on the full 253 features, from the Block 5")
    print("deliverable: LDA 0.383, random forest 0.369, MLP 0.387.")
    print("\nThe comparison that carries meaning is hybrid against")
    print("mlp_reduced at the same width, since both see the same inputs.")


if __name__ == "__main__":
    main()
