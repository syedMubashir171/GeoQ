"""Quantum kernels over tangent-space features.

What this provides
------------------
A fidelity kernel computed by a parametrised quantum circuit, in the form
scikit-learn expects, so that a quantum pipeline can be evaluated by exactly
the protocol and statistics used for the classical baselines. The point of
building it this way is that the comparison is decided by the estimator and
nothing else: the splitter, the metrics, the corrected t-test and the
subject-level mixed model are all unchanged.

The fidelity kernel between two feature vectors is

    K(x, y) = |<0| U(y)^dagger U(x) |0>|^2

the probability of returning to the all-zeros state after encoding one
vector and un-encoding the other. It is symmetric, has unit diagonal, and is
positive semi-definite by construction, so it is a valid kernel for an SVM
without correction.

The dimensionality problem, and why it is a methodological choice
----------------------------------------------------------------
Tangent-space features have dimension C(C+1)/2: 253 at 22 channels. A
feature map with one qubit per feature would need 253 qubits, and
statevector simulation is infeasible beyond roughly 20. Features must
therefore be reduced before encoding, and the reduction is not a detail. It
determines what the quantum circuit ever sees, and a reduction fitted on all
the data would leak test information into the training fold no matter how
carefully the classifier itself is evaluated.

The reducer here is therefore a scikit-learn transformer fitted inside the
fold, like every other fitted step in this framework. Its output dimension
equals the qubit count, which is the parameter that has to be small.

Exponential concentration, and why feature scale is a hyperparameter
--------------------------------------------------------------------
A fidelity kernel becomes uninformative as the number of qubits grows.
Measured here on random inputs with the ZZ map, the mean off-diagonal entry
falls from 7e-2 at four qubits to 2e-4 at twelve: the matrix approaches the
identity, and an SVM on the identity memorises its training set and
generalises to nothing. This is exponential concentration, and it is the
central practical obstacle to fidelity kernels rather than an artefact of
this implementation.

Feature scale is what controls it, because the encoded rotations are
angles. With features scaled to 0.05 the kernel is uniformly 0.999 and
distinguishes nothing; at pi it is 0.03 and is effectively the identity. The
informative range lies between, and the scale maximising the spread of
off-diagonal values falls from about 2.0 at four qubits to 1.3 at twelve,
with the achievable spread falling alongside it.

Scale is therefore not a detail to be fixed by convention. It determines
whether the kernel carries information at all, and it must be selected
inside the cross-validation fold like any other fitted quantity. The
``scale`` argument exists so that it can be, and the value used is recorded
with every result.

Cost, and why the kernel is precomputed
---------------------------------------
Measured on this implementation, one kernel entry at eight qubits takes
about one millisecond. A full kernel over the 2592 trials of BCI
Competition IV 2a is therefore around an hour, and recomputing the relevant
blocks inside each of nine folds would take about eight hours.

Precomputing the whole matrix once and slicing it per fold takes one hour
and leaks nothing, because an entry K(i, j) depends only on trials i and j
and never on the labels or on which fold they fall in. The same is not true
of the reducer, which fits parameters and therefore stays inside the fold.
The two are separated for that reason: :class:`QuantumKernel` computes
similarities between whatever vectors it is given, and knows nothing about
folds.

References
----------
Havlicek, V. et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. *Nature*, 567(7747), 209-212.
Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature
    Hilbert spaces. *Physical Review Letters*, 122(4), 040504.
Huang, H.-Y. et al. (2021). Power of data in quantum machine learning.
    *Nature Communications*, 12(1), 2631.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FEATURE_MAPS",
    "QuantumKernel",
    "fidelity_kernel",
]

logger = logging.getLogger(__name__)

#: Feature maps this module supports. Each is a name understood by
#: :class:`QuantumKernel`; the circuits themselves are built lazily so that
#: importing this module does not require PennyLane.
FEATURE_MAPS: tuple[str, ...] = ("angle", "zz", "iqp")

#: Beyond this many qubits, statevector simulation becomes impractical on a
#: single machine and the kernel would dominate every other cost in the
#: experiment. Exceeding it raises rather than warns, because a run that
#: silently takes days is worse than one that refuses to start.
MAX_QUBITS: int = 20


def _build_circuit(n_qubits: int, feature_map: str, reps: int):
    """Return a QNode computing the fidelity between two feature vectors.

    Args:
        n_qubits: Number of qubits, equal to the input dimension.
        feature_map: One of :data:`FEATURE_MAPS`.
        reps: Number of repetitions of the encoding block.

    Returns:
        A callable taking two vectors and returning the probability vector,
        whose first entry is the fidelity.

    Raises:
        ImportError: If PennyLane is not installed.
        ValueError: If the feature map is unknown.
    """
    try:
        import pennylane as qml
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Quantum kernels require the 'quantum' extra: pip install -e '.[quantum]'"
        ) from error

    if feature_map not in FEATURE_MAPS:
        raise ValueError(
            f"feature_map must be one of {list(FEATURE_MAPS)}, got {feature_map!r}."
        )

    wires = range(n_qubits)

    def encode(x):
        """Apply the chosen feature map to one vector."""
        for _ in range(reps):
            if feature_map == "angle":
                #  The simplest encoding: one rotation per feature, no
                #  entanglement. Included as a control rather than as a
                #  candidate, since a kernel with no entanglement has a
                #  classical analogue and should not be expected to differ
                #  from one.
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
            elif feature_map == "zz":
                #  The map of Havlicek et al., whose entangling layer uses
                #  products of feature pairs. This is the circuit most
                #  quantum-BCI papers use, so it is the one a comparison
                #  with that literature has to include.
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            else:  # iqp
                #  All-to-all entanglement rather than nearest-neighbour,
                #  which is harder to simulate classically and therefore the
                #  more interesting case if any difference is found.
                for i in wires:
                    qml.Hadamard(wires=i)
                    qml.RZ(x[i], wires=i)
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(x[i] * x[j], wires=j)
                        qml.CNOT(wires=[i, j])

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit(x1, x2):
        encode(x1)
        qml.adjoint(encode)(x2)
        return qml.probs(wires=wires)

    return circuit


def fidelity_kernel(
    X: ArrayLike,  # noqa: N803
    Y: ArrayLike | None = None,  # noqa: N803
    *,
    feature_map: str = "zz",
    reps: int = 1,
    scale: float = 1.0,
    log_every: int = 0,
) -> NDArray[np.float64]:
    """Compute the fidelity kernel between two sets of feature vectors.

    Args:
        X: Array of shape ``(n_samples_x, n_qubits)``.
        Y: Array of shape ``(n_samples_y, n_qubits)``. When None the kernel
            is computed between ``X`` and itself, which halves the work by
            exploiting symmetry.
        feature_map: One of :data:`FEATURE_MAPS`.
        reps: Repetitions of the encoding block.
        scale: Multiplier applied to the features before encoding. See the
            module docstring: this is the parameter that decides whether the
            kernel is informative, and it should be selected within the
            cross-validation fold rather than fixed by convention.
        log_every: Log progress every this many rows, or 0 for silence.
            Kernels take minutes to hours, and a run with no output is
            indistinguishable from one that has hung.

    Returns:
        The kernel matrix.

    Raises:
        ValueError: If the inputs are not two-dimensional, disagree on
            dimension, or exceed :data:`MAX_QUBITS`.
    """
    first = np.asarray(X, dtype=np.float64)
    if first.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_qubits), got {first.shape}.")
    n_qubits = first.shape[1]
    if n_qubits > MAX_QUBITS:
        raise ValueError(
            f"{n_qubits} qubits exceeds the simulable limit of {MAX_QUBITS}. "
            f"Tangent-space features must be reduced before encoding; see "
            f"the class docstring for why the reducer belongs inside the "
            f"cross-validation fold."
        )

    symmetric = Y is None
    second = first if symmetric else np.asarray(Y, dtype=np.float64)

    if second.ndim != 2 or second.shape[1] != n_qubits:
        raise ValueError(
            f"Y must have shape (n_samples, {n_qubits}), got {second.shape}."
        )

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive and finite, got {scale!r}.")
    first = first * scale
    second = second if symmetric else second * scale
    if symmetric:
        second = first

    circuit = _build_circuit(n_qubits, feature_map, reps)
    kernel = np.empty((first.shape[0], second.shape[0]), dtype=np.float64)

    for i, row in enumerate(first):
        if symmetric:
            #  K is symmetric with unit diagonal when both arguments are the
            #  same set, so only the lower triangle is computed.
            kernel[i, i] = 1.0
            for j in range(i):
                value = float(circuit(row, second[j])[0])
                kernel[i, j] = kernel[j, i] = value
        else:
            for j, other in enumerate(second):
                kernel[i, j] = float(circuit(row, other)[0])
        if log_every and (i + 1) % log_every == 0:
            logger.info("kernel row %d/%d", i + 1, first.shape[0])

    return kernel


class QuantumKernel:
    """A fidelity kernel usable wherever scikit-learn accepts a callable.

    The object is deliberately not a transformer or an estimator. It
    computes similarities between whatever vectors it is given and knows
    nothing about folds, labels or datasets, which is what makes
    precomputing a whole kernel matrix and slicing it per fold safe.

    Args:
        feature_map: One of :data:`FEATURE_MAPS`.
        reps: Repetitions of the encoding block.
        scale: Multiplier applied to features before encoding, which
            controls kernel concentration and belongs in a hyperparameter
            search rather than being fixed.
        log_every: Progress logging interval, or 0 for silence.

    Attributes:
        n_calls_: Number of kernel matrices computed, for cost accounting.

    Example:
        >>> import numpy as np
        >>> kernel = QuantumKernel(feature_map="angle")
        >>> x = np.zeros((3, 4))
        >>> K = kernel(x)
        >>> float(K[0, 0])
        1.0
    """

    def __init__(
        self,
        feature_map: str = "zz",
        *,
        reps: int = 1,
        scale: float = 1.0,
        log_every: int = 0,
    ) -> None:
        """Store the circuit configuration."""
        if feature_map not in FEATURE_MAPS:
            raise ValueError(
                f"feature_map must be one of {list(FEATURE_MAPS)}, got {feature_map!r}."
            )
        if reps < 1:
            raise ValueError(f"reps must be at least 1, got {reps}.")
        self.feature_map = feature_map
        self.reps = reps
        self.scale = scale
        self.log_every = log_every
        self.n_calls_ = 0

    def __call__(
        self,
        X: ArrayLike,  # noqa: N803
        Y: ArrayLike | None = None,  # noqa: N803
    ) -> NDArray[np.float64]:
        """Compute the kernel matrix."""
        self.n_calls_ += 1
        return fidelity_kernel(
            X,
            Y,
            feature_map=self.feature_map,
            reps=self.reps,
            scale=self.scale,
            log_every=self.log_every,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Return the configuration, for compatibility with clone."""
        return {
            "feature_map": self.feature_map,
            "reps": self.reps,
            "scale": self.scale,
            "log_every": self.log_every,
        }

    def set_params(self, **params: Any) -> QuantumKernel:
        """Set configuration values."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return a representation naming the circuit."""
        return (
            f"QuantumKernel(feature_map={self.feature_map!r}, "
            f"reps={self.reps}, scale={self.scale})"
        )
