"""Quantum kernel pipeline, evaluated by the protocol used for the baselines.

Design
------
A quantum kernel classifier is assembled from three parts: a reducer that
maps tangent-space features down to the qubit count, the fidelity kernel of
:mod:`geoq.models.quantum.kernel`, and a support vector machine. Only the
reducer and the SVM fit parameters, and both do so inside the
cross-validation fold.

The comparison against the classical baselines is decided by the estimator
and by nothing else. The splitter, the metrics, the corrected resampled
t-test and the subject-level mixed model are all the ones used for MDM and
TS+LDA, so a difference between the two can be attributed to the classifier
rather than to the evaluation.

Why the kernel is precomputed and the reducer is not
----------------------------------------------------
A kernel entry depends only on the two trials it relates, never on labels or
on which fold they fall in, so computing the whole matrix once and slicing
it per fold leaks nothing and is roughly nine times faster than recomputing
the relevant blocks in each of nine folds.

The reducer is different. It fits parameters, so a reducer fitted on all
the data would carry test information into the training fold no matter how
carefully the classifier is evaluated. This is exactly the kind of leak that
is invisible in a results table, and it is why the two are separated here:
:func:`precompute_kernel` runs once over fixed features, and
:class:`QuantumKernelClassifier` fits its reducer inside the fold.

The consequence is a constraint. Precomputation requires the features
entering the kernel to be the same in every fold, so it is available only
when the reduction is itself fold-independent, such as a fixed selection of
channels. When the reducer is fitted, the kernel must be computed per fold
and the run costs about nine times more. Both paths are provided, and which
was used is recorded with the result.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from geoq.models.quantum.kernel import MAX_QUBITS, QuantumKernel

__all__ = [
    "QuantumKernelClassifier",
    "precompute_kernel",
]

logger = logging.getLogger(__name__)


def precompute_kernel(
    features: ArrayLike,
    *,
    feature_map: str = "zz",
    reps: int = 1,
    scale: float = 1.0,
    log_every: int = 100,
) -> NDArray[np.float64]:
    """Compute the full kernel matrix once, for slicing per fold.

    Valid only when the features entering the kernel do not depend on the
    fold. Where a fitted reducer is used, the kernel must be recomputed
    inside each fold and this function does not apply.

    Args:
        features: Array of shape ``(n_trials, n_qubits)``.
        feature_map: Circuit family.
        reps: Repetitions of the encoding block.
        scale: Multiplier applied before encoding.
        log_every: Progress logging interval.

    Returns:
        The symmetric kernel matrix.
    """
    kernel = QuantumKernel(
        feature_map=feature_map, reps=reps, scale=scale, log_every=log_every
    )
    logger.info(
        "Precomputing a %d x %d kernel. At roughly one millisecond per entry "
        "this is the dominant cost of the experiment.",
        len(features),
        len(features),
    )
    return kernel(features)


class QuantumKernelClassifier(ClassifierMixin, BaseEstimator):
    """Reduce, encode, and classify with a support vector machine.

    Args:
        n_qubits: Output dimension of the reducer, and the width of the
            circuit. Must not exceed the simulable limit.
        feature_map: Circuit family; see
            :data:`geoq.models.quantum.kernel.FEATURE_MAPS`.
        reps: Repetitions of the encoding block.
        scale: Multiplier applied to features before encoding. This governs
            kernel concentration and belongs in a hyperparameter search
            rather than being fixed by convention.
        C: Regularisation strength of the support vector machine.
        reducer: ``"pca"`` for principal components, or ``"none"`` when the
            features already have dimension ``n_qubits``.
        standardise: Whether to standardise before reduction. Left on by
            default because the encoded rotations are angles, so features on
            very different scales would occupy the circuit unevenly.

    Attributes:
        classes_: The class labels seen during fit.
        kernel_: The configured kernel object.
        train_features_: The reduced training features, retained because the
            kernel between test and training data is needed at predict time.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, (20, 12))
        >>> y = rng.integers(0, 2, 20)
        >>> model = QuantumKernelClassifier(n_qubits=4, feature_map="angle")
        >>> model.fit(x, y).predict(x[:3]).shape
        (3,)
    """

    def __init__(
        self,
        n_qubits: int = 8,
        *,
        feature_map: str = "zz",
        reps: int = 1,
        scale: float = 1.0,
        C: float = 1.0,  # noqa: N803
        reducer: str = "pca",
        standardise: bool = True,
    ) -> None:
        """Store hyperparameters; see the class docstring."""
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.reps = reps
        self.scale = scale
        self.C = C
        self.reducer = reducer
        self.standardise = standardise

    def _validate(self) -> None:
        """Check hyperparameters at fit time.

        Raises:
            ValueError: If any value is out of range.
        """
        if not 1 <= self.n_qubits <= MAX_QUBITS:
            raise ValueError(
                f"n_qubits must lie between 1 and {MAX_QUBITS}, got "
                f"{self.n_qubits}. The upper limit is where statevector "
                f"simulation stops being affordable; see the kernel module."
            )
        if self.reducer not in {"pca", "none"}:
            raise ValueError(f"reducer must be 'pca' or 'none', got {self.reducer!r}.")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}.")

    def _reduce_fit(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Fit the reduction chain and return the reduced training features.

        Both steps fit parameters and are therefore fitted here, inside the
        fold, rather than once over the whole dataset.

        Args:
            X: Training features.

        Returns:
            The reduced features.

        Raises:
            ValueError: If no reduction is requested and the input dimension
                does not already match the qubit count.
        """
        self.scaler_ = StandardScaler() if self.standardise else None
        reduced = self.scaler_.fit_transform(X) if self.scaler_ else X

        if self.reducer == "none":
            if reduced.shape[1] != self.n_qubits:
                raise ValueError(
                    f"reducer='none' requires the input dimension to equal "
                    f"n_qubits, but the input has {reduced.shape[1]} features "
                    f"and n_qubits is {self.n_qubits}."
                )
            self.pca_ = None
            return reduced

        n_components = min(self.n_qubits, *reduced.shape)
        if n_components < self.n_qubits:
            logger.warning(
                "Only %d components are available from %d samples of %d "
                "features, fewer than the %d qubits requested. The circuit "
                "will encode padded zeros, which carry no information.",
                n_components,
                reduced.shape[0],
                reduced.shape[1],
                self.n_qubits,
            )
        self.pca_ = PCA(n_components=n_components, random_state=0)
        return self._pad(self.pca_.fit_transform(reduced))

    def _reduce_transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Apply the fitted reduction chain."""
        reduced = self.scaler_.transform(X) if self.scaler_ else X
        if self.pca_ is None:
            return reduced
        return self._pad(self.pca_.transform(reduced))

    def _pad(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Pad with zeros when fewer components exist than qubits."""
        if X.shape[1] == self.n_qubits:
            return X
        padded = np.zeros((X.shape[0], self.n_qubits))
        padded[:, : X.shape[1]] = X
        return padded

    def fit(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike,
    ) -> QuantumKernelClassifier:
        """Fit the reducer and the support vector machine.

        Args:
            X: Features of shape ``(n_samples, n_features)``.
            y: Class labels.

        Returns:
            The fitted classifier.
        """
        self._validate()
        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if features.ndim != 2:
            raise ValueError(
                f"X must have shape (n_samples, n_features), got {features.shape}."
            )

        self.classes_ = np.unique(labels)
        self.n_features_in_ = features.shape[1]
        self.train_features_ = self._reduce_fit(features)
        self.kernel_ = QuantumKernel(
            feature_map=self.feature_map, reps=self.reps, scale=self.scale
        )

        gram = self.kernel_(self.train_features_)
        self._report_concentration(gram)
        self.svm_ = SVC(C=self.C, kernel="precomputed")
        self.svm_.fit(gram, labels)
        return self

    def _report_concentration(self, gram: NDArray[np.float64]) -> None:
        """Warn when the kernel has concentrated to near the identity.

        A kernel whose off-diagonal entries have collapsed carries no
        information, and an SVM fitted on it memorises the training set. The
        symptom is a training accuracy near one with chance test
        performance, which is easy to misread as overfitting rather than as
        a degenerate kernel, so it is reported explicitly.
        """
        off_diagonal = gram[~np.eye(gram.shape[0], dtype=bool)]
        self.kernel_mean_ = float(off_diagonal.mean())
        self.kernel_std_ = float(off_diagonal.std())
        if self.kernel_std_ < 0.01:
            #  A vanishing spread has two distinct causes with opposite
            #  remedies, so the message names which one occurred. Reporting
            #  both as "close to the identity" would send the reader in the
            #  wrong direction half the time.
            direction = (
                "close to a matrix of ones, so every pair looks identical "
                "and the scale should be increased"
                if self.kernel_mean_ > 0.5
                else "close to the identity, so every pair looks unrelated "
                "and the scale should be decreased"
            )
            logger.warning(
                "Kernel off-diagonal entries have mean %.4f and standard "
                "deviation %.4f. The matrix is %s. A kernel with this little "
                "spread carries almost no information, and the classifier "
                "will memorise its training set.",
                self.kernel_mean_,
                self.kernel_std_,
                direction,
            )

    def predict(self, X: ArrayLike) -> NDArray[Any]:  # noqa: N803
        """Predict class labels.

        Args:
            X: Features of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels.
        """
        check_is_fitted(self, "svm_")
        features = np.asarray(X, dtype=np.float64)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features to match fit, got "
                f"{features.shape[1]}."
            )
        reduced = self._reduce_transform(features)
        return self.svm_.predict(self.kernel_(reduced, self.train_features_))

    def __sklearn_tags__(self):  # noqa: D105
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = True
        return tags
