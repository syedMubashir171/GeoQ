"""Minimum Distance to Riemannian Mean (MDM) classifier.

The simplest strong baseline in Riemannian BCI, and the first classifier in
this framework. Training estimates one Frechet mean per class; prediction
assigns each trial to the nearest centroid under the chosen metric.

Why this baseline is the honest one to beat
-------------------------------------------
MDM has no hyperparameters beyond the metric, no regularisation to tune, and
no capacity to overfit a training fold: a class centroid is a summary of that
class and nothing more. That makes it nearly impossible to inflate by accident,
which is exactly why it is the right first comparison for any quantum claim. A
quantum model that cannot beat MDM under a strict subject-independent protocol
has not demonstrated anything, regardless of how it compares to a weaker or
more heavily tuned alternative.

It is also fast to reason about when a result looks wrong. Tangent-space
pipelines involve a projection, a scaler, and a discriminative model, any of
which can leak. MDM has one learned object per class.

Leakage
-------
The centroids are estimated in :meth:`fit` and reused verbatim in
:meth:`predict`, exactly as in
:class:`geoq.features.tangent_space.TangentSpace`. Nothing is recomputed at
prediction time, so within a :class:`sklearn.pipeline.Pipeline` a fold's
centroids cannot observe their own test data.

References
----------
Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). Multiclass
    brain-computer interface classification by Riemannian geometry.
    *IEEE TBME*, 59(4), 920-928.
Congedo, M., Barachant, A., & Bhatia, R. (2017). Riemannian geometry for
    EEG-based brain-computer interfaces: a primer and a review.
    *Brain-Computer Interfaces*, 4(3), 155-174.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from geoq.geometry.riemannian import (
    DEFAULT_MEAN_MAX_ITER,
    DEFAULT_MEAN_TOL,
    METRICS,
    Metric,
    distance,
    frechet_mean,
)
from geoq.geometry.spd import FloatArray, check_spd

__all__ = ["MDM"]

logger = logging.getLogger(__name__)

#: Metrics with an implemented mean. Stein has a distance but its mean
#: requires a distinct fixed-point iteration that is not implemented, and
#: silently substituting the AIRM mean under a Stein label would be wrong.
SUPPORTED_METRICS: tuple[str, ...] = ("airm", "logeuclid", "euclid")


class MDM(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Classify SPD matrices by distance to per-class Riemannian means.

    Args:
        metric: Geometry used for both the class means and the distances. One
            of ``"airm"``, ``"logeuclid"``, ``"euclid"``. AIRM is the default
            and the principled choice; the others are available so that the
            geometry can be varied as an experimental factor rather than
            assumed.
        tol: Convergence tolerance for the iterative AIRM mean.
        max_iter: Iteration cap for the iterative AIRM mean.
        warn_on_non_convergence: Whether a class mean that hits the iteration
            cap is reported through the logger.

    Attributes:
        classes_: Sorted class labels seen during :meth:`fit`.
        centroids_: Class means, SPD of shape
            ``(n_classes, n_channels, n_channels)``, ordered to match
            ``classes_``.
        n_channels_: Channel count seen during :meth:`fit`.
        n_features_in_: Scikit-learn convention, set to ``n_channels``.
        centroid_converged_: Per-class convergence flags.
        centroid_n_iter_: Per-class iteration counts.

    Example:
        >>> import numpy as np
        >>> from geoq.geometry.spd import random_spd
        >>> rng = np.random.default_rng(0)
        >>> x = random_spd(4, rng=rng, batch=40)
        >>> y = rng.integers(0, 2, size=40)
        >>> MDM().fit(x, y).predict(x).shape
        (40,)
    """

    def __init__(
        self,
        metric: Metric = "airm",
        *,
        tol: float = DEFAULT_MEAN_TOL,
        max_iter: int = DEFAULT_MEAN_MAX_ITER,
        warn_on_non_convergence: bool = True,
    ) -> None:
        """Store hyperparameters unmodified; see the class docstring for detail."""
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.warn_on_non_convergence = warn_on_non_convergence

    # ----------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------- #

    def _validate_parameters(self) -> None:
        """Check constructor arguments at fit time.

        Raises:
            ValueError: If any argument is outside its valid range.
        """
        if self.metric not in SUPPORTED_METRICS:
            if self.metric in METRICS:
                raise ValueError(
                    f"metric={self.metric!r} has a distance but no implemented "
                    f"mean, so no class centroid can be formed. Substituting "
                    f"the affine-invariant mean under a different label would "
                    f"report a geometry that was not used. Choose one of "
                    f"{list(SUPPORTED_METRICS)}."
                )
            raise ValueError(
                f"metric must be one of {list(SUPPORTED_METRICS)}, got {self.metric!r}."
            )
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError(f"tol must be a positive finite float, got {self.tol!r}.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be a positive integer, got {self.max_iter!r}."
            )

    def _validate_covariances(self, x: ArrayLike, *, name: str = "X") -> FloatArray:
        """Validate a stack of covariance matrices.

        Scikit-learn's ``check_array`` rejects three-dimensional input, and
        flattening would destroy the structure the geometry operates on, so
        validation goes through :func:`geoq.geometry.spd.check_spd` instead.

        Args:
            x: Candidate array of shape ``(n_trials, n_channels, n_channels)``.
            name: Symbol name used in error messages.

        Returns:
            The validated array.

        Raises:
            ValueError: If the array is not a non-empty 3-d stack.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(
                f"{name} must have shape (n_trials, n_channels, n_channels), "
                f"got {arr.shape}. MDM operates on covariance matrices; place "
                f"a Covariances step before it if these are raw epochs."
            )
        if arr.shape[0] == 0:
            raise ValueError(f"{name} contains no trials.")
        return check_spd(arr, name=name)

    # ----------------------------------------------------------------- #
    # Estimator interface
    # ----------------------------------------------------------------- #

    def fit(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> MDM:
        """Estimate one class centroid from the training data.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.
            y: Labels of shape ``(n_trials,)``.
            sample_weight: Optional non-negative weights of shape
                ``(n_trials,)``, applied within each class. Useful for
                reweighting an imbalanced training fold without discarding
                trials.

        Returns:
            The fitted classifier.

        Raises:
            ValueError: If shapes disagree or fewer than two classes are
                present.
        """
        self._validate_parameters()
        x = self._validate_covariances(X)
        labels = np.asarray(y)

        check_classification_targets(labels)
        if labels.shape[0] != x.shape[0]:
            raise ValueError(
                f"X has {x.shape[0]} trials but y has {labels.shape[0]} labels."
            )

        self.classes_ = np.unique(labels)
        if self.classes_.shape[0] < 2:
            raise ValueError(
                f"MDM needs at least two classes, found "
                f"{self.classes_.shape[0]}: {self.classes_.tolist()}. A "
                f"single-class fold usually means a cross-validation split "
                f"was made without stratifying, or a subject performed only "
                f"one task condition."
            )

        weights = self._validate_sample_weight(sample_weight, x.shape[0])

        centroids: list[FloatArray] = []
        converged: list[bool] = []
        iterations: list[int] = []

        for label in self.classes_:
            mask = labels == label
            result = frechet_mean(
                x[mask],
                metric=self.metric,
                weights=None if weights is None else weights[mask],
                tol=self.tol,
                max_iter=self.max_iter,
                return_info=True,
            )
            centroids.append(result.mean)
            converged.append(result.converged)
            iterations.append(result.n_iter)

            if self.warn_on_non_convergence and not result.converged:
                logger.warning(
                    "MDM: centroid for class %r did not converge in %d "
                    "iterations (criterion %.3e, %d trials). The centroid is "
                    "the last iterate, so this class's decision boundary is "
                    "displaced by an unknown amount. Inspect the conditioning "
                    "of this fold before interpreting its accuracy.",
                    label,
                    self.max_iter,
                    result.final_criterion,
                    int(mask.sum()),
                )

        self.centroids_ = np.stack(centroids)
        self.centroid_converged_ = np.array(converged)
        self.centroid_n_iter_ = np.array(iterations)
        self.n_channels_ = x.shape[-1]
        self.n_features_in_ = x.shape[-1]
        return self

    @staticmethod
    def _validate_sample_weight(
        sample_weight: ArrayLike | None, n_trials: int
    ) -> FloatArray | None:
        """Validate optional per-trial weights.

        Args:
            sample_weight: Candidate weights, or None.
            n_trials: Expected length.

        Returns:
            The validated weights, or None.

        Raises:
            ValueError: If the shape is wrong or any weight is negative.
        """
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (n_trials,):
            raise ValueError(
                f"sample_weight must have shape ({n_trials},), got {weights.shape}."
            )
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative.")
        return weights

    def transform(self, X: ArrayLike) -> FloatArray:  # noqa: N803
        """Return the distance from each trial to each class centroid.

        Exposing distances rather than only labels lets MDM act as a feature
        extractor, and makes its decisions auditable: a trial sitting almost
        equidistant from two centroids is a genuinely ambiguous one, which an
        accuracy figure alone conceals.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.

        Returns:
            Distances of shape ``(n_trials, n_classes)``, ordered to match
            ``classes_``.
        """
        check_is_fitted(self, "centroids_")
        x = self._validate_covariances(X)
        if x.shape[-1] != self.n_channels_:
            raise ValueError(
                f"Expected {self.n_channels_} channels to match the centroids "
                f"learned during fit, got {x.shape[-1]}."
            )
        # Broadcast trials against centroids: (n_trials, 1, n, n) against
        # (1, n_classes, n, n) gives every pairing in one vectorised call.
        return distance(
            x[:, None],
            self.centroids_[None, :],
            metric=self.metric,
            validate=False,
        )

    def predict(self, X: ArrayLike) -> NDArray[Any]:  # noqa: N803
        """Assign each trial to its nearest class centroid.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.

        Returns:
            Predicted labels of shape ``(n_trials,)``, in the dtype of the
            labels seen during :meth:`fit`.
        """
        #  Checked here as well as in transform: Python evaluates the
        #  subscripted object before the subscript, so `self.classes_[...]`
        #  would raise AttributeError before transform ever ran, and callers
        #  catching NotFittedError would miss it.
        check_is_fitted(self, "centroids_")
        return self.classes_[np.argmin(self.transform(X), axis=1)]

    def predict_proba(self, X: ArrayLike) -> FloatArray:  # noqa: N803
        """Softmax over negative squared distances.

        Matches pyRiemann's convention, so scores are comparable with
        published results. Note what this is and is not: a monotone transform
        of distance, not a calibrated posterior. Reporting it as a confidence
        would overstate what MDM knows, and any use of these numbers for
        thresholding or expected-cost decisions needs explicit calibration
        first.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.

        Returns:
            Row-stochastic array of shape ``(n_trials, n_classes)``.
        """
        squared = -(self.transform(X) ** 2)
        # Subtract the row maximum before exponentiating. Without it, squared
        # distances of a few hundred -- routine for ill-conditioned
        # covariances -- underflow every entry of the row to zero and the
        # normalisation produces NaN.
        stabilised = squared - squared.max(axis=1, keepdims=True)
        exponentiated = np.exp(stabilised)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def __sklearn_tags__(self):  # noqa: D105
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        return tags

    def _more_tags(self) -> dict[str, Any]:
        return {"X_types": ["3darray"], "requires_y": True}
