"""Scikit-learn transformer for the Riemannian tangent space.

This is the first component in the framework where data leakage becomes
possible, and the reason it is a scikit-learn ``Transformer`` rather than a
function.

Why the estimator interface is a correctness guarantee, not a style choice
-------------------------------------------------------------------------
The tangent-space projection depends on a reference point, and that reference
is *estimated from data*. Estimating it from the whole dataset before
splitting is a textbook leak: every test trial then contributes to the
coordinate system its own prediction is made in. The effect is small enough to
be invisible in a results table and large enough to inflate accuracy by
several points, which is precisely the class of error this thesis exists to
expose.

Splitting the operation into ``fit`` and ``transform`` makes the leak
structurally impossible inside a :class:`sklearn.pipeline.Pipeline`. The
cross-validator calls ``fit`` with training indices only; ``transform`` is
then called on the test fold using the stored reference and no other
information. There is no code path by which a fold's reference can observe its
own test data, so avoiding the leak stops depending on the author remembering
to avoid it.

On ``tsupdate``
---------------
pyRiemann's ``TangentSpace`` offers ``tsupdate=True``, which re-estimates the
reference point from the data passed to ``transform``. That is deliberately
not implemented here. On a test fold it uses test data to build the feature
space -- defensible as unsupervised test-time adaptation, indefensible when
reported without saying so, and impossible to distinguish from a bug once it
is buried in a pipeline. Cross-subject re-centring is a real and useful
technique; it belongs in an explicit alignment transformer whose name says
what it does, applied per subject with a stated protocol, not as a boolean
flag on a projection.

References
----------
Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). Multiclass
    brain-computer interface classification by Riemannian geometry.
    *IEEE TBME*, 59(4), 920-928.
Varoquaux, G. et al. (2017). Assessing and tuning brain decoders:
    cross-validation, caveats, and guidelines. *NeuroImage*, 145, 166-179.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from geoq.geometry.riemannian import (
    DEFAULT_MEAN_MAX_ITER,
    DEFAULT_MEAN_TOL,
    Metric,
    frechet_mean,
)
from geoq.geometry.spd import FloatArray, check_spd
from geoq.geometry.tangent import (
    matrix_dimension,
    tangent_space,
    untangent_space,
    vector_dimension,
)

__all__ = ["TangentSpace"]

logger = logging.getLogger(__name__)

#: Metrics that have a tangent-space formulation. Stein and Euclid do not.
SUPPORTED_METRICS: tuple[str, ...] = ("airm", "logeuclid")


class TangentSpace(TransformerMixin, BaseEstimator):
    """Project SPD matrices into the tangent space at a learned reference point.

    Turns a stack of covariance matrices of shape ``(n_trials, n_channels,
    n_channels)`` into a feature matrix of shape ``(n_trials, n_channels *
    (n_channels + 1) / 2)`` suitable for any linear classifier. For a
    22-channel montage that is 253 features per trial.

    The reference point is the Frechet mean of the data seen by :meth:`fit`,
    and nothing else. It is stored as ``reference_`` and reused verbatim by
    :meth:`transform`.

    Args:
        metric: ``"airm"`` for the affine-invariant geometry, or
            ``"logeuclid"`` for the cheaper log-Euclidean approximation. AIRM
            is the default because only it makes the feature space isometric
            to the manifold.
        tol: Convergence tolerance for the iterative AIRM mean.
        max_iter: Iteration cap for the iterative AIRM mean.
        warn_on_non_convergence: Whether a mean that hits the iteration cap
            raises a warning through the logger. Leave enabled: a silently
            non-converged reference produces a coordinate system that is
            merely close to the right one, and the resulting accuracy drop
            looks like a modelling result rather than a numerical failure.

    Attributes:
        reference_: The learned reference point, SPD of shape
            ``(n_channels, n_channels)``.
        n_channels_: Number of channels seen during :meth:`fit`.
        n_features_in_: Scikit-learn's convention for input dimensionality.
            Set to ``n_channels`` so that pipeline introspection works.
        n_features_out_: Length of each output feature vector.
        mean_converged_: Whether the reference-point estimation converged.
        mean_n_iter_: Iterations used to estimate the reference point.

    Example:
        >>> import numpy as np
        >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        >>> from sklearn.pipeline import make_pipeline
        >>> from geoq.geometry.spd import random_spd
        >>> rng = np.random.default_rng(0)
        >>> x = random_spd(4, rng=rng, batch=40)
        >>> y = rng.integers(0, 2, size=40)
        >>> pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        >>> _ = pipeline.fit(x, y)
        >>> pipeline.predict(x).shape
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
        #  Scikit-learn requires __init__ to store its arguments unmodified
        #  and to perform no validation. get_params/set_params, clone, and
        #  grid search all depend on it; validating here would break cloning
        #  of an estimator that has not yet been fitted. Validation therefore
        #  happens in fit, which is where scikit-learn expects it.
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
            raise ValueError(
                f"metric must be one of {list(SUPPORTED_METRICS)}, got "
                f"{self.metric!r}. 'stein' and 'euclid' have no tangent-space "
                f"formulation."
            )
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError(f"tol must be a positive finite float, got {self.tol!r}.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be a positive integer, got {self.max_iter!r}."
            )

    def _validate_covariances(self, x: ArrayLike, *, name: str = "X") -> FloatArray:
        """Validate a stack of covariance matrices.

        Scikit-learn's own ``check_array`` rejects three-dimensional input by
        default, and coercing it to two dimensions would destroy the matrix
        structure this entire framework is built on. Validation is therefore
        done through :func:`geoq.geometry.spd.check_spd`, which enforces the
        stronger contract that actually matters: symmetric, positive definite,
        finite.

        Args:
            x: Candidate array of shape ``(n_trials, n_channels, n_channels)``.
            name: Symbol name used in error messages.

        Returns:
            The validated array.

        Raises:
            ValueError: If the array is not a 3-d stack or is empty.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(
                f"{name} must have shape (n_trials, n_channels, n_channels), "
                f"got shape {arr.shape}. If the covariances were flattened "
                f"upstream, reshape before this step: the manifold structure "
                f"is what this transformer operates on."
            )
        if arr.shape[0] == 0:
            raise ValueError(
                f"{name} contains no trials. An empty fold usually means a "
                f"cross-validation split produced a group with no members."
            )
        return check_spd(arr, name=name)

    # ----------------------------------------------------------------- #
    # Estimator interface
    # ----------------------------------------------------------------- #

    def fit(self, X: ArrayLike, y: Any = None) -> TangentSpace:  # noqa: ARG002, N803
        """Estimate the reference point from ``X`` alone.

        Called by a cross-validator with training indices only. ``y`` is
        accepted and ignored, as the scikit-learn API requires; the reference
        point is unsupervised by construction, which is what allows this step
        to sit inside a pipeline without label leakage on top of data leakage.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.
            y: Ignored, present for API compatibility.

        Returns:
            The fitted transformer.
        """
        self._validate_parameters()
        x = self._validate_covariances(X)

        result = frechet_mean(
            x,
            metric=self.metric,
            tol=self.tol,
            max_iter=self.max_iter,
            return_info=True,
        )

        self.reference_ = result.mean
        self.mean_converged_ = result.converged
        self.mean_n_iter_ = result.n_iter
        self.n_channels_ = x.shape[-1]
        self.n_features_in_ = x.shape[-1]
        self.n_features_out_ = vector_dimension(self.n_channels_)

        if self.warn_on_non_convergence and not result.converged:
            logger.warning(
                "TangentSpace: reference point did not converge in %d iterations "
                "(criterion %.3e). Features computed from it describe a "
                "coordinate system near, but not at, the Frechet mean. Inspect "
                "the conditioning of this fold's covariances before "
                "interpreting downstream accuracy.",
                self.max_iter,
                result.final_criterion,
            )

        return self

    def transform(self, X: ArrayLike) -> FloatArray:  # noqa: N803
        """Project ``X`` using the reference point learned during :meth:`fit`.

        Uses ``reference_`` and nothing else. No statistic is recomputed from
        ``X``, which is what makes this call safe on a test fold.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.

        Returns:
            Feature matrix of shape ``(n_trials, n_channels (n_channels + 1) / 2)``.

        Raises:
            ValueError: If the channel count differs from that seen in
                :meth:`fit`.
        """
        check_is_fitted(self, "reference_")
        x = self._validate_covariances(X)
        self._check_channel_count(x.shape[-1])
        return tangent_space(x, self.reference_, metric=self.metric, validate=False)

    def inverse_transform(self, X: ArrayLike) -> FloatArray:  # noqa: N803
        """Map feature vectors back to SPD matrices.

        Needed to interpret anything computed in the feature space as a
        covariance again: a class centroid, a classifier's decision direction
        read as a spatial pattern, or a synthetic trial.

        Args:
            X: Feature matrix of shape ``(n_trials, n_features_out)``.

        Returns:
            SPD stack of shape ``(n_trials, n_channels, n_channels)``.

        Raises:
            ValueError: If the feature dimension does not match the fit.
        """
        check_is_fitted(self, "reference_")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"X must have shape (n_trials, n_features), got {arr.shape}."
            )
        n = matrix_dimension(arr.shape[1])
        self._check_channel_count(n)
        return untangent_space(arr, self.reference_, metric=self.metric, validate=False)

    def _check_channel_count(self, n_channels: int) -> None:
        """Verify the channel count matches the fitted reference.

        Args:
            n_channels: Channel count of the incoming data.

        Raises:
            ValueError: On mismatch. This most often means two datasets with
                different montages were combined, or a channel-selection step
                ran on one side of the split only.
        """
        if n_channels != self.n_channels_:
            raise ValueError(
                f"Expected {self.n_channels_} channels to match the reference "
                f"point learned during fit, got {n_channels}. A mismatch here "
                f"usually means datasets with different montages were mixed, "
                f"or channel selection was applied inconsistently across the "
                f"train/test split."
            )

    def get_feature_names_out(
        self,
        input_features: Any = None,  # noqa: ARG002
    ) -> NDArray[np.object_]:
        """Names for the output features, as ``ts_i_j`` over the upper triangle.

        Makes a fitted pipeline's coefficients interpretable: a large weight
        on ``ts_3_7`` points at the covariance between channels 3 and 7, which
        is a claim a neuroscientist can evaluate.

        Args:
            input_features: Ignored, present for API compatibility.

        Returns:
            Array of feature names of length ``n_features_out_``.
        """
        check_is_fitted(self, "reference_")
        rows, cols = np.triu_indices(self.n_channels_)
        return np.array(
            [f"ts_{i}_{j}" for i, j in zip(rows, cols, strict=True)], dtype=object
        )

    # ----------------------------------------------------------------- #
    # Scikit-learn tags
    #
    # Declaring three-dimensional input is what stops scikit-learn's own
    # validation helpers from trying to flatten covariance matrices into rows.
    # Both spellings are provided: __sklearn_tags__ for 1.6 and later,
    # _more_tags for earlier versions. Supporting both keeps the framework
    # working across the scikit-learn versions a five-year project will span.
    # ----------------------------------------------------------------- #

    def __sklearn_tags__(self):  # noqa: D105
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = False
        return tags

    def _more_tags(self) -> dict[str, Any]:
        return {"X_types": ["3darray"], "requires_y": False}
