"""Spatial covariance estimation from raw EEG epochs.

This is the entry point of every pipeline in the framework: it turns epochs of
shape ``(n_trials, n_channels, n_times)`` into the SPD matrices that the
Riemannian geometry operates on.

The conditioning problem, stated plainly
----------------------------------------
The sample covariance of an epoch with ``T`` time samples and ``N`` channels
has rank at most ``min(N, T - 1)``. When ``T <= N`` the result is singular and
is not an SPD matrix at all -- the manifold is undefined there, and every
subsequent operation is meaningless rather than merely inaccurate.

Full rank is not sufficient either. Even at ``T = 2N`` the condition number is
routinely above ``1e6``, and the geometry layer's measurements put the
relative error of an AIRM distance at roughly ``eps * kappa ** 2`` -- because
the affine-invariant metric whitens, and whitening inverts. At ``kappa = 1e6``
a geodesic distance retains about four significant digits; at ``1e8``, two.

This module therefore does three things that a bare ``np.cov`` does not:

* refuses outright when ``T <= N``, with a message naming the cause;
* offers shrinkage estimators, which are the standard remedy and reduce the
  condition number by orders of magnitude at a small cost in bias;
* audits the conditioning of every output and reports the distribution, so
  that a badly conditioned dataset is a logged finding rather than a silent
  degradation discovered months later.

Leakage
-------
Unlike :class:`geoq.features.tangent_space.TangentSpace`, this transformer is
free of leakage risk by construction. Each covariance is computed from a
single epoch, and shrinkage intensity -- for Ledoit-Wolf and OAS alike -- is
estimated from that same epoch alone. No statistic is pooled across trials, so
``fit`` learns nothing that ``transform`` could leak. The estimator interface
is kept anyway, because a step that silently skipped ``fit`` would be the one
place in a pipeline where the reader could not verify that claim.

References
----------
Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
    large-dimensional covariance matrices. *J. Multivariate Analysis*, 88(2),
    365-411.
Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage
    algorithms for MMSE covariance estimation. *IEEE TSP*, 58(10), 5016-5029.
Congedo, M., Barachant, A., & Bhatia, R. (2017). Riemannian geometry for
    EEG-based brain-computer interfaces: a primer and a review.
    *Brain-Computer Interfaces*, 4(3), 155-174.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import OAS, LedoitWolf, empirical_covariance
from sklearn.utils.validation import check_is_fitted

from geoq.geometry.spd import (
    FloatArray,
    check_spd,
    condition_number,
    is_spd,
    shrink_toward_identity,
    symmetrize,
)

__all__ = ["CONDITION_WARNING_THRESHOLD", "Covariances", "Estimator"]

logger = logging.getLogger(__name__)

Estimator = Literal["scm", "ledoit_wolf", "oas"]
"""Supported covariance estimators, as they appear in YAML configuration."""

SUPPORTED_ESTIMATORS: tuple[str, ...] = ("scm", "ledoit_wolf", "oas")

CONDITION_WARNING_THRESHOLD: float = 1e6
"""Condition number above which the geometry loses meaningful precision.

Chosen from measurement, not convention. AIRM computations carry a relative
error of about ``eps * kappa ** 2``; at ``kappa = 1e6`` that is roughly
``5e-5``, the point past which distances stop being trustworthy to the
precision a published result implies.
"""

MIN_SAMPLES_PER_CHANNEL: float = 2.0
"""Ratio ``n_times / n_channels`` below which a warning is emitted.

Full rank requires only ``T > N``. Usable conditioning requires considerably
more; the literature's rule of thumb is ``T >= 10 N``, and below ``2 N`` the
sample covariance is essentially noise dressed as structure.
"""


class Covariances(TransformerMixin, BaseEstimator):
    """Estimate one spatial covariance matrix per epoch.

    Args:
        estimator: ``"scm"`` for the plain sample covariance, ``"ledoit_wolf"``
            or ``"oas"`` for shrinkage estimators. The default is ``"scm"``
            because it is what published Riemannian BCI results use, and
            reproducing them is a prerequisite for departing from them. Switch
            to a shrinkage estimator when the conditioning audit says to.
        assume_centered: If True, skip mean removal. Correct only when the
            epochs are already zero-mean, which band-pass filtering above
            roughly 0.5 Hz makes approximately true. Left False by default
            because an unremoved mean inflates the covariance by a rank-one
            term that is not neural.
        shrinkage: Optional explicit trace-preserving shrinkage toward the
            identity, applied after estimation. Independent of the estimator
            choice, and intended as a documented last resort for datasets that
            fail the conditioning audit even under Ledoit-Wolf.
        audit_conditioning: Whether to log the condition-number distribution of
            each transformed batch.

    Attributes:
        n_channels_: Channel count seen during :meth:`fit`.
        n_times_: Sample count per epoch seen during :meth:`fit`.
        n_features_in_: Scikit-learn convention, set to ``n_channels``.
        condition_numbers_: Condition numbers of the most recent
            :meth:`transform` output, for reporting.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> epochs = rng.standard_normal((20, 8, 256))
        >>> covariances = Covariances().fit_transform(epochs)
        >>> covariances.shape
        (20, 8, 8)
    """

    def __init__(
        self,
        estimator: Estimator = "scm",
        *,
        assume_centered: bool = False,
        shrinkage: float | None = None,
        audit_conditioning: bool = True,
    ) -> None:
        """Store hyperparameters unmodified; see the class docstring for detail."""
        self.estimator = estimator
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage
        self.audit_conditioning = audit_conditioning

    # ----------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------- #

    def _validate_parameters(self) -> None:
        """Check constructor arguments at fit time.

        Raises:
            ValueError: If any argument is outside its valid range.
        """
        if self.estimator not in SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {list(SUPPORTED_ESTIMATORS)}, got "
                f"{self.estimator!r}."
            )
        if self.shrinkage is not None and not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError(
                f"shrinkage must lie in [0, 1] or be None, got {self.shrinkage!r}."
            )

    def _validate_epochs(self, x: ArrayLike, *, name: str = "X") -> FloatArray:
        """Validate raw epochs and check that covariances can exist at all.

        Args:
            x: Candidate array of shape ``(n_trials, n_channels, n_times)``.
            name: Symbol name used in error messages.

        Returns:
            The validated array.

        Raises:
            ValueError: If the shape is wrong, the array is empty, it contains
                non-finite values, a channel is flat, or there are too few
                time samples for a full-rank covariance.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(
                f"{name} must have shape (n_trials, n_channels, n_times), got "
                f"{arr.shape}. Epochs are expected channels-first, matching "
                f"MNE's convention; a (n_trials, n_times, n_channels) array "
                f"needs transposing before this step."
            )
        n_trials, n_channels, n_times = arr.shape
        if n_trials == 0:
            raise ValueError(f"{name} contains no trials.")
        if n_channels == 0:
            raise ValueError(f"{name} contains no channels.")

        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            trial = int(np.argwhere(~np.isfinite(arr))[0][0])
            raise ValueError(
                f"{name} contains {n_bad} non-finite sample(s), first in trial "
                f"{trial}. This usually means an epoch overlapped a recording "
                f"gap or an interpolation step failed."
            )

        # Rank is the hard constraint: with T <= N the covariance is singular,
        # so the SPD manifold is not merely ill-conditioned but undefined.
        if n_times <= n_channels:
            #  A (n_trials, n_times, n_channels) array has the correct rank and
            #  the wrong meaning. It usually presents here, because time
            #  samples vastly outnumber channels, so the transposed array looks
            #  like a huge montage recorded for a few samples. Naming that
            #  possibility turns a confusing rank error into a one-line fix.
            transposed_hint = (
                f" The axis order also looks suspicious: {n_channels} "
                f"'channels' and {n_times} 'times' suggests this array is "
                f"(n_trials, n_times, n_channels) and needs transposing to "
                f"channels-first."
                if n_channels > 4 * max(n_times, 1)
                else ""
            )
            raise ValueError(
                f"{name} has n_times={n_times} <= n_channels={n_channels}, so "
                f"every sample covariance is singular and no SPD matrix "
                f"exists. Lengthen the epoch, reduce the channel count, or "
                f"raise the sampling rate. Shrinkage cannot repair this: it "
                f"would fabricate the missing rank rather than estimate it."
                f"{transposed_hint}"
            )

        flat = np.ptp(arr, axis=-1) == 0.0
        if flat.any():
            trial, channel = (int(i) for i in np.argwhere(flat)[0])
            raise ValueError(
                f"{name} has {int(flat.sum())} flat channel(s) with zero "
                f"variance, first at trial {trial}, channel {channel}. A dead "
                f"or disconnected electrode produces a singular covariance; "
                f"drop the channel or interpolate it before this step."
            )

        return arr

    def _warn_on_sample_ratio(self, n_channels: int, n_times: int) -> None:
        """Warn when epochs are short relative to the channel count.

        Args:
            n_channels: Channel count.
            n_times: Samples per epoch.
        """
        ratio = n_times / n_channels
        if ratio < MIN_SAMPLES_PER_CHANNEL:
            logger.warning(
                "Covariances: n_times/n_channels = %.1f is below %.1f. The "
                "sample covariance is technically full rank but severely "
                "ill-conditioned at this ratio, and AIRM distances computed "
                "from it will carry a relative error near eps * kappa ** 2. "
                "Prefer estimator='ledoit_wolf' or 'oas', or lengthen the "
                "epoch.",
                ratio,
                MIN_SAMPLES_PER_CHANNEL,
            )

    # ----------------------------------------------------------------- #
    # Estimation
    # ----------------------------------------------------------------- #

    def _estimate_one(self, epoch: FloatArray) -> FloatArray:
        """Estimate the covariance of a single epoch.

        Args:
            epoch: Array of shape ``(n_channels, n_times)``.

        Returns:
            Covariance of shape ``(n_channels, n_channels)``.
        """
        # scikit-learn's covariance estimators expect (n_samples, n_features),
        # the transpose of the channels-first convention used throughout EEG.
        samples = epoch.T

        if self.estimator == "scm":
            #  scikit-learn's empirical_covariance normalises by n_times, not
            #  by n_times - 1. pyRiemann re-implements the same function, so
            #  using it directly is what makes the parity test exact.
            #
            #  The choice is scientifically immaterial for this framework: a
            #  factor applied to every covariance cancels in the whitened
            #  product P^-1/2 X P^-1/2, leaving AIRM distances and tangent
            #  features unchanged. It matters only for reproducing published
            #  numbers, and reproducing them is a prerequisite for departing
            #  from them.
            return symmetrize(
                empirical_covariance(samples, assume_centered=self.assume_centered)
            )

        estimator_class = LedoitWolf if self.estimator == "ledoit_wolf" else OAS
        fitted = estimator_class(assume_centered=self.assume_centered).fit(samples)
        return symmetrize(fitted.covariance_)

    def _audit(self, covariances: FloatArray) -> FloatArray:
        """Compute and optionally log the conditioning of a batch.

        A dataset whose covariances are poorly conditioned is a finding about
        the data, and it belongs in the run's log and ultimately in the
        methods section -- not in an unexplained accuracy gap.

        Args:
            covariances: SPD stack of shape ``(n_trials, n, n)``.

        Returns:
            Condition numbers of shape ``(n_trials,)``.
        """
        kappa = condition_number(covariances, name="covariances")
        if not self.audit_conditioning:
            return kappa

        above = int(np.sum(kappa > CONDITION_WARNING_THRESHOLD))
        median = float(np.median(kappa))
        worst = float(np.max(kappa))
        if above:
            logger.warning(
                "Covariances: %d/%d epochs exceed a condition number of %.0e "
                "(median %.2e, worst %.2e). AIRM distances carry a relative "
                "error near eps * kappa ** 2, so at this conditioning they "
                "retain roughly %d significant digits. Consider "
                "estimator='ledoit_wolf' or an explicit shrinkage.",
                above,
                kappa.shape[0],
                CONDITION_WARNING_THRESHOLD,
                median,
                worst,
                max(0, int(16 - 2 * np.log10(worst))),
            )
        else:
            logger.debug(
                "Covariances: conditioning healthy (median %.2e, worst %.2e).",
                median,
                worst,
            )
        return kappa

    # ----------------------------------------------------------------- #
    # Estimator interface
    # ----------------------------------------------------------------- #

    def fit(self, X: ArrayLike, y: Any = None) -> Covariances:  # noqa: ARG002, N803
        """Validate input and record its shape.

        Nothing is learned from the data: each covariance depends only on its
        own epoch. ``fit`` exists so that the transformer composes into a
        pipeline and so that shape validation happens once, at a point where
        the error message can name the training set.

        Args:
            X: Epochs of shape ``(n_trials, n_channels, n_times)``.
            y: Ignored, present for API compatibility.

        Returns:
            The fitted transformer.
        """
        self._validate_parameters()
        epochs = self._validate_epochs(X)

        self.n_channels_ = epochs.shape[1]
        self.n_times_ = epochs.shape[2]
        self.n_features_in_ = epochs.shape[1]
        self._warn_on_sample_ratio(self.n_channels_, self.n_times_)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:  # noqa: N803
        """Estimate one covariance matrix per epoch.

        Args:
            X: Epochs of shape ``(n_trials, n_channels, n_times)``.

        Returns:
            SPD stack of shape ``(n_trials, n_channels, n_channels)``.

        Raises:
            ValueError: If the channel count differs from that seen in
                :meth:`fit`.
            NotPositiveDefiniteError: If an estimated covariance is singular
                despite passing the rank check, which indicates linearly
                dependent channels -- most often an average reference or an
                ICA step that reduced rank without dropping a channel.
        """
        check_is_fitted(self, "n_channels_")
        epochs = self._validate_epochs(X)

        if epochs.shape[1] != self.n_channels_:
            raise ValueError(
                f"Expected {self.n_channels_} channels to match fit, got "
                f"{epochs.shape[1]}. Mixing montages produces matrices that "
                f"live on different manifolds and cannot be compared."
            )

        covariances = np.stack([self._estimate_one(epoch) for epoch in epochs])

        if self.shrinkage is not None:
            covariances = shrink_toward_identity(covariances, alpha=self.shrinkage)

        singular = ~is_spd(covariances)
        if np.any(singular):
            bad = int(np.argmax(singular))
            raise self._singular_error(
                bad, int(singular.sum()), epochs.shape, covariances[bad]
            )

        self.condition_numbers_ = self._audit(covariances)
        return check_spd(covariances, name="covariances")

    def _singular_error(
        self,
        first_bad: int,
        n_bad: int,
        shape: tuple[int, ...],
        offending: FloatArray,
    ) -> Exception:
        """Build the error raised when an estimated covariance is not SPD.

        Args:
            first_bad: Index of the first offending trial.
            n_bad: Number of offending trials.
            shape: Shape of the input epochs.
            offending: The first covariance that failed the SPD check, used to
                distinguish exact rank deficiency from ill-conditioning.

        Returns:
            The exception to raise.
        """
        remedy = (
            "Try estimator='ledoit_wolf', which is guaranteed positive definite."
            if self.estimator == "scm"
            else "Increase the explicit shrinkage, or lengthen the epoch."
        )

        #  Two very different causes produce a matrix that fails the SPD check,
        #  and the remedy differs. An exactly rank-deficient covariance has
        #  eigenvalues at the level of round-off: some direction carries no
        #  variance at all, which is a preprocessing artefact. A merely
        #  ill-conditioned one has a small but genuine smallest eigenvalue and
        #  is a sampling problem. Reporting the first explanation for the
        #  second sends the reader hunting for an average reference that was
        #  never applied.
        eigenvalues = np.linalg.eigvalsh(offending)
        smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
        exactly_deficient = smallest <= 1e-14 * largest

        if exactly_deficient:
            cause = (
                "At least one direction carries no variance at all "
                f"(lambda_min = {smallest:.3e} against lambda_max = "
                f"{largest:.3e}), so the channels are exactly linearly "
                "dependent. The usual causes are an average reference, which "
                "removes exactly one degree of freedom, or an ICA cleaning "
                "step that reduced rank without dropping a channel."
            )
        else:
            cause = (
                f"The smallest eigenvalue is genuinely positive "
                f"({smallest:.3e}) but the condition number is "
                f"{largest / smallest:.2e}, past the framework's limit of "
                f"1e12. This is a sampling problem, not a rank defect: "
                f"n_times={shape[2]} is too close to n_channels={shape[1]} "
                f"for a well-conditioned estimate."
            )

        return ValueError(
            f"{n_bad}/{shape[0]} estimated covariances are not positive "
            f"definite, first at trial {first_bad}. {cause} {remedy}"
        )

    def __sklearn_tags__(self):  # noqa: D105
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = False
        return tags

    def _more_tags(self) -> dict[str, Any]:
        return {"X_types": ["3darray"], "requires_y": False}
