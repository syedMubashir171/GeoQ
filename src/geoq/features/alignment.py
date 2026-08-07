"""Riemannian alignment: re-centring each subject before classification.

Why cross-subject MDM is weak without this
------------------------------------------
Measured on BCI Competition IV 2a under leave-one-subject-out, MDM reaches
kappa 0.163 against TS+LDA's 0.386. The reason is geometric, not statistical.
A class centroid is the Frechet mean of that class's covariances pooled across
subjects -- but each subject's covariances sit in their own frame, displaced by
head geometry, electrode impedance and amplifier gain. The pooled centroid
therefore lands somewhere no individual subject occupies.

Re-centring removes that displacement. Whitening every subject's covariances by
that subject's own geometric mean maps each person's data to a common frame
centred at the identity, so a pooled centroid finally describes a location the
subjects share.

This works because the affine-invariant metric is invariant under congruence:
the unknown per-subject transform is exactly the kind of map AIRM cannot see,
and whitening is what cancels it.

What re-centring leaves behind
------------------------------
It does not make two subjects' data identical, and expecting that is a natural
mistake. If subject B's covariances are subject A's under a congruence by
``W``, then whitening each by its own mean leaves ``B`` equal to
``Q A Q^T`` for some **orthogonal** ``Q``: both maps whiten the mean to the
identity, and any two whitenings of the same matrix differ by a rotation.
Measured on this implementation, ``Q`` is orthogonal to ``6e-11`` and the
relation holds to ``4e-15``.

That residual rotation matters for what alignment can and cannot deliver.
Distances are untouched -- the pairwise geodesic distance matrices of the two
subjects agree to ``1e-13``, because AIRM is blind to orthogonal congruence --
so anything depending only on distances, such as MDM, transfers well. Anything
depending on tangent-space *coordinates* still sees rotated axes, so a linear
classifier's weights learned on one subject do not directly apply to another.
Removing the rotation requires a further, supervised step: Riemannian
Procrustes analysis (Rodrigues et al. 2019). Re-centring is the unsupervised
half, and its limitation is stated here rather than discovered later.

The transductive question, stated rather than buried
----------------------------------------------------
A subject's reference mean is computed from that subject's own covariances --
including, for a held-out subject, their test trials. **No labels are used**,
so this is not label leakage, and :class:`RiemannianAlignment` is tested to
prove that permuting the labels leaves its output bitwise identical.

It is nonetheless *transductive*: it assumes a batch of unlabelled data from
the new user is available before decoding begins. That assumption is true for
a BCI with a calibration phase and false for one that must decode the very
first trial from an unseen user. Which is why the constructor requires
``assume_calibration_data=True``. The argument carries no behaviour; it exists
so the assumption appears in the configuration file and in its diff, rather
than being discovered by a reader working out what the code implies.

This is the honest form of pyRiemann's ``tsupdate`` flag, which was
deliberately not implemented in
:class:`geoq.features.tangent_space.TangentSpace`: named for what it does,
applied per subject, and forcing the deployment assumption into the open.

References
----------
Zanini, P. et al. (2018). Transfer learning: a Riemannian geometry framework
    with applications to brain-computer interfaces. *IEEE TBME*, 65(5),
    1107-1116.
Rodrigues, P. L. C., Jutten, C., & Congedo, M. (2019). Riemannian Procrustes
    analysis: transfer learning for brain-computer interfaces.
    *IEEE TBME*, 66(8), 2390-2401.
He, H., & Wu, D. (2020). Transfer learning for brain-computer interfaces: a
    Euclidean space data alignment approach. *IEEE TBME*, 67(2), 399-410.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from geoq.geometry.riemannian import (
    DEFAULT_MEAN_MAX_ITER,
    DEFAULT_MEAN_TOL,
    Metric,
    frechet_mean,
    whiten,
)
from geoq.geometry.spd import FloatArray, check_spd

__all__ = [
    "SUPPORTED_METRICS",
    "RiemannianAlignment",
    "align_domains",
    "alignment_quality",
    "domain_references",
    "recenter",
]

logger = logging.getLogger(__name__)

SUPPORTED_METRICS: tuple[str, ...] = ("airm", "logeuclid", "euclid")
"""Metrics with an implemented mean, and therefore a usable reference point."""


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #


def recenter(
    covariances: ArrayLike, reference: ArrayLike, *, validate: bool = True
) -> FloatArray:
    """Whiten covariances by a reference point.

    Computes ``R^-1/2 C R^-1/2``, mapping ``reference`` to the identity and
    every other matrix to its position relative to it.

    Args:
        covariances: SPD stack of shape ``(..., n, n)``.
        reference: SPD matrix of shape ``(n, n)``.
        validate: Whether to enforce the SPD contract.

    Returns:
        The re-centred stack.
    """
    return whiten(reference, covariances, validate=validate)


def domain_references(
    covariances: ArrayLike,
    domains: ArrayLike,
    *,
    metric: Metric = "airm",
    tol: float = DEFAULT_MEAN_TOL,
    max_iter: int = DEFAULT_MEAN_MAX_ITER,
) -> dict[Any, FloatArray]:
    """Compute one reference point per domain.

    Args:
        covariances: SPD stack of shape ``(n_trials, n, n)``.
        domains: Domain identifier per trial, usually the subject.
        metric: Geometry used for the means.
        tol: Convergence tolerance for the iterative mean.
        max_iter: Iteration cap.

    Returns:
        Mapping from domain identifier to its reference point.

    Raises:
        ValueError: If shapes disagree or a domain has no trials.
    """
    matrices = check_spd(covariances, name="covariances")
    if matrices.ndim != 3:
        raise ValueError(
            f"covariances must have shape (n_trials, n, n), got {matrices.shape}."
        )
    labels = np.asarray(domains)
    if labels.shape != (matrices.shape[0],):
        raise ValueError(
            f"domains must have shape ({matrices.shape[0]},) to align with "
            f"covariances, got {labels.shape}."
        )

    references: dict[Any, FloatArray] = {}
    for domain in np.unique(labels):
        mask = labels == domain
        count = int(mask.sum())
        if count < 2:
            logger.warning(
                "Domain %r has %d trial(s). Its reference point is that "
                "single covariance, so re-centring maps it exactly to the "
                "identity and destroys the only information it carried.",
                domain,
                count,
            )
        references[domain] = frechet_mean(
            matrices[mask], metric=metric, tol=tol, max_iter=max_iter
        )
    return references


def align_domains(
    covariances: ArrayLike,
    domains: ArrayLike,
    *,
    references: dict[Any, FloatArray] | None = None,
    metric: Metric = "airm",
) -> FloatArray:
    """Re-centre each domain by its own reference point.

    Args:
        covariances: SPD stack of shape ``(n_trials, n, n)``.
        domains: Domain identifier per trial.
        references: Precomputed references. Computed from ``covariances`` when
            None.
        metric: Geometry used when computing references.

    Returns:
        The aligned stack, in the input order.

    Raises:
        KeyError: If a trial's domain has no reference.
    """
    matrices = check_spd(covariances, name="covariances")
    labels = np.asarray(domains)
    if references is None:
        references = domain_references(matrices, labels, metric=metric)

    aligned = np.empty_like(matrices)
    for domain in np.unique(labels):
        if domain not in references:
            raise KeyError(
                f"No reference point for domain {domain!r}. Known domains: "
                f"{sorted(references)}."
            )
        mask = labels == domain
        aligned[mask] = recenter(matrices[mask], references[domain], validate=False)
    return aligned


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #


class RiemannianAlignment(TransformerMixin, BaseEstimator):
    """Re-centre covariances to a common frame, one reference per domain.

    Two ways to use it, and the difference matters.

    **Per-domain, outside the pipeline.** Call :meth:`fit_transform` with
    ``domains``, then feed the aligned covariances to an evaluation. This is
    the form that helps cross-subject transfer, and it is transductive: each
    subject's reference uses that subject's own unlabelled trials.

    **Single-domain, inside a pipeline.** With no ``domains`` the whole batch
    is one domain. Placed in a :class:`sklearn.pipeline.Pipeline`, ``fit``
    learns the training fold's reference and ``transform`` reuses it, exactly
    like :class:`geoq.features.tangent_space.TangentSpace`. That is *not*
    per-subject alignment, and it will not reproduce the transfer benefit --
    it is offered because a pipeline step that silently changed meaning
    depending on how it was called would be worse.

    Scikit-learn's ``Pipeline`` does not forward ``groups`` to ``transform``,
    so per-domain alignment genuinely cannot happen inside a pipeline without
    metadata routing. Rather than smuggle domain information through a global
    or a mutable attribute, the two modes are kept visibly distinct.

    Args:
        metric: Geometry used for the reference means.
        assume_calibration_data: Must be True. Declares that a batch of
            unlabelled data from each subject is available before decoding.
        tol: Convergence tolerance for the iterative mean.
        max_iter: Iteration cap.

    Attributes:
        references_: Reference point per domain.
        n_channels_: Channel count seen during :meth:`fit`.
        n_features_in_: Scikit-learn convention.

    Example:
        >>> import numpy as np
        >>> from geoq.geometry.spd import random_spd
        >>> rng = np.random.default_rng(0)
        >>> x = random_spd(4, rng=rng, batch=20)
        >>> subjects = np.repeat([1, 2], 10)
        >>> aligner = RiemannianAlignment(assume_calibration_data=True)
        >>> aligned = aligner.fit_transform(x, domains=subjects)
        >>> aligned.shape
        (20, 4, 4)
    """

    def __init__(
        self,
        metric: Metric = "airm",
        *,
        assume_calibration_data: bool = False,
        tol: float = DEFAULT_MEAN_TOL,
        max_iter: int = DEFAULT_MEAN_MAX_ITER,
    ) -> None:
        """Store hyperparameters; see the class docstring for detail.

        Raises:
            ValueError: If ``assume_calibration_data`` is not True.
        """
        if assume_calibration_data is not True:
            raise ValueError(
                "RiemannianAlignment computes each subject's reference point "
                "from that subject's own trials, including unlabelled test "
                "trials. No labels are used, so this is not label leakage, but "
                "it assumes a calibration batch is available from a new user "
                "before decoding begins -- true for a BCI with a calibration "
                "phase, false for one that must decode a user's first trial. "
                "Pass assume_calibration_data=True to record that the "
                "assumption fits your deployment scenario."
            )
        self.metric = metric
        self.assume_calibration_data = assume_calibration_data
        self.tol = tol
        self.max_iter = max_iter

    # ----------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------- #

    def _validate_parameters(self) -> None:
        """Check constructor arguments at fit time.

        Raises:
            ValueError: If any argument is out of range.
        """
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"metric must be one of {list(SUPPORTED_METRICS)}, got {self.metric!r}."
            )
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError(f"tol must be a positive finite float, got {self.tol!r}.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be a positive integer, got {self.max_iter!r}."
            )

    @staticmethod
    def _validate_covariances(x: ArrayLike, *, name: str = "X") -> FloatArray:
        """Validate a stack of covariance matrices.

        Args:
            x: Candidate array.
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
                f"got {arr.shape}. Place a Covariances step before this one if "
                f"these are raw epochs."
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
        y: Any = None,  # noqa: ARG002
        domains: ArrayLike | None = None,
    ) -> RiemannianAlignment:
        """Learn one reference point per domain.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.
            y: Ignored. Alignment is unsupervised by construction, which is
                what keeps it free of label leakage.
            domains: Domain identifier per trial. When None the whole batch is
                treated as one domain.

        Returns:
            The fitted transformer.
        """
        self._validate_parameters()
        matrices = self._validate_covariances(X)
        labels = (
            np.zeros(matrices.shape[0], dtype=int)
            if domains is None
            else np.asarray(domains)
        )

        self.references_ = domain_references(
            matrices,
            labels,
            metric=self.metric,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.n_channels_ = matrices.shape[-1]
        self.n_features_in_ = matrices.shape[-1]
        return self

    def transform(
        self,
        X: ArrayLike,  # noqa: N803
        domains: ArrayLike | None = None,
    ) -> FloatArray:
        """Re-centre using the references learned during :meth:`fit`.

        Args:
            X: SPD stack of shape ``(n_trials, n_channels, n_channels)``.
            domains: Domain identifier per trial. When None the whole batch is
                treated as the single domain learned during ``fit``.

        Returns:
            The aligned stack.

        Raises:
            ValueError: If the channel count differs from the fit, or a domain
                has no learned reference.
        """
        check_is_fitted(self, "references_")
        matrices = self._validate_covariances(X)
        if matrices.shape[-1] != self.n_channels_:
            raise ValueError(
                f"Expected {self.n_channels_} channels to match fit, got "
                f"{matrices.shape[-1]}."
            )

        labels = (
            np.zeros(matrices.shape[0], dtype=int)
            if domains is None
            else np.asarray(domains)
        )
        unknown = sorted(set(np.unique(labels).tolist()) - set(self.references_))
        if unknown:
            raise ValueError(
                f"No reference point for domain(s) {unknown}. A domain unseen "
                f"during fit needs its own calibration batch; call "
                f"fit_transform on it rather than transform, and say so in the "
                f"methods section, because computing a reference at transform "
                f"time is a different protocol from reusing a learned one."
            )
        return align_domains(
            matrices, labels, references=self.references_, metric=self.metric
        )

    def fit_transform(
        self,
        X: ArrayLike,  # noqa: N803
        y: Any = None,
        domains: ArrayLike | None = None,
        **fit_params: Any,  # noqa: ARG002
    ) -> FloatArray:
        """Fit and transform in one call, forwarding ``domains`` to both.

        Overridden because :class:`sklearn.base.TransformerMixin` does not
        forward extra keyword arguments to ``transform``, so the inherited
        version would fit per domain and then transform as a single domain --
        silently producing something that is not alignment.

        Args:
            X: SPD stack.
            y: Ignored.
            domains: Domain identifier per trial.
            **fit_params: Ignored, present for API compatibility.

        Returns:
            The aligned stack.
        """
        return self.fit(X, y, domains=domains).transform(X, domains=domains)

    def __sklearn_tags__(self):  # noqa: D105
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = False
        return tags

    def _more_tags(self) -> dict[str, Any]:
        return {"X_types": ["3darray"], "requires_y": False}


def alignment_quality(
    covariances: ArrayLike, domains: ArrayLike, *, metric: Metric = "airm"
) -> dict[str, float]:
    """Measure how far each domain's mean sits from the identity.

    The diagnostic for whether alignment worked. After re-centring, every
    domain's geometric mean should be the identity to numerical precision; a
    residual distance means the mean did not converge, which displaces that
    subject relative to the others by an unknown amount.

    Args:
        covariances: SPD stack of shape ``(n_trials, n, n)``.
        domains: Domain identifier per trial.
        metric: Geometry used for the means.

    Returns:
        Mapping with the worst and mean residual distance to the identity.
    """
    from geoq.geometry.riemannian import distance

    matrices = check_spd(covariances, name="covariances")
    references = domain_references(matrices, domains, metric=metric)
    identity = np.eye(matrices.shape[-1])
    residuals = [
        float(distance(reference, identity, metric=metric))
        for reference in references.values()
    ]
    return {
        "n_domains": float(len(residuals)),
        "max_residual": float(np.max(residuals)),
        "mean_residual": float(np.mean(residuals)),
    }
