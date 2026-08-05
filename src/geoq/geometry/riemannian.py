"""Riemannian geometry of the SPD manifold.

This module turns the spectral primitives of :mod:`geoq.geometry.spd` into
geometry: distances, the Exp and Log maps at an arbitrary reference point,
geodesics, parallel transport, and Frechet means.

Why the metric is a parameter, not a hardcoded choice
-----------------------------------------------------
Every function that depends on a notion of distance takes a ``metric``
argument resolved through :data:`METRICS`. The affine-invariant metric (AIRM)
is the default and the scientifically principled choice, but it is not the
only one, and the difference matters to this thesis in three concrete ways:

* **Cost.** AIRM has no closed-form mean; the Frechet mean requires iterative
  optimisation. Log-Euclidean has a closed form and is orders of magnitude
  faster. Under leave-one-subject-out with nested inner folds, that gap is the
  difference between a run that finishes overnight and one that does not.
* **Invariance.** Only AIRM is affine-invariant, which is what makes
  Riemannian re-centring work across subjects and sessions. Log-Euclidean is
  invariant only to orthogonal congruence. Any claim about cross-subject
  transfer must state which invariance it relies on.
* **Falsifiability.** A quantum encoding that improves results under one
  metric and not another has not demonstrated a quantum effect; it has
  demonstrated sensitivity to a preprocessing choice. Making the metric a
  configured factor is what allows that confound to be measured rather than
  assumed away.

Conventions
-----------
All functions accept stacks of shape ``(..., n, n)`` and broadcast over the
leading dimensions. Reference points are validated as SPD; tangent vectors are
validated as symmetric but may be indefinite, since the tangent space at any
point is the full space of symmetric matrices.

References
----------
Pennec, X., Fillard, P., & Ayache, N. (2006). A Riemannian framework for
    tensor computing. *IJCV*, 66(1), 41-66.
Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). Multiclass
    brain-computer interface classification by Riemannian geometry.
    *IEEE TBME*, 59(4), 920-928.
Congedo, M., Barachant, A., & Bhatia, R. (2017). Riemannian geometry for
    EEG-based brain-computer interfaces: a primer and a review.
    *Brain-Computer Interfaces*, 4(3), 155-174.
Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2007). Geometric means in
    a novel vector space structure on SPD matrices. *SIMAX*, 29(1), 328-347.
Sra, S. (2012). A new metric on the manifold of kernel matrices with
    application to matrix geometric means. *NeurIPS*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike

from geoq.geometry.spd import (
    FloatArray,
    check_spd,
    check_symmetric,
    condition_number,
    expm_sym,
    invsqrtm_spd,
    logdet_spd,
    logm_spd,
    powm_spd,
    sqrtm_spd,
    symmetrize,
)

__all__ = [
    "DEFAULT_MEAN_MAX_ITER",
    "DEFAULT_MEAN_TOL",
    "METRICS",
    "TOLERANCE_FLOOR_FACTOR",
    "ConvergenceWarning",
    "MeanResult",
    "Metric",
    "distance",
    "distance_airm",
    "distance_euclid",
    "distance_logeuclid",
    "distance_stein",
    "exp_map",
    "frechet_mean",
    "geodesic",
    "log_map",
    "mean_euclid",
    "mean_logeuclid",
    "pairwise_distances",
    "parallel_transport",
    "whiten",
]

logger = logging.getLogger(__name__)

Metric = Literal["airm", "logeuclid", "stein", "euclid"]
"""Names of the supported metrics, as they appear in YAML configuration."""

METRICS: Final[tuple[str, ...]] = ("airm", "logeuclid", "stein", "euclid")

DEFAULT_MEAN_TOL: Final[float] = 1e-10
"""Requested convergence tolerance for the Frechet mean.

Treated as a target, not a guarantee: see :data:`TOLERANCE_FLOOR_FACTOR`.
"""

TOLERANCE_FLOOR_FACTOR: Final[float] = 1.0
"""Multiplier on ``eps * kappa`` giving the achievable tolerance floor.

The Frechet-mean iteration measures its own progress with a criterion built
from Exp and Log maps, both of which whiten and therefore invert. The
criterion consequently cannot be driven below a floor that grows linearly with
the condition number of the input set. Measured on this implementation, the
best achievable criterion is ``2.1e-15`` at ``kappa = 1``, ``4.8e-12`` at
``1e6``, and ``4.9e-10`` at ``1e8``.

A fixed tolerance of ``1e-10`` is therefore unreachable past ``kappa`` of
roughly ``1e7``, and every ill-conditioned fold would report a spurious
non-convergence. On a leave-one-subject-out run that is hundreds of warnings
about a failure that did not occur, which trains the reader to ignore the
warning that eventually matters.

:func:`frechet_mean` therefore raises the requested tolerance to
``TOLERANCE_FLOOR_FACTOR * eps * kappa`` when that is larger, and records the
value actually used. The factor of one leaves roughly 45x margin over the
worst floor measured across ``kappa`` in ``[1, 1e10]``.
"""

DEFAULT_MEAN_MAX_ITER: Final[int] = 100
"""Iteration cap for the Frechet mean."""


class ConvergenceWarning(UserWarning):
    """Raised as a warning when an iterative solver hits its iteration cap."""


# --------------------------------------------------------------------------- #
# Whitening -- the operation underneath every affine-invariant quantity
# --------------------------------------------------------------------------- #


def whiten(
    reference: ArrayLike, target: ArrayLike, *, validate: bool = True
) -> FloatArray:
    """Congruence-transform ``target`` into the frame whitened by ``reference``.

    Computes ``R^-1/2 T R^-1/2``. This single operation is what makes the
    affine-invariant metric affine-invariant: it moves the reference point to
    the identity, so all subsequent geometry happens in a canonical frame.
    Riemannian re-centring for cross-subject transfer is exactly this
    operation with the subject's own mean as the reference.

    Args:
        reference: SPD stack of shape ``(..., n, n)`` defining the frame.
        target: SPD or symmetric stack of shape ``(..., n, n)``.
        validate: Whether to enforce the input contracts.

    Returns:
        The whitened stack, shape ``(..., n, n)``, exactly symmetric.
    """
    whitener = invsqrtm_spd(reference, validate=validate, name="reference")
    target_arr = (
        check_symmetric(target, name="target") if validate else symmetrize(target)
    )
    return symmetrize(whitener @ target_arr @ whitener)


def _whitened_spectrum(
    a: ArrayLike, b: ArrayLike, *, validate: bool = True
) -> FloatArray:
    """Eigenvalues of ``A^-1/2 B A^-1/2``, i.e. the generalised spectrum of ``(B, A)``.

    Every affine-invariant quantity in this module is a function of these
    numbers alone, which is the computational statement of affine invariance.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        b: SPD stack of shape ``(..., n, n)``.
        validate: Whether to enforce the SPD contract on both arguments.

    Returns:
        Ascending eigenvalues of shape ``(..., n)``, strictly positive.
    """
    if validate:
        a = check_spd(a, name="a")
        b = check_spd(b, name="b")
    return np.linalg.eigvalsh(whiten(a, b, validate=False))


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #


def distance_airm(a: ArrayLike, b: ArrayLike, *, validate: bool = True) -> FloatArray:
    r"""Affine-invariant Riemannian distance.

    .. math::
        d(A, B) = \left\| \log\left(A^{-1/2} B A^{-1/2}\right) \right\|_F
                = \sqrt{\sum_i \log^2 \lambda_i}

    where :math:`\lambda_i` are the generalised eigenvalues of ``(B, A)``.

    The implementation sums squared log-eigenvalues rather than forming the
    matrix logarithm and taking its Frobenius norm. The two are mathematically
    identical; the eigenvalue form performs one decomposition instead of two
    and avoids reconstructing a matrix that is immediately collapsed to a
    scalar. On a 22-channel LOSO run this is called millions of times.

    This distance is invariant under ``A -> W A W^T`` for any invertible
    ``W`` -- the property that makes Riemannian alignment work across subjects
    with different head geometry and electrode impedance.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        b: SPD stack of shape ``(..., n, n)``.
        validate: Whether to enforce the SPD contract.

    Returns:
        Distances of shape ``(...)``, non-negative.
    """
    eigenvalues = _whitened_spectrum(a, b, validate=validate)
    return np.sqrt(np.sum(np.log(eigenvalues) ** 2, axis=-1))


def distance_logeuclid(
    a: ArrayLike, b: ArrayLike, *, validate: bool = True
) -> FloatArray:
    r"""Log-Euclidean distance, :math:`\|\log A - \log B\|_F`.

    Flattens the manifold by mapping everything through the matrix logarithm
    once, then measuring in the resulting vector space. Cheaper than AIRM and
    endowed with a closed-form mean, at the cost of full affine invariance:
    it is invariant to orthogonal congruence only.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        b: SPD stack of shape ``(..., n, n)``.
        validate: Whether to enforce the SPD contract.

    Returns:
        Distances of shape ``(...)``.
    """
    difference = logm_spd(a, validate=validate, name="a") - logm_spd(
        b, validate=validate, name="b"
    )
    return np.linalg.norm(difference, axis=(-2, -1))


def distance_stein(a: ArrayLike, b: ArrayLike, *, validate: bool = True) -> FloatArray:
    r"""Stein (S-divergence) distance.

    .. math::
        d(A, B) = \sqrt{\log\det\frac{A + B}{2}
                        - \tfrac{1}{2}\log\det(AB)}

    A symmetric divergence whose square root is a true metric (Sra 2012). It
    requires no eigendecomposition of a whitened product -- only three
    log-determinants -- which makes it the cheapest of the three curved
    options and a useful sanity check: a result that holds under AIRM but
    vanishes under Stein is worth investigating before it is published.

    The radicand is clipped at zero before the square root. It is
    non-negative in exact arithmetic, but for nearly identical matrices it can
    reach a small negative value through cancellation, and an unguarded
    ``sqrt`` would return NaN for a pair of matrices that are merely very
    close.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        b: SPD stack of shape ``(..., n, n)``.
        validate: Whether to enforce the SPD contract.

    Returns:
        Distances of shape ``(...)``.
    """
    if validate:
        a = check_spd(a, name="a")
        b = check_spd(b, name="b")
    else:
        a = symmetrize(a)
        b = symmetrize(b)

    midpoint = 0.5 * (a + b)
    radicand = (
        logdet_spd(midpoint, validate=validate, name="midpoint")
        - 0.5 * logdet_spd(a, validate=False)
        - 0.5 * logdet_spd(b, validate=False)
    )
    return np.sqrt(np.maximum(radicand, 0.0))


def distance_euclid(a: ArrayLike, b: ArrayLike, *, validate: bool = True) -> FloatArray:
    """Euclidean (Frobenius) distance, ignoring the manifold entirely.

    Included as the null geometry. Papers 1 and 4 both need a baseline that
    treats covariance matrices as flat vectors, because "the geometry helps"
    is only a claim if the flat alternative has actually been measured.

    Args:
        a: Symmetric stack of shape ``(..., n, n)``.
        b: Symmetric stack of shape ``(..., n, n)``.
        validate: Whether to enforce symmetry.

    Returns:
        Distances of shape ``(...)``.
    """
    if validate:
        a = check_symmetric(a, name="a")
        b = check_symmetric(b, name="b")
    return np.linalg.norm(
        np.asarray(a, dtype=np.float64) - np.asarray(b), axis=(-2, -1)
    )


_DISTANCE_FUNCTIONS = {
    "airm": distance_airm,
    "logeuclid": distance_logeuclid,
    "stein": distance_stein,
    "euclid": distance_euclid,
}


def _resolve(metric: str, table: dict) -> object:
    """Look up ``metric`` in ``table``, failing loudly on an unknown name.

    Args:
        metric: Metric name from configuration.
        table: Mapping from metric name to implementation.

    Returns:
        The resolved implementation.

    Raises:
        ValueError: If the name is not recognised. A silent fallback to a
            default metric would mean a typo in a YAML file changes the
            geometry of an experiment without changing anything visible in the
            results.
    """
    try:
        return table[metric]
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(table)}."
        ) from None


def distance(
    a: ArrayLike,
    b: ArrayLike,
    *,
    metric: Metric = "airm",
    validate: bool = True,
) -> FloatArray:
    """Distance between SPD matrices under the named metric.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        b: SPD stack of shape ``(..., n, n)``.
        metric: One of :data:`METRICS`.
        validate: Whether to enforce the input contracts.

    Returns:
        Distances of shape ``(...)``.

    Raises:
        ValueError: If ``metric`` is unknown.
    """
    func = _resolve(metric, _DISTANCE_FUNCTIONS)
    return func(a, b, validate=validate)  # type: ignore[operator]


def pairwise_distances(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    metric: Metric = "airm",
    validate: bool = True,
) -> FloatArray:
    """Full distance matrix between two sets of SPD matrices.

    This is the object a kernel method consumes, and the bridge to Block 5.5:
    a quantum kernel matrix is compared against exactly this, computed under
    AIRM, to test whether the encoding preserves manifold geometry.

    When ``y`` is None the matrix is symmetric, and only the upper triangle is
    computed -- halving the work, which matters because this is
    :math:`O(N^2)` in the number of epochs.

    Args:
        x: SPD stack of shape ``(m, n, n)``.
        y: Optional SPD stack of shape ``(k, n, n)``. Defaults to ``x``.
        metric: One of :data:`METRICS`.
        validate: Whether to enforce the input contracts.

    Returns:
        Distance matrix of shape ``(m, k)``, or ``(m, m)`` when ``y`` is None.
    """
    x_arr = check_spd(x, name="x") if validate else symmetrize(x)
    if x_arr.ndim != 3:
        raise ValueError(f"x must have shape (m, n, n), got {x_arr.shape}.")

    if y is None:
        m = x_arr.shape[0]
        result = np.zeros((m, m), dtype=np.float64)
        rows, cols = np.triu_indices(m, k=1)
        result[rows, cols] = distance(
            x_arr[rows], x_arr[cols], metric=metric, validate=False
        )
        # The diagonal is exactly zero by construction rather than by
        # computation: d(A, A) evaluates to ~1e-8 for ill-conditioned A
        # through cancellation, and a kernel built on a non-zero diagonal is
        # not positive semi-definite.
        return result + result.T

    y_arr = check_spd(y, name="y") if validate else symmetrize(y)
    if y_arr.ndim != 3:
        raise ValueError(f"y must have shape (k, n, n), got {y_arr.shape}.")
    if y_arr.shape[-1] != x_arr.shape[-1]:
        raise ValueError(
            f"x and y must have the same matrix dimension; got "
            f"{x_arr.shape[-1]} and {y_arr.shape[-1]}."
        )
    return distance(x_arr[:, None], y_arr[None, :], metric=metric, validate=False)


# --------------------------------------------------------------------------- #
# Exp / Log maps and geodesics
# --------------------------------------------------------------------------- #


def log_map(
    reference: ArrayLike, target: ArrayLike, *, validate: bool = True
) -> FloatArray:
    r"""Riemannian logarithm at ``reference``.

    .. math::
        \mathrm{Log}_R(T) = R^{1/2}
            \log\left(R^{-1/2} T R^{-1/2}\right) R^{1/2}

    Maps a point on the manifold to a tangent vector at ``reference``. The
    tangent space is a genuine vector space, which is what allows a linear
    classifier to be applied to curved data -- the entire basis of the
    Tangent Space + LDA baseline.

    Note that the tangent vector is symmetric but generally *indefinite*: it
    is a displacement, not a covariance.

    Args:
        reference: SPD stack of shape ``(..., n, n)``, the base point.
        target: SPD stack of shape ``(..., n, n)``.
        validate: Whether to enforce the SPD contract.

    Returns:
        Symmetric tangent vectors of shape ``(..., n, n)``.
    """
    if validate:
        reference = check_spd(reference, name="reference")
        target = check_spd(target, name="target")
    root = sqrtm_spd(reference, validate=False)
    whitened = whiten(reference, target, validate=False)
    return symmetrize(root @ logm_spd(whitened, validate=False) @ root)


def exp_map(
    reference: ArrayLike, tangent: ArrayLike, *, validate: bool = True
) -> FloatArray:
    r"""Riemannian exponential at ``reference``, the inverse of :func:`log_map`.

    .. math::
        \mathrm{Exp}_R(S) = R^{1/2}
            \exp\left(R^{-1/2} S R^{-1/2}\right) R^{1/2}

    Args:
        reference: SPD stack of shape ``(..., n, n)``, the base point.
        tangent: Symmetric stack of shape ``(..., n, n)``. Not required to be
            positive definite -- the tangent space at any point is the full
            space of symmetric matrices.
        validate: Whether to enforce the input contracts.

    Returns:
        SPD stack of shape ``(..., n, n)``.
    """
    if validate:
        reference = check_spd(reference, name="reference")
        tangent = check_symmetric(tangent, name="tangent")
    root = sqrtm_spd(reference, validate=False)
    whitener = invsqrtm_spd(reference, validate=False)
    inner = symmetrize(whitener @ tangent @ whitener)
    return symmetrize(root @ expm_sym(inner, validate=False) @ root)


def geodesic(
    a: ArrayLike,
    b: ArrayLike,
    t: float | ArrayLike,
    *,
    metric: Metric = "airm",
    validate: bool = True,
) -> FloatArray:
    r"""Point at parameter ``t`` along the geodesic from ``a`` to ``b``.

    For AIRM, :math:`\gamma(t) = A^{1/2}(A^{-1/2} B A^{-1/2})^t A^{1/2}`, with
    ``t=0`` giving ``a``, ``t=1`` giving ``b``, and ``t=0.5`` the geometric
    mean of the pair -- the only case where the Frechet mean has a closed form.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        b: SPD stack of shape ``(..., n, n)``.
        t: Interpolation parameter, scalar or broadcastable to ``(...)``.
            Values outside ``[0, 1]`` extrapolate along the geodesic, which is
            well defined and occasionally useful for data augmentation.
        metric: One of ``"airm"``, ``"logeuclid"``, ``"euclid"``. Stein has no
            geodesic in closed form and is rejected.
        validate: Whether to enforce the SPD contract.

    Returns:
        SPD stack of shape ``(..., n, n)``.

    Raises:
        ValueError: If ``metric`` has no closed-form geodesic.
    """
    if validate:
        a = check_spd(a, name="a")
        b = check_spd(b, name="b")
    t_arr = np.asarray(t, dtype=np.float64)

    if metric == "airm":
        root = sqrtm_spd(a, validate=False)
        whitened = whiten(a, b, validate=False)
        powered = _powm_broadcast(whitened, t_arr)
        return symmetrize(root @ powered @ root)
    if metric == "logeuclid":
        log_a = logm_spd(a, validate=False)
        log_b = logm_spd(b, validate=False)
        return expm_sym(
            (1.0 - t_arr[..., None, None]) * log_a + t_arr[..., None, None] * log_b,
            validate=False,
        )
    if metric == "euclid":
        return symmetrize(
            (1.0 - t_arr[..., None, None]) * a + t_arr[..., None, None] * b
        )
    raise ValueError(
        f"Metric {metric!r} has no closed-form geodesic. Use 'airm', "
        f"'logeuclid', or 'euclid'."
    )


def _powm_broadcast(a: FloatArray, t: FloatArray) -> FloatArray:
    """Matrix power with a per-matrix exponent.

    :func:`geoq.geometry.spd.powm_spd` takes a scalar exponent. Geodesic
    interpolation needs a different ``t`` per matrix in a batch, so the
    spectral form is applied directly here rather than looping.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        t: Exponents broadcastable to ``(...)``.

    Returns:
        SPD stack of shape ``(..., n, n)``.
    """
    if t.ndim == 0:
        return powm_spd(a, float(t), validate=False)
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    powered = np.power(eigenvalues, t[..., None])
    return symmetrize(
        (eigenvectors * powered[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    )


def parallel_transport(
    source: ArrayLike,
    destination: ArrayLike,
    tangent: ArrayLike,
    *,
    validate: bool = True,
) -> FloatArray:
    r"""Parallel-transport a tangent vector from ``source`` to ``destination``.

    .. math::
        \Gamma_{S \to D}(V) = E V E^T, \quad E = (D S^{-1})^{1/2}

    Tangent vectors at different base points live in different spaces and
    cannot be compared directly. Transport is what makes them comparable, and
    it is the correct formulation of cross-subject domain adaptation: rather
    than re-centring each subject and hoping the tangent coordinates align,
    transport moves them into a shared frame along the manifold.

    Args:
        source: SPD stack of shape ``(..., n, n)``, current base point.
        destination: SPD stack of shape ``(..., n, n)``, target base point.
        tangent: Symmetric stack of shape ``(..., n, n)`` at ``source``.
        validate: Whether to enforce the input contracts.

    Returns:
        Symmetric tangent vectors at ``destination``, shape ``(..., n, n)``.
    """
    if validate:
        source = check_spd(source, name="source")
        destination = check_spd(destination, name="destination")
        tangent = check_symmetric(tangent, name="tangent")

    # E = (D S^-1)^1/2 is computed via the symmetric congruence
    # S^-1/2 (S^-1/2 D S^-1/2)^1/2 S^1/2, because D S^-1 is a product of two
    # symmetric matrices and is generally *not* symmetric -- feeding it to a
    # symmetric square root would silently discard its antisymmetric part.
    source_root = sqrtm_spd(source, validate=False)
    source_inv_root = invsqrtm_spd(source, validate=False)
    inner = symmetrize(source_inv_root @ destination @ source_inv_root)
    transporter = source_root @ sqrtm_spd(inner, validate=False) @ source_inv_root
    return symmetrize(
        transporter
        @ np.asarray(tangent, dtype=np.float64)
        @ np.swapaxes(transporter, -1, -2)
    )


# --------------------------------------------------------------------------- #
# Means
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MeanResult:
    """Outcome of an iterative Frechet mean computation.

    Returned by :func:`frechet_mean` when ``return_info`` is set, so that
    convergence behaviour can be logged per fold. Silent non-convergence
    inside a cross-validation loop produces a plausible-looking but wrong
    class centroid, and an MDM classifier built on it fails in a way that is
    almost impossible to diagnose from accuracy alone.

    Attributes:
        mean: The estimated mean, SPD of shape ``(n, n)``.
        n_iter: Iterations performed.
        converged: Whether the tolerance was reached before the cap.
        final_criterion: Final relative tangent norm.
        history: Criterion value at each iteration.
        tolerance_used: The tolerance actually applied. Differs from the
            requested value when the input's conditioning made the request
            unachievable; recording it keeps the convergence claim auditable
            rather than implicit.
    """

    mean: FloatArray
    n_iter: int
    converged: bool
    final_criterion: float
    history: tuple[float, ...]
    tolerance_used: float = 0.0


def mean_euclid(x: ArrayLike, *, weights: ArrayLike | None = None) -> FloatArray:
    """Arithmetic mean of a set of symmetric matrices.

    Args:
        x: Stack of shape ``(m, n, n)``.
        weights: Optional non-negative weights of shape ``(m,)``.

    Returns:
        The mean, shape ``(n, n)``.
    """
    arr = check_symmetric(x, name="x")
    w = _normalise_weights(weights, arr.shape[0])
    return symmetrize(np.tensordot(w, arr, axes=(0, 0)))


def mean_logeuclid(x: ArrayLike, *, weights: ArrayLike | None = None) -> FloatArray:
    """Log-Euclidean mean, ``exp(mean(log(X_i)))``.

    Closed-form and therefore fast. Also the standard initialisation for the
    AIRM Frechet mean: starting from it rather than from the arithmetic mean
    typically halves the iteration count, because it already respects the
    multiplicative structure of the manifold.

    Args:
        x: SPD stack of shape ``(m, n, n)``.
        weights: Optional non-negative weights of shape ``(m,)``.

    Returns:
        SPD mean of shape ``(n, n)``.
    """
    arr = check_spd(x, name="x")
    w = _normalise_weights(weights, arr.shape[0])
    log_mean = np.tensordot(w, logm_spd(arr, validate=False), axes=(0, 0))
    return expm_sym(log_mean, validate=False)


def _normalise_weights(weights: ArrayLike | None, m: int) -> FloatArray:
    """Validate weights and normalise them to sum to one.

    Args:
        weights: Optional weights of shape ``(m,)``.
        m: Expected number of samples.

    Returns:
        Normalised weights of shape ``(m,)``.

    Raises:
        ValueError: If the shape is wrong, any weight is negative, or the sum
            is not positive.
    """
    if weights is None:
        return np.full(m, 1.0 / m, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (m,):
        raise ValueError(f"weights must have shape ({m},), got {w.shape}.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return w / total


def frechet_mean(
    x: ArrayLike,
    *,
    metric: Metric = "airm",
    weights: ArrayLike | None = None,
    init: ArrayLike | None = None,
    tol: float = DEFAULT_MEAN_TOL,
    max_iter: int = DEFAULT_MEAN_MAX_ITER,
    return_info: bool = False,
) -> FloatArray | MeanResult:
    r"""Frechet (Karcher) mean under the named metric.

    For AIRM there is no closed form for more than two matrices, so the mean
    is found by Riemannian gradient descent on the sum of squared geodesic
    distances. The update is

    .. math::
        M \leftarrow \mathrm{Exp}_M\!\left(\sum_i w_i \mathrm{Log}_M(X_i)\right)

    which is Newton-like near the optimum and converges in a handful of
    iterations for real EEG covariances.

    Two implementation choices are worth stating:

    * **Initialisation from the Log-Euclidean mean.** It is closed-form and
      already respects the manifold's multiplicative structure, typically
      halving the iterations relative to an arithmetic-mean start.
    * **Step halving on divergence.** The full step can overshoot for widely
      dispersed inputs. Halving on any increase of the criterion guarantees
      monotone descent, at the cost of an occasional extra iteration. Without
      it the iteration can oscillate indefinitely and hit the cap while
      appearing to make progress.

    The convergence criterion is the tangent norm relative to the norm of the
    current estimate. An absolute criterion would be unit-dependent: the same
    data in volts and microvolts would need different tolerances.

    Args:
        x: SPD stack of shape ``(m, n, n)``.
        metric: One of :data:`METRICS`. ``"logeuclid"`` and ``"euclid"`` return
            immediately via their closed forms.
        weights: Optional non-negative weights of shape ``(m,)``.
        init: Optional SPD starting point of shape ``(n, n)``. Passing the
            previous fold's mean is a meaningful speed-up in MDM.
        tol: Requested convergence tolerance on the relative tangent norm.
            Raised automatically to the achievable floor when the input's
            conditioning makes it unreachable; see
            :data:`TOLERANCE_FLOOR_FACTOR`.
        max_iter: Iteration cap.
        return_info: If True, return a :class:`MeanResult` instead of the bare
            mean.

    Returns:
        The SPD mean of shape ``(n, n)``, or a :class:`MeanResult`.

    Raises:
        ValueError: If ``metric`` is unknown or ``x`` is not a 3-d stack.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Available: {sorted(METRICS)}.")

    arr = check_spd(x, name="x")
    if arr.ndim != 3:
        raise ValueError(f"x must have shape (m, n, n), got {arr.shape}.")

    w = _normalise_weights(weights, arr.shape[0])

    if metric == "euclid":
        result = mean_euclid(arr, weights=w)
        return _wrap(result, 0, True, 0.0, ()) if return_info else result
    if metric == "logeuclid":
        result = mean_logeuclid(arr, weights=w)
        return _wrap(result, 0, True, 0.0, ()) if return_info else result

    # Stein has an iterative fixed point of its own; rather than approximate
    # it with the AIRM iteration and mislabel the result, it is refused.
    if metric == "stein":
        raise ValueError(
            "The Stein mean requires a distinct fixed-point iteration that is "
            "not implemented. Use metric='airm' for the geometric mean, or "
            "'logeuclid' for a fast closed-form approximation."
        )

    mean = (
        check_spd(init, name="init")
        if init is not None
        else mean_logeuclid(arr, weights=w)
    )

    #  Raise the tolerance to what the input's conditioning actually permits.
    #  Requesting more precision than float64 can deliver does not produce a
    #  better mean; it produces a false report of failure.
    kappa = float(np.max(condition_number(arr, name="x")))
    floor = TOLERANCE_FLOOR_FACTOR * float(np.finfo(np.float64).eps) * kappa
    effective_tol = max(tol, floor)
    if effective_tol > tol:
        logger.debug(
            "frechet_mean: tolerance raised from %.3e to %.3e; the input set "
            "has condition number %.2e, below which the convergence criterion "
            "cannot be driven.",
            tol,
            effective_tol,
            kappa,
        )

    history: list[float] = []
    converged = False
    criterion = np.inf
    step = 1.0
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        tangent_mean = np.tensordot(w, log_map(mean, arr, validate=False), axes=(0, 0))
        criterion = float(
            np.linalg.norm(tangent_mean) / max(np.linalg.norm(mean), 1e-300)
        )
        history.append(criterion)

        if criterion < effective_tol:
            converged = True
            break

        # Monotone descent guard: if the criterion grew, the previous step
        # overshot, so shrink it. Growth is otherwise a silent oscillation.
        if len(history) > 1 and criterion > history[-2]:
            step *= 0.5
            logger.debug(
                "frechet_mean: criterion increased at iteration %d; step -> %.4g",
                n_iter,
                step,
            )

        mean = exp_map(mean, step * tangent_mean, validate=False)

    if not converged:
        logger.warning(
            "frechet_mean did not converge in %d iterations (criterion %.3e, "
            "effective tolerance %.3e, condition number %.2e). The result is "
            "the last iterate and may not be the geometric mean; inspect the "
            "dispersion of the input set before using it as a class centroid.",
            max_iter,
            criterion,
            effective_tol,
            kappa,
        )

    return (
        _wrap(mean, n_iter, converged, criterion, tuple(history), effective_tol)
        if return_info
        else mean
    )


def _wrap(
    mean: FloatArray,
    n_iter: int,
    converged: bool,
    criterion: float,
    history: tuple[float, ...],
    tolerance_used: float = 0.0,
) -> MeanResult:
    """Package a mean computation into a :class:`MeanResult`."""
    return MeanResult(
        mean=mean,
        n_iter=n_iter,
        converged=converged,
        final_criterion=criterion,
        history=history,
        tolerance_used=tolerance_used,
    )
