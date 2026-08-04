"""Symmetric positive-definite (SPD) matrix primitives.

This module is the numerical bedrock of the framework. Every Riemannian
operation (AIRM geodesic distance, Exp/Log maps, Frechet mean, tangent-space
projection) and every geometry-aware quantum encoding is expressed in terms of
the spectral primitives defined here.

Design contract
---------------
1. **Spectral, not generic.** All matrix functions are computed via the
   symmetric eigendecomposition ``A = V diag(w) V^T`` and applied to ``w``.
   This is faster than the Schur-based routines in :mod:`scipy.linalg`, is
   exact for the SPD case, and -- unlike ``scipy.linalg.logm`` -- can never
   return a complex result for a valid input.
2. **Scale-relative tolerances.** Positive-definiteness is judged by
   ``lambda_min > rtol * lambda_max``, never by an absolute threshold. EEG
   covariance matrices carry physical units (microvolts squared) and their
   eigenvalues routinely span ``1e-14`` to ``1e-8``; an absolute tolerance
   would reject valid data or accept singular data depending only on the
   recording amplifier's gain.
3. **No silent repair.** Regularisation and eigenvalue flooring are explicit,
   opt-in operations that log when they actually change a matrix. Silently
   conditioning an input would corrupt geodesic distances and produce results
   that cannot be reproduced from the raw data.
4. **Batched by default.** Every function accepts a stack of shape
   ``(..., n, n)``. A single EEG session yields hundreds of covariance
   matrices; looping in Python over them is the difference between seconds and
   minutes on every cross-validation fold.
5. **Output symmetry is enforced.** Floating-point round-off in ``V f(w) V^T``
   breaks exact symmetry at the 1e-16 level. Downstream validators would then
   reject a mathematically valid result, so results are symmetrised before
   return.

References:
----------
Bhatia, R. (2007). *Positive Definite Matrices*. Princeton University Press.
Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. SIAM.
Congedo, M., Barachant, A., & Bhatia, R. (2017). Riemannian geometry for
    EEG-based brain-computer interfaces: a primer and a review.
    *Brain-Computer Interfaces*, 4(3), 155-174.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_PD_RTOL",
    "DEFAULT_SYMMETRY_RTOL",
    "FloatArray",
    "NotPositiveDefiniteError",
    "NotSquareError",
    "NotSymmetricError",
    "SPDError",
    "check_spd",
    "check_symmetric",
    "condition_number",
    "eigh_symmetric",
    "expm_sym",
    "inv_spd",
    "invsqrtm_spd",
    "is_spd",
    "is_symmetric",
    "logdet_spd",
    "logm_spd",
    "powm_spd",
    "project_to_spd",
    "random_spd",
    "regularize_spd",
    "shrink_toward_identity",
    "spd_funm",
    "sqrtm_spd",
    "symmetrize",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
"""Alias for the only floating dtype used in ``core``.

Float32 is never used here: the AIRM metric involves a matrix logarithm of
eigenvalue ratios, and float32 loses roughly half the significant digits of a
geodesic distance for ill-conditioned EEG covariances.
"""

DEFAULT_SYMMETRY_RTOL: Final[float] = 1e-10
"""Relative Frobenius tolerance for the symmetry check."""

DEFAULT_PD_RTOL: Final[float] = 1e-12
"""Relative tolerance for positive-definiteness: ``w_min > rtol * w_max``."""


# --------------------------------------------------------------------------- #
# Exceptions
#
# These are defined here rather than in a package-level ``exceptions`` module
# so that ``core`` remains importable with zero intra-package dependencies.
# ``geoq.exceptions`` will re-export them as a facade; it will not redefine
# them, so there is exactly one class object per error type.
# --------------------------------------------------------------------------- #


class SPDError(ValueError):
    """Base class for all violations of the SPD contract."""


class NotSquareError(SPDError):
    """Raised when an array is not a stack of square matrices."""


class NotSymmetricError(SPDError):
    """Raised when a matrix is not symmetric to within tolerance."""


class NotPositiveDefiniteError(SPDError):
    """Raised when a symmetric matrix has a non-positive eigenvalue."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _as_matrix_stack(a: ArrayLike, *, name: str = "matrix") -> FloatArray:
    """Coerce ``a`` to a float64 stack of square matrices and validate shape.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        name: Symbol name used in error messages, so a failure deep inside a
            cross-validation fold names the offending argument.

    Returns:
        A C-contiguous float64 array of shape ``(..., n, n)``.

    Raises:
        NotSquareError: If ``a`` has fewer than two dimensions or its trailing
            two dimensions differ.
        SPDError: If ``a`` contains NaN or infinity.
    """
    arr = np.asarray(a, dtype=np.float64)

    if arr.ndim < 2:
        raise NotSquareError(
            f"{name!r} must have at least 2 dimensions, got shape {arr.shape}."
        )
    if arr.shape[-1] != arr.shape[-2]:
        raise NotSquareError(
            f"{name!r} must be a stack of square matrices with shape (..., n, n); "
            f"got trailing dimensions {arr.shape[-2:]}."
        )

    finite = np.isfinite(arr)
    if not finite.all():
        n_bad = int((~finite).sum())
        first_bad = tuple(int(i) for i in np.argwhere(~finite)[0])
        raise SPDError(
            f"{name!r} contains {n_bad} non-finite value(s); first at index "
            f"{first_bad}. Non-finite covariances usually indicate an empty "
            f"epoch, a flat (dead) EEG channel, or an unhandled NaN in the "
            f"raw recording."
        )

    return np.ascontiguousarray(arr)


def _matrix_indices(mask: NDArray[np.bool_]) -> str:
    """Format the batch indices flagged by ``mask`` for an error message."""
    bad = np.argwhere(mask)
    shown = ", ".join(str(tuple(int(i) for i in idx)) for idx in bad[:5])
    suffix = ", ..." if bad.shape[0] > 5 else ""
    return f"{bad.shape[0]} matrix/matrices at batch index {shown}{suffix}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def symmetrize(a: ArrayLike) -> FloatArray:
    """Return the symmetric part ``(A + A^T) / 2`` of each matrix in a stack.

    Args:
        a: Array-like of shape ``(..., n, n)``.

    Returns:
        The symmetrised stack, shape ``(..., n, n)``.
    """
    arr = _as_matrix_stack(a, name="a")
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def is_symmetric(
    a: ArrayLike, *, rtol: float = DEFAULT_SYMMETRY_RTOL
) -> NDArray[np.bool_]:
    """Test symmetry using a scale-relative Frobenius criterion.

    The test is ``||A - A^T||_F <= rtol * ||A||_F`` rather than an elementwise
    absolute comparison, so the verdict is invariant to rescaling the units of
    the recording.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        rtol: Relative tolerance.

    Returns:
        Boolean array of shape ``(...)``. For a single matrix this is a 0-d
        array, which behaves correctly in an ``if`` statement.
    """
    arr = _as_matrix_stack(a, name="a")
    asymmetry = np.linalg.norm(arr - np.swapaxes(arr, -1, -2), axis=(-2, -1))
    scale = np.linalg.norm(arr, axis=(-2, -1))
    # A zero matrix is trivially symmetric; guard the division without
    # special-casing it out of the comparison.
    scale = np.where(scale == 0.0, 1.0, scale)
    return asymmetry <= rtol * scale


def is_spd(
    a: ArrayLike,
    *,
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
    pd_rtol: float = DEFAULT_PD_RTOL,
) -> NDArray[np.bool_]:
    """Test whether each matrix in a stack is symmetric positive definite.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        symmetry_rtol: Relative tolerance for the symmetry test.
        pd_rtol: Relative tolerance for positive-definiteness; a matrix passes
            when ``w_min > pd_rtol * w_max``.

    Returns:
        Boolean array of shape ``(...)``.
    """
    arr = _as_matrix_stack(a, name="a")
    symmetric = is_symmetric(arr, rtol=symmetry_rtol)
    # eigvalsh reads only the lower triangle, so it must not be trusted before
    # the symmetry test has run.
    w = np.linalg.eigvalsh(symmetrize(arr))
    w_min = w[..., 0]
    w_max = w[..., -1]
    positive = (w_max > 0.0) & (w_min > pd_rtol * w_max)
    return symmetric & positive


def check_symmetric(
    a: ArrayLike,
    *,
    name: str = "matrix",
    rtol: float = DEFAULT_SYMMETRY_RTOL,
) -> FloatArray:
    """Validate symmetry and return the symmetrised stack.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        name: Symbol name used in the error message.
        rtol: Relative tolerance for the symmetry test.

    Returns:
        The validated, symmetrised stack.

    Raises:
        NotSymmetricError: If any matrix fails the symmetry test.
    """
    arr = _as_matrix_stack(a, name=name)
    ok = is_symmetric(arr, rtol=rtol)
    if not np.all(ok):
        raise NotSymmetricError(
            f"{name!r} is not symmetric to relative tolerance {rtol:g}: "
            f"{_matrix_indices(~np.atleast_1d(ok))} failed."
        )
    return symmetrize(arr)


def check_spd(
    a: ArrayLike,
    *,
    name: str = "matrix",
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
    pd_rtol: float = DEFAULT_PD_RTOL,
) -> FloatArray:
    """Validate the full SPD contract and return the symmetrised stack.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        name: Symbol name used in error messages.
        symmetry_rtol: Relative tolerance for the symmetry test.
        pd_rtol: Relative tolerance for positive-definiteness.

    Returns:
        The validated, symmetrised stack.

    Raises:
        NotSymmetricError: If any matrix is not symmetric.
        NotPositiveDefiniteError: If any matrix is singular or indefinite. The
            message reports the offending eigenvalue and condition number,
            which is the information needed to decide between shrinkage,
            more samples per epoch, or dropping a dead channel.
    """
    arr = check_symmetric(a, name=name, rtol=symmetry_rtol)
    w = np.linalg.eigvalsh(arr)
    w_min = w[..., 0]
    w_max = w[..., -1]
    bad = ~((w_max > 0.0) & (w_min > pd_rtol * w_max))
    if np.any(bad):
        idx = np.argwhere(np.atleast_1d(bad))[0]
        sel = tuple(int(i) for i in idx) if arr.ndim > 2 else ()
        raise NotPositiveDefiniteError(
            f"{name!r} is not positive definite: "
            f"{_matrix_indices(np.atleast_1d(bad))} failed. "
            f"First failure has lambda_min={float(np.atleast_1d(w_min)[tuple(idx)]):.6e}, "
            f"lambda_max={float(np.atleast_1d(w_max)[tuple(idx)]):.6e} "
            f"(batch index {sel}). Rank-deficient EEG covariances typically "
            f"come from too few samples per epoch (need n_times >> n_channels), "
            f"a flat channel, or an average-reference/ICA rank reduction. Use "
            f"regularize_spd or shrink_toward_identity explicitly."
        )
    return arr


# --------------------------------------------------------------------------- #
# Spectral primitives
# --------------------------------------------------------------------------- #


def eigh_symmetric(
    a: ArrayLike,
    *,
    validate: bool = True,
    name: str = "matrix",
) -> tuple[FloatArray, FloatArray]:
    """Eigendecompose a stack of symmetric matrices.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        validate: If True, enforce symmetry before decomposing. Disable only
            inside hot loops where the caller has already validated.
        name: Symbol name used in error messages.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` of shapes ``(..., n)`` and
        ``(..., n, n)``. Eigenvalues are in ascending order; column ``i`` of
        the eigenvector array corresponds to eigenvalue ``i``.

    Raises:
        NotSymmetricError: If ``validate`` is True and symmetry fails.
    """
    arr = (
        check_symmetric(a, name=name)
        if validate
        else symmetrize(_as_matrix_stack(a, name=name))
    )
    eigenvalues, eigenvectors = np.linalg.eigh(arr)
    return eigenvalues, eigenvectors


def spd_funm(
    a: ArrayLike,
    func: Callable[[FloatArray], FloatArray],
    *,
    require_pd: bool = True,
    eigenvalue_floor: float | None = None,
    validate: bool = True,
    name: str = "matrix",
) -> FloatArray:
    """Apply a scalar function to the spectrum of a symmetric matrix stack.

    This is the single primitive behind :func:`logm_spd`, :func:`expm_sym`,
    :func:`sqrtm_spd`, :func:`invsqrtm_spd` and :func:`powm_spd`. Keeping one
    implementation means there is one place where numerical policy lives.

    Args:
        a: Array-like of shape ``(..., n, n)``.
        func: Vectorised scalar function applied to the eigenvalues, e.g.
            :func:`numpy.log`.
        require_pd: If True, validate positive-definiteness before applying
            ``func``. Set False for functions defined on all reals, such as
            the matrix exponential of a symmetric (tangent-space) matrix.
        eigenvalue_floor: If given, eigenvalues are clipped from below at
            ``eigenvalue_floor * lambda_max`` before ``func`` is applied, and a
            warning is logged whenever clipping actually occurs. Mutually
            exclusive in spirit with ``require_pd=True``: flooring is a repair,
            and repairs must be visible.
        validate: If False, skip symmetry validation (hot-loop escape hatch).
        name: Symbol name used in error messages.

    Returns:
        The transformed stack, symmetrised, shape ``(..., n, n)``.

    Raises:
        NotSymmetricError: If the input is not symmetric.
        NotPositiveDefiniteError: If ``require_pd`` is True and the input is
            not positive definite.
    """
    if validate and require_pd:
        arr = check_spd(a, name=name)
    elif validate:
        arr = check_symmetric(a, name=name)
    else:
        arr = symmetrize(_as_matrix_stack(a, name=name))

    eigenvalues, eigenvectors = np.linalg.eigh(arr)

    if eigenvalue_floor is not None:
        floor = eigenvalue_floor * eigenvalues[..., -1:]
        n_clipped = int(np.sum(eigenvalues < floor))
        if n_clipped:
            logger.warning(
                "spd_funm(%s): floored %d eigenvalue(s) at %g * lambda_max. "
                "This modifies the data; record it in the experiment provenance.",
                name,
                n_clipped,
                eigenvalue_floor,
            )
        eigenvalues = np.maximum(eigenvalues, floor)

    transformed = func(eigenvalues)
    # V diag(f(w)) V^T, written as a broadcast column scaling to avoid
    # materialising the diagonal matrix for every trial in the batch.
    result = (eigenvectors * transformed[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return symmetrize(result)


def logm_spd(
    a: ArrayLike, *, validate: bool = True, name: str = "matrix"
) -> FloatArray:
    """Matrix logarithm of an SPD stack.

    This is the Riemannian Log map at the identity and the entry point to the
    tangent space. It is also the object Paper 4's geometry-aware encoding acts
    on: its eigenvalues become rotation angles and its eigenvectors a basis
    change.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        validate: Whether to enforce the SPD contract.
        name: Symbol name used in error messages.

    Returns:
        Symmetric (generally indefinite) stack of shape ``(..., n, n)``.
    """
    return spd_funm(a, np.log, require_pd=True, validate=validate, name=name)


def expm_sym(
    a: ArrayLike, *, validate: bool = True, name: str = "matrix"
) -> FloatArray:
    """Matrix exponential of a symmetric stack, returning SPD matrices.

    Inverse of :func:`logm_spd`. Defined for any symmetric input, so positive
    definiteness is deliberately not required.

    Args:
        a: Symmetric stack of shape ``(..., n, n)``.
        validate: Whether to enforce symmetry.
        name: Symbol name used in error messages.

    Returns:
        SPD stack of shape ``(..., n, n)``.
    """
    return spd_funm(a, np.exp, require_pd=False, validate=validate, name=name)


def sqrtm_spd(
    a: ArrayLike, *, validate: bool = True, name: str = "matrix"
) -> FloatArray:
    """Principal square root of an SPD stack."""
    return spd_funm(a, np.sqrt, require_pd=True, validate=validate, name=name)


def invsqrtm_spd(
    a: ArrayLike, *, validate: bool = True, name: str = "matrix"
) -> FloatArray:
    """Inverse principal square root of an SPD stack.

    This is the whitening operator behind AIRM geodesic distance and
    Riemannian re-centring for cross-subject transfer.
    """
    return spd_funm(
        a, lambda w: 1.0 / np.sqrt(w), require_pd=True, validate=validate, name=name
    )


def powm_spd(
    a: ArrayLike, p: float, *, validate: bool = True, name: str = "matrix"
) -> FloatArray:
    """Real matrix power ``A ** p`` of an SPD stack.

    The common half-integer exponents dispatch to the dedicated
    implementations above rather than to ``np.power``. Two reasons, both
    load-bearing:

    * **Accuracy.** ``np.power(w, -0.5)`` is evaluated as
      ``exp(-0.5 * log(w))`` and loses roughly one decimal digit relative to
      the direct reciprocal square root.
    * **Uniqueness of the answer.** Without dispatch, ``powm_spd(A, -0.5)``
      and ``invsqrtm_spd(A)`` return arrays that differ in the last ulp. Both
      are whitening operators feeding AIRM geodesic distance, so two call
      sites computing "the same" quantity would produce results that disagree
      at the 1e-16 level -- enough to break bitwise reproducibility checks and
      to make a locked results file irreproducible across refactors.

    Args:
        a: SPD stack of shape ``(..., n, n)``.
        p: Real exponent.
        validate: Whether to enforce the SPD contract.
        name: Symbol name used in error messages.

    Returns:
        SPD stack of shape ``(..., n, n)``.
    """
    special_cases: dict[float, Callable[..., FloatArray]] = {
        0.5: sqrtm_spd,
        -0.5: invsqrtm_spd,
        -1.0: inv_spd,
    }
    handler = special_cases.get(float(p))
    if handler is not None:
        return handler(a, validate=validate, name=name)

    return spd_funm(
        a, lambda w: np.power(w, p), require_pd=True, validate=validate, name=name
    )


def inv_spd(a: ArrayLike, *, validate: bool = True, name: str = "matrix") -> FloatArray:
    """Inverse of an SPD stack, computed spectrally.

    Slower than a Cholesky solve, but it validates conditioning on the way and
    guarantees an exactly symmetric result, which matters because the inverse
    feeds geodesic distances where asymmetry compounds.
    """
    return spd_funm(a, lambda w: 1.0 / w, require_pd=True, validate=validate, name=name)


def logdet_spd(
    a: ArrayLike, *, validate: bool = True, name: str = "matrix"
) -> FloatArray:
    """Log-determinant of an SPD stack, computed as ``sum(log(w))``.

    Computed from the spectrum rather than from ``det`` because the
    determinant of a 22-channel EEG covariance underflows float64.

    Returns:
        Array of shape ``(...)``.
    """
    arr = check_spd(a, name=name) if validate else symmetrize(_as_matrix_stack(a))
    return np.sum(np.log(np.linalg.eigvalsh(arr)), axis=-1)


def condition_number(a: ArrayLike, *, name: str = "matrix") -> FloatArray:
    """Spectral condition number ``lambda_max / lambda_min`` of a symmetric stack.

    Report this in every dataset audit: it is the single best predictor of
    whether a Riemannian pipeline will be numerically trustworthy on a given
    EEG recording.

    Returns:
        Array of shape ``(...)``; ``inf`` where the matrix is singular.
    """
    arr = check_symmetric(a, name=name)
    w = np.linalg.eigvalsh(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(w[..., -1]) / np.abs(w[..., 0])


# --------------------------------------------------------------------------- #
# Explicit conditioning (never automatic)
# --------------------------------------------------------------------------- #


def regularize_spd(a: ArrayLike, *, epsilon: float = 1e-8) -> FloatArray:
    """Add a trace-relative ridge: ``A + epsilon * (tr(A) / n) * I``.

    The ridge is scaled by the mean eigenvalue rather than being absolute, so
    the amount of regularisation is invariant to the recording's units and
    comparable across datasets. An absolute ridge would be a no-op on one
    dataset and dominant on another.

    Args:
        a: Symmetric stack of shape ``(..., n, n)``.
        epsilon: Relative ridge strength.

    Returns:
        Regularised stack of shape ``(..., n, n)``.

    Raises:
        ValueError: If ``epsilon`` is negative.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    arr = check_symmetric(a, name="a")
    n = arr.shape[-1]
    mean_eigenvalue = np.trace(arr, axis1=-2, axis2=-1)[..., None, None] / n
    return arr + epsilon * mean_eigenvalue * np.eye(n)


def shrink_toward_identity(a: ArrayLike, *, alpha: float) -> FloatArray:
    """Ledoit-Wolf style shrinkage: ``(1 - alpha) A + alpha (tr(A) / n) I``.

    Preferred over :func:`regularize_spd` when epochs are short relative to the
    channel count, because it is trace-preserving and therefore does not
    inflate the overall signal power.

    Args:
        a: Symmetric stack of shape ``(..., n, n)``.
        alpha: Shrinkage intensity in ``[0, 1]``.

    Returns:
        Shrunk stack of shape ``(..., n, n)``.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")
    arr = check_symmetric(a, name="a")
    n = arr.shape[-1]
    mean_eigenvalue = np.trace(arr, axis1=-2, axis2=-1)[..., None, None] / n
    return (1.0 - alpha) * arr + alpha * mean_eigenvalue * np.eye(n)


def project_to_spd(a: ArrayLike, *, floor: float = 1e-10) -> FloatArray:
    """Project a symmetric stack onto the SPD cone by flooring its spectrum.

    A last-resort repair for matrices that fail :func:`check_spd`. It logs a
    warning whenever it changes anything, because a projection that fires
    during an experiment is a data-quality finding, not an implementation
    detail, and must appear in the run's provenance record.

    Args:
        a: Symmetric stack of shape ``(..., n, n)``.
        floor: Eigenvalue floor relative to ``lambda_max``.

    Returns:
        SPD stack of shape ``(..., n, n)``.
    """
    return spd_funm(
        a,
        lambda w: w,
        require_pd=False,
        eigenvalue_floor=floor,
        name="a",
    )


# --------------------------------------------------------------------------- #
# Synthetic data (tests, toy circuits, unit-level sanity checks)
# --------------------------------------------------------------------------- #


def random_spd(
    n: int,
    *,
    rng: np.random.Generator,
    batch: int | tuple[int, ...] | None = None,
    condition_number: float | None = None,
    scale: float = 1.0,
) -> FloatArray:
    """Draw random SPD matrices with optional control over conditioning.

    Args:
        n: Matrix dimension.
        rng: NumPy generator. Required, and never defaulted: every stochastic
            call site in this framework must receive its seed explicitly so
            that runs are reproducible from the config alone.
        batch: Batch shape. ``None`` returns a single ``(n, n)`` matrix.
        condition_number: If given, eigenvalues are geometrically spaced from
            ``1`` to this value, letting tests target the ill-conditioned
            regime that real EEG covariances occupy. If ``None``, eigenvalues
            are drawn log-uniformly from ``[0.1, 10]``.
        scale: Multiplicative factor applied to all eigenvalues, used to check
            that results are invariant to the recording's units.

    Returns:
        SPD stack of shape ``(*batch, n, n)``.

    Raises:
        ValueError: If ``n < 1``, ``condition_number < 1``, or ``scale <= 0``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if condition_number is not None and condition_number < 1.0:
        raise ValueError(f"condition_number must be >= 1, got {condition_number}.")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}.")

    batch_shape: tuple[int, ...]
    if batch is None:
        batch_shape = ()
    elif isinstance(batch, int):
        batch_shape = (batch,)
    else:
        batch_shape = tuple(batch)

    # A Haar-distributed orthogonal basis via QR of a Gaussian matrix, with the
    # sign correction that makes the distribution genuinely Haar rather than
    # biased by LAPACK's choice of R's diagonal signs.
    gaussian = rng.standard_normal((*batch_shape, n, n))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diagonal(r, axis1=-2, axis2=-1))
    signs = np.where(signs == 0.0, 1.0, signs)
    q = q * signs[..., None, :]

    if condition_number is None:
        eigenvalues = np.exp(
            rng.uniform(np.log(0.1), np.log(10.0), size=(*batch_shape, n))
        )
    elif n == 1:
        eigenvalues = np.ones((*batch_shape, 1))
    else:
        eigenvalues = np.geomspace(1.0, condition_number, n)
        eigenvalues = np.broadcast_to(eigenvalues, (*batch_shape, n)).copy()

    eigenvalues = np.sort(eigenvalues, axis=-1) * scale
    result = (q * eigenvalues[..., None, :]) @ np.swapaxes(q, -1, -2)
    return symmetrize(result)