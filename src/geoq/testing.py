"""Numerical helpers for testing geometry code.

Shipped as part of the package rather than kept in ``tests/`` for two reasons.
First, ``from conftest import ...`` relies on pytest inserting the test
directory onto ``sys.path``, which stops working under
``--import-mode=importlib`` -- a fragile foundation for a suite that must keep
running for five years. Second, anyone extending this framework for their own
experiments needs the same tolerance policy; a private helper would leave them
inventing their own, and inconsistent tolerances across a codebase are how
numerically wrong results get accepted.

Tolerance policy
----------------
Absolute tolerances are never compared against a matrix's raw magnitude. EEG
covariance entries span roughly ``1e-14`` to ``1e-8`` depending only on whether
a recording is stored in volts or microvolts, so a fixed epsilon would make a
test's verdict depend on the amplifier's gain. Errors are judged relative to
the norm of the reference, and where a bound is needed it is expressed as a
multiple of ``eps * kappa``, the standard error bound for a spectral matrix
function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DIMENSIONS",
    "EPS",
    "assert_exactly_symmetric",
    "assert_spd",
    "relative_error",
    "spectral_error_bound",
]

EPS: float = float(np.finfo(np.float64).eps)

#: Matrix dimensions the thesis actually touches: 2 is the Paper-4
#: proof-of-concept case, 3 and 8 are toy and small-montage sizes, and 22 is
#: BCI Competition IV 2a's full channel count. Testing at 5 and 50 instead
#: would prove nothing about the data this framework will see.
DIMENSIONS: list[int] = [2, 3, 8, 22]


def relative_error(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
    """Frobenius error relative to the norm of ``expected``.

    Args:
        actual: Computed array.
        expected: Reference array.

    Returns:
        ``||actual - expected||_F / ||expected||_F``, falling back to the
        absolute error when ``expected`` is the zero array.
    """
    denominator = float(np.linalg.norm(expected))
    numerator = float(np.linalg.norm(actual - expected))
    return numerator / denominator if denominator > 0 else numerator


def spectral_error_bound(
    a: NDArray[np.float64],
    *,
    factor: float = 250.0,
    kappa_power: float = 0.0,
) -> float:
    r"""Admissible relative error for a spectral matrix function applied to ``a``.

    The bound is

    .. math::
        \text{tol} = \text{factor} \cdot \epsilon \cdot \sqrt{n}
                     \cdot \left(1 + \log_{10} \kappa(A)\right)

    which reflects the two mechanisms that actually generate error in
    ``V f(w) V^T``:

    * **Dimension.** LAPACK's symmetric eigensolver has backward error
      :math:`O(n \epsilon \|A\|_2)`, and the two matrix products in the
      reconstruction contribute again. Measured across ``n`` from 2 to 128,
      the growth is close to :math:`\sqrt{n}` and saturates thereafter.
    * **Conditioning, logarithmically.** Perturbing an eigenvalue by a
      relative :math:`\delta` shifts its logarithm by an absolute
      :math:`\delta`, so a spectrum spanning :math:`\kappa` contributes a
      term growing like :math:`\log \kappa` -- not like :math:`\kappa`.

    Why not ``factor * eps * kappa``
    --------------------------------
    That form, used in an earlier revision, was wrong in both directions.

    At :math:`\kappa = 1` it collapses to exactly ``factor * eps`` with no
    allowance for the :math:`O(n)` accumulation that is always present, so a
    perfectly conditioned matrix could fail at 107 eps against a 100 eps
    bound. Worse, at :math:`\kappa = 10^8` it permitted a relative error of
    ``2.2e-06``: measurement shows the true round-trip error there is around
    ``300 eps``, so a genuine regression of six orders of magnitude would have
    passed silently. A tolerance loose enough to hide real defects is not a
    conservative choice; it is an absent test.

    The constant 250 was chosen by measurement, not taste. Across ``n`` in
    ``[2, 128]`` and :math:`\kappa` in :math:`[1, 10^{11}]`, it leaves a
    tightest margin of 6.4x over the worst observed error and a loosest of
    112x -- enough headroom that the test does not flake under a new
    Hypothesis draw, while remaining roughly a million times tighter than the
    formula it replaces.

    Choosing ``kappa_power``
    ------------------------
    Not every operation is conditioning-insensitive, so the caller must state
    which error model applies. Measured on this implementation:

    ============================  =============  ====================
    Operation                     ``kappa_power``  Observed error
    ============================  =============  ====================
    ``expm(logm(A))``             ``0``          ~300 eps, flat in kappa
    ``powm(powm(A, 0.5), 2)``     ``0``          ~26 eps, flat in kappa
    ``powm(powm(A, -1), -1)``     ``1``          ~0.5 * eps * kappa
    ============================  =============  ====================

    The logarithm compresses the spectrum, so log-type round trips do not
    degrade with conditioning. Inversion does not compress it: a relative
    perturbation of the smallest eigenvalue becomes a relative perturbation of
    the largest, and the error grows linearly in kappa. Passing ``0`` for an
    inversion-based identity produces a false failure; passing ``1`` for a
    log-based one produces a bound loose enough to hide a real defect.

    Args:
        a: The input matrix or stack of shape ``(..., n, n)``.
        factor: Slack multiplier. Raise it for compositions of several
            spectral operations, where errors accumulate across steps.
        kappa_power: Exponent on the condition number. Zero for
            conditioning-insensitive operations; one for anything whose error
            is amplified by inversion.

    Returns:
        A relative error tolerance.
    """
    from geoq.geometry.spd import condition_number

    dimension = int(np.asarray(a).shape[-1])
    kappa = float(np.max(condition_number(a)))
    # A condition number below 1 is numerical noise on a near-identity input;
    # clamping keeps the logarithm non-negative so the bound never shrinks
    # below its dimension-driven floor.
    log_kappa = np.log10(max(kappa, 1.0))
    base = factor * EPS * np.sqrt(dimension) * (1.0 + log_kappa)
    return base * max(kappa, 1.0) ** kappa_power


def assert_spd(a: NDArray[np.float64], *, name: str = "result") -> None:
    """Assert that every matrix in a stack satisfies the SPD contract.

    Args:
        a: Array of shape ``(..., n, n)``.
        name: Symbol name used in the assertion message.

    Raises:
        AssertionError: If any matrix is not symmetric positive definite.
    """
    from geoq.geometry.spd import is_spd

    assert bool(np.all(is_spd(a))), f"{name} is not SPD"


def assert_exactly_symmetric(a: NDArray[np.float64], *, name: str = "result") -> None:
    """Assert bitwise symmetry, not approximate symmetry.

    Every function in :mod:`geoq.geometry.spd` symmetrises before returning, so
    downstream code may rely on exact symmetry. Checking this with ``allclose``
    would let a regression through.

    Args:
        a: Array of shape ``(..., n, n)``.
        name: Symbol name used in the assertion message.

    Raises:
        AssertionError: If the array is not bitwise symmetric.
    """
    assert np.array_equal(a, np.swapaxes(a, -1, -2)), f"{name} is not exactly symmetric"