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


def spectral_error_bound(a: NDArray[np.float64], *, factor: float = 100.0) -> float:
    """Admissible relative error for a spectral matrix function applied to ``a``.

    The accuracy of ``V f(w) V^T`` is governed by the conditioning of the
    eigendecomposition, so the tolerance must scale with the condition number.
    A hard-coded ``1e-12`` would be either vacuous for well-conditioned inputs
    or spuriously red for ill-conditioned ones.

    Args:
        a: The input matrix or stack of shape ``(..., n, n)``.
        factor: Slack over the textbook ``eps * kappa`` bound, covering error
            accumulated across the decomposition, the spectral scaling, and
            the reconstruction.

    Returns:
        A relative error tolerance.
    """
    from geoq.geometry.spd import condition_number

    kappa = float(np.max(condition_number(a)))
    return factor * EPS * kappa


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