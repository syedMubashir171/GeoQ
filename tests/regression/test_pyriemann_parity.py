"""Parity of :mod:`geoq.geometry` against pyRiemann.

Why this lives in ``tests/regression/`` and not beside the unit tests
--------------------------------------------------------------------
Matching a reference implementation and *being correct* are different claims.
pyRiemann could change a convention, deprecate a module, or carry a bug of its
own; none of those should turn the geometry layer's own test suite red. The
property tests in ``tests/geometry/`` stand on the mathematics -- metric
axioms, affine invariance, isometry of transport -- and hold regardless of
what any external package computes. This file adds the independent
cross-check on top.

It is also the executable form of the syllabus's Block-1 success criterion:
an implementation written from scratch is understood once it reproduces the
established library's numbers.

Every test here is marked ``regression`` and skips cleanly when pyRiemann is
absent, which is what keeps ``pip install -e ".[dev]"`` a sufficient
environment for developing the geometry layer.

Import path
-----------
pyRiemann 0.12 deprecated ``pyriemann.utils.*`` in favour of
``pyriemann.geometry.*``, with removal scheduled for 0.14. The helper below
prefers the new path and falls back to the old one, so this file keeps working
across the transition instead of breaking on an unrelated dependency upgrade.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from geoq.geometry.riemannian import (
    distance_airm,
    distance_euclid,
    distance_logeuclid,
    distance_stein,
    frechet_mean,
    geodesic,
    log_map,
    mean_logeuclid,
)
from geoq.geometry.spd import expm_sym, invsqrtm_spd, logm_spd, random_spd, sqrtm_spd
from geoq.testing import relative_error

pytestmark = pytest.mark.regression

#  Dimensions matching the datasets in the roadmap: 3 is a toy case, 8 a small
#  montage, 22 the full BCI Competition IV 2a channel count.
PARITY_DIMENSIONS = [3, 8, 22]

#  Parity is asserted at a fixed relative tolerance rather than through
#  spectral_error_bound. The quantity being bounded is the gap between two
#  independent implementations of the same formula, not the error of one of
#  them against exact arithmetic, so the conditioning-aware bound does not
#  apply. Inputs are drawn well-conditioned for the same reason: a parity
#  failure should mean the formulas disagree, not that both ran out of digits.
PARITY_RTOL = 1e-9
PARITY_CONDITION_NUMBER = 1e3


def _submodule(name: str) -> ModuleType:
    """Import a pyRiemann submodule, preferring the post-0.12 location.

    Args:
        name: Submodule name, e.g. ``"distance"``.

    Returns:
        The imported module.
    """
    pytest.importorskip(
        "pyriemann", reason="requires the 'eeg' extra: pip install -e '.[eeg]'"
    )
    from importlib import import_module

    try:
        return import_module(f"pyriemann.geometry.{name}")
    except ImportError:
        return import_module(f"pyriemann.utils.{name}")


@pytest.fixture(scope="module")
def reference() -> dict[str, ModuleType]:
    """The pyRiemann submodules used for comparison."""
    return {
        name: _submodule(name)
        for name in ("distance", "mean", "geodesic", "tangentspace")
    }


@pytest.fixture
def well_conditioned(rng: np.random.Generator):
    """A factory for well-conditioned SPD matrices at a chosen dimension."""

    def _make(n: int, batch: int | None = None):
        return random_spd(
            n, rng=rng, batch=batch, condition_number=PARITY_CONDITION_NUMBER
        )

    return _make


class TestDistanceParity:
    """Each distance reproduces pyRiemann's value."""

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_airm(self, n: int, well_conditioned, reference) -> None:
        a, b = well_conditioned(n), well_conditioned(n)
        assert float(distance_airm(a, b)) == pytest.approx(
            reference["distance"].distance_riemann(a, b), rel=PARITY_RTOL
        )

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_logeuclid(self, n: int, well_conditioned, reference) -> None:
        a, b = well_conditioned(n), well_conditioned(n)
        assert float(distance_logeuclid(a, b)) == pytest.approx(
            reference["distance"].distance_logeuclid(a, b), rel=PARITY_RTOL
        )

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_stein(self, n: int, well_conditioned, reference) -> None:
        """PyRiemann names this ``distance_logdet``; it is the same divergence."""
        a, b = well_conditioned(n), well_conditioned(n)
        assert float(distance_stein(a, b)) == pytest.approx(
            reference["distance"].distance_logdet(a, b), rel=PARITY_RTOL
        )

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_euclid(self, n: int, well_conditioned, reference) -> None:
        a, b = well_conditioned(n), well_conditioned(n)
        assert float(distance_euclid(a, b)) == pytest.approx(
            reference["distance"].distance_euclid(a, b), rel=PARITY_RTOL
        )

    def test_agreement_holds_across_many_random_pairs(
        self, well_conditioned, reference
    ) -> None:
        """A single lucky pair proves little; this checks the whole batch."""
        a = well_conditioned(8, batch=50)
        b = well_conditioned(8, batch=50)
        mine = distance_airm(a, b)
        theirs = np.array(
            [
                reference["distance"].distance_riemann(x, y)
                for x, y in zip(a, b, strict=True)
            ]
        )
        assert np.allclose(mine, theirs, rtol=PARITY_RTOL)


class TestMeanParity:
    """Frechet means agree with pyRiemann.

    The AIRM comparison is looser than the others by design. Both sides run an
    iterative solver to their own tolerance and stop at slightly different
    points near the same optimum, so exact agreement is not the right
    expectation. That the two independently-implemented iterations land within
    1e-8 of each other is the meaningful claim.
    """

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_airm_mean(self, n: int, well_conditioned, reference) -> None:
        x = well_conditioned(n, batch=40)
        assert relative_error(frechet_mean(x), reference["mean"].mean_riemann(x)) < 1e-8

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_logeuclid_mean(self, n: int, well_conditioned, reference) -> None:
        x = well_conditioned(n, batch=40)
        assert (
            relative_error(mean_logeuclid(x), reference["mean"].mean_logeuclid(x))
            < PARITY_RTOL
        )

    def test_mean_of_ill_conditioned_set(self, rng, reference) -> None:
        """The regime real EEG covariances actually occupy.

        Short epochs on a 22-channel montage routinely produce condition
        numbers near 1e6. The tolerance is relaxed to match the precision
        genuinely available there, rather than pretending the well-conditioned
        bound still applies.
        """
        x = random_spd(22, rng=rng, batch=30, condition_number=1e6)
        assert relative_error(frechet_mean(x), reference["mean"].mean_riemann(x)) < 1e-5


class TestGeodesicParity:
    """Geodesic interpolation matches at every parameter value."""

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_airm_geodesic(self, t: float, well_conditioned, reference) -> None:
        a, b = well_conditioned(8), well_conditioned(8)
        assert (
            relative_error(
                geodesic(a, b, t),
                reference["geodesic"].geodesic_riemann(a, b, alpha=t),
            )
            < PARITY_RTOL
        )


class TestTangentSpaceParity:
    """The tangent-space map underpinning the TS+LDA baseline.

    pyRiemann's ``tangent_space`` returns the *whitened, vectorised* upper
    triangle, whereas :func:`geoq.geometry.riemannian.log_map` returns the
    tangent vector as a matrix. The two are related by whitening at the
    reference point, so this test reconstructs pyRiemann's convention rather
    than comparing incompatible objects -- and in doing so pins down exactly
    which convention this framework uses, which is the detail that silently
    breaks reproductions of published tangent-space results.
    """

    def test_log_map_reproduces_pyriemann_tangent_vectors(
        self, well_conditioned, reference
    ) -> None:
        reference_point = well_conditioned(6)
        x = well_conditioned(6, batch=15)

        theirs = reference["tangentspace"].tangent_space(x, reference_point)
        whitener = invsqrtm_spd(reference_point)
        mine = whitener @ log_map(reference_point, x) @ whitener

        n = 6
        rows, cols = np.triu_indices(n)
        # pyRiemann scales strictly-upper entries by sqrt(2) so that the
        # Euclidean norm of the vector equals the Frobenius norm of the matrix.
        coefficients = np.where(rows == cols, 1.0, np.sqrt(2.0))
        mine_vectorised = mine[:, rows, cols] * coefficients

        assert np.allclose(mine_vectorised, theirs, rtol=1e-8, atol=1e-12)

    def test_whitened_log_map_equals_logm_of_whitened(self, well_conditioned) -> None:
        """Internal consistency of the identity the previous test relies on."""
        reference_point = well_conditioned(5)
        target = well_conditioned(5)
        whitener = invsqrtm_spd(reference_point)
        left = whitener @ log_map(reference_point, target) @ whitener
        right = logm_spd(whitener @ target @ whitener)
        assert relative_error(left, right) < 1e-10


def _scipy_reference(func: Callable, a: np.ndarray) -> np.ndarray:
    """Evaluate a SciPy matrix function, tolerating its self-reported error.

    SciPy's ``logm`` and ``sqrtm`` use inverse scaling-and-squaring on the
    Schur form and emit a ``RuntimeWarning`` when their internal residual
    estimate exceeds a threshold -- around 3e-13 for a well-conditioned 8x8
    SPD matrix here. The project turns warnings into errors, which is right
    everywhere else, but this particular warning is the *reference* reporting
    its own inaccuracy, not a defect in the code under test. The spectral
    implementation has no comparable residual because it never forms a Schur
    factorisation.

    Suppression is scoped to this one call so that a warning raised anywhere
    else in the test still fails the suite.

    Args:
        func: The SciPy matrix function.
        a: Input matrix.

    Returns:
        The real part of the result.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="logm result may be inaccurate", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", message="sqrtm result may be inaccurate", category=RuntimeWarning
        )
        return np.real(func(a))


class TestPrimitiveParity:
    """The spd-layer primitives, against SciPy rather than pyRiemann.

    Included here because it is the same kind of claim: an independent
    algorithm computing the same object. SciPy's ``logm`` and ``sqrtm`` use
    Schur-based methods, so agreement is evidence the spectral shortcut is
    correct rather than merely self-consistent.
    """

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_logm_against_scipy(self, n: int, well_conditioned) -> None:
        scipy_linalg = pytest.importorskip("scipy.linalg")
        a = well_conditioned(n)
        reference_value = _scipy_reference(scipy_linalg.logm, a)
        assert relative_error(logm_spd(a), reference_value) < 1e-10

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_sqrtm_against_scipy(self, n: int, well_conditioned) -> None:
        scipy_linalg = pytest.importorskip("scipy.linalg")
        a = well_conditioned(n)
        reference_value = _scipy_reference(scipy_linalg.sqrtm, a)
        assert relative_error(sqrtm_spd(a), reference_value) < 1e-10

    @pytest.mark.parametrize("n", PARITY_DIMENSIONS)
    def test_expm_against_scipy(self, n: int, well_conditioned) -> None:
        scipy_linalg = pytest.importorskip("scipy.linalg")
        symmetric = logm_spd(well_conditioned(n))
        assert relative_error(expm_sym(symmetric), scipy_linalg.expm(symmetric)) < 1e-10