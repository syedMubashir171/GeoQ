"""Tests for :mod:`geoq.core.spd`.

Organising principle
--------------------
These tests are grouped by the *scientific property* being defended, not by
the function under test. A single function may appear in several classes; what
matters is that each class states one claim the framework depends on, so that
a failure names the broken assumption rather than the broken line.

The claims, in order:

1. ``TestRoundTrip`` -- the Block-0 deliverable, as an executable contract.
2. ``TestScaleInvariance`` -- results must not depend on the recording's units.
   This is the property that makes the module usable on real EEG covariances
   whose eigenvalues sit near 1e-12 in microvolts squared.
3. ``TestConditioning`` -- behaviour must degrade gracefully, and predictably,
   as matrices approach singularity.
4. ``TestBatching`` -- vectorised paths must agree exactly with the naive loop.
   Batching is a performance optimisation, and an optimisation that changes
   the answer is a bug.
5. ``TestOutputContract`` -- every returned matrix satisfies the invariants the
   next layer will assume (symmetry, definiteness, dtype).
6. ``TestValidation`` -- every documented failure mode raises the documented
   exception type with an actionable message.
7. ``TestConditioningHelpers`` -- explicit repairs behave as advertised and
   stay opt-in.
8. ``TestRandomSPD`` -- the synthetic generator is itself trustworthy, since
   every other test depends on it.
9. ``TestProperties`` -- Hypothesis property tests over arbitrary SPD inputs,
   which catch the cases hand-written examples miss.

Tolerance policy
----------------
Absolute tolerances are never hard-coded against a matrix's raw magnitude.
Round-trip error is judged relative to the norm of the input, because a matrix
with entries near 1e-12 cannot be compared to one with entries near 1e6 using
the same absolute epsilon. Where a bound must be stated, it is stated as a
multiple of ``eps * kappa(A)``, which is the textbook error bound for a
spectral matrix function.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from geoq.geometry.spd import (
    DEFAULT_PD_RTOL,
    NotPositiveDefiniteError,
    NotSquareError,
    NotSymmetricError,
    SPDError,
    check_spd,
    condition_number,
    eigh_symmetric,
    expm_sym,
    inv_spd,
    invsqrtm_spd,
    is_spd,
    is_symmetric,
    logdet_spd,
    logm_spd,
    powm_spd,
    project_to_spd,
    random_spd,
    regularize_spd,
    shrink_toward_identity,
    spd_funm,
    sqrtm_spd,
    symmetrize,
)
from geoq.testing import DIMENSIONS, relative_error, spectral_error_bound

# --------------------------------------------------------------------------- #
# 1. The Block-0 deliverable
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    """``expm(logm(A)) == A`` and the result remains SPD.

    This is the syllabus's Block-0 deliverable stated as an executable
    contract. Every Riemannian operation in the framework is a composition of
    these two maps, so if this class fails nothing above it can be trusted.
    """

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_expm_inverts_logm(self, n: int, rng: np.random.Generator) -> None:
        a = random_spd(n, rng=rng)
        recovered = expm_sym(logm_spd(a))
        assert relative_error(recovered, a) < spectral_error_bound(a)

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_logm_inverts_expm(self, n: int, rng: np.random.Generator) -> None:
        # Round-tripping from the tangent space is the direction the Exp map
        # is used in during Riemannian interpolation and re-centring.
        symmetric = symmetrize(rng.standard_normal((n, n)))
        recovered = logm_spd(expm_sym(symmetric))
        assert relative_error(recovered, symmetric) < 1e-10

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_round_trip_preserves_definiteness(
        self, n: int, rng: np.random.Generator
    ) -> None:
        a = random_spd(n, rng=rng)
        assert bool(is_spd(expm_sym(logm_spd(a))))

    def test_logm_of_identity_is_zero(self) -> None:
        assert np.allclose(logm_spd(np.eye(5)), 0.0, atol=1e-14)

    def test_logm_matches_scipy_reference(self, rng: np.random.Generator) -> None:
        """Cross-check the spectral implementation against an independent one.

        SciPy's ``logm`` uses inverse scaling-and-squaring on the Schur form --
        a completely different algorithm. Agreement is strong evidence the
        spectral shortcut is correct, not merely self-consistent.
        """
        scipy_linalg = pytest.importorskip("scipy.linalg")
        a = random_spd(6, rng=rng)
        reference = np.real(scipy_linalg.logm(a))
        assert relative_error(logm_spd(a), reference) < 1e-10


# --------------------------------------------------------------------------- #
# 2. Scale invariance -- the microvolt-squared property
# --------------------------------------------------------------------------- #


class TestScaleInvariance:
    """Validation and geometry must not depend on the recording's units.

    EEG covariance eigenvalues land anywhere from 1e-14 to 1e-8 depending on
    whether the data is stored in volts, microvolts, or arbitrary amplifier
    units. A framework that only works in one of those is not usable.
    """

    @pytest.mark.parametrize("exponent", range(-12, 13, 2))
    def test_is_spd_is_scale_invariant(
        self, exponent: int, rng: np.random.Generator
    ) -> None:
        a = random_spd(8, rng=rng, scale=10.0**exponent)
        assert bool(is_spd(a)), f"rejected a valid matrix at scale 1e{exponent}"

    @pytest.mark.parametrize("exponent", range(-12, 13, 4))
    def test_round_trip_is_scale_invariant(
        self, exponent: int, rng: np.random.Generator
    ) -> None:
        a = random_spd(8, rng=rng, scale=10.0**exponent)
        assert relative_error(expm_sym(logm_spd(a)), a) < spectral_error_bound(a)

    def test_condition_number_is_scale_invariant(
        self, rng: np.random.Generator
    ) -> None:
        a = random_spd(10, rng=rng, condition_number=1e6)
        scaled = a * 1e-11
        assert np.isclose(
            float(condition_number(a)), float(condition_number(scaled)), rtol=1e-9
        )

    def test_logm_of_scaled_matrix_shifts_by_log_scale(
        self, rng: np.random.Generator
    ) -> None:
        """``log(cA) = log(A) + log(c) I`` -- the defining scale identity.

        This is what makes the AIRM metric's scale behaviour predictable, and
        it is the reason tangent-space features remain comparable across
        sessions recorded at different gains.
        """
        a = random_spd(7, rng=rng)
        c = 1e-9
        expected = logm_spd(a) + np.log(c) * np.eye(7)
        assert relative_error(logm_spd(c * a), expected) < 1e-10

    def test_regularization_strength_is_scale_invariant(
        self, rng: np.random.Generator
    ) -> None:
        """A trace-relative ridge changes the spectrum identically at any scale.

        With an absolute ridge this test fails, which is precisely why
        :func:`regularize_spd` is trace-relative.
        """
        a = random_spd(6, rng=rng, condition_number=1e4)
        ratio_unit = condition_number(regularize_spd(a, epsilon=1e-3))
        ratio_tiny = condition_number(regularize_spd(a * 1e-12, epsilon=1e-3))
        assert np.isclose(float(ratio_unit), float(ratio_tiny), rtol=1e-6)


# --------------------------------------------------------------------------- #
# 3. Conditioning
# --------------------------------------------------------------------------- #


class TestConditioning:
    """Degradation near singularity must be graceful and predictable."""

    # The admissible range stops below 1 / DEFAULT_PD_RTOL; see
    # test_pd_tolerance_defines_the_maximum_condition_number below.
    @pytest.mark.parametrize("kappa", [1e2, 1e6, 1e10, 1e11])
    def test_round_trip_within_spectral_bound(
        self, kappa: float, rng: np.random.Generator
    ) -> None:
        a = random_spd(8, rng=rng, condition_number=kappa)
        assert relative_error(expm_sym(logm_spd(a)), a) < spectral_error_bound(a)

    def test_pd_tolerance_defines_the_maximum_condition_number(
        self, rng: np.random.Generator
    ) -> None:
        """``kappa = 1 / DEFAULT_PD_RTOL`` is the exact rejection boundary.

        Because the definiteness test is ``w_min > rtol * w_max``, a matrix
        with condition number exactly ``1 / rtol`` fails and anything better
        passes. This is a documented capability limit of the framework, not an
        accident: it tells you the worst-conditioned EEG covariance the
        pipeline will accept before demanding explicit shrinkage.
        """
        limit = 1.0 / DEFAULT_PD_RTOL
        assert bool(is_spd(random_spd(8, rng=rng, condition_number=limit / 10.0)))
        assert not bool(is_spd(random_spd(8, rng=rng, condition_number=limit)))

    @pytest.mark.parametrize("kappa", [1e2, 1e6, 1e10])
    def test_condition_number_is_recovered(
        self, kappa: float, rng: np.random.Generator
    ) -> None:
        a = random_spd(12, rng=rng, condition_number=kappa)
        assert float(condition_number(a)) == pytest.approx(kappa, rel=1e-6)

    def test_near_singular_matrix_is_rejected(self) -> None:
        """A matrix inside the PD tolerance must fail, not silently proceed.

        Rank-deficient covariances arise routinely from average referencing or
        ICA cleaning. Catching them here, with a message naming the cause, is
        far better than a NaN appearing three layers up inside a CV fold.
        """
        a = np.diag([1.0, 1.0, DEFAULT_PD_RTOL / 100.0])
        assert not bool(is_spd(a))
        with pytest.raises(NotPositiveDefiniteError, match="not positive definite"):
            check_spd(a, name="covariance")

    def test_logdet_survives_underflow(self, rng: np.random.Generator) -> None:
        """``det`` underflows for a 22-channel microvolt covariance; ``logdet`` must not.

        This test would be impossible to pass via ``log(det(A))``, which is why
        the implementation sums log-eigenvalues instead.
        """
        # 22 channels at 1e-16 gives det ~ 1e-350, below the float64 minimum
        # subnormal (~5e-324). At the more typical microvolt-squared scale of
        # 1e-12 the determinant is ~1e-259: small, but not yet zero. The
        # margin is thinner than it looks, which is the point.
        a = random_spd(22, rng=rng, scale=1e-16)
        assert np.linalg.det(a) == 0.0  # confirms the underflow is real
        result = float(logdet_spd(a))
        assert np.isfinite(result)
        assert result == pytest.approx(float(np.linalg.slogdet(a)[1]), rel=1e-9)


# --------------------------------------------------------------------------- #
# 4. Batching
# --------------------------------------------------------------------------- #


class TestBatching:
    """Vectorised paths must agree with the naive loop, exactly.

    Batching exists because LOSO with nested inner folds calls these functions
    tens of thousands of times per experiment. An optimisation that perturbs
    the answer would silently change published numbers.
    """

    @pytest.mark.parametrize(
        "func", [logm_spd, sqrtm_spd, invsqrtm_spd, inv_spd, expm_sym]
    )
    def test_batched_matches_loop(self, func, rng: np.random.Generator) -> None:
        batch = random_spd(6, rng=rng, batch=17)
        vectorised = func(batch)
        looped = np.stack([func(m) for m in batch])
        assert np.array_equal(vectorised, looped)

    @pytest.mark.parametrize("batch_shape", [(), (1,), (5,), (3, 4), (2, 3, 4)])
    def test_shapes_propagate(
        self, batch_shape: tuple[int, ...], rng: np.random.Generator
    ) -> None:
        a = random_spd(5, rng=rng, batch=batch_shape or None)
        assert a.shape == (*batch_shape, 5, 5)
        assert logm_spd(a).shape == (*batch_shape, 5, 5)
        assert logdet_spd(a).shape == batch_shape
        assert is_spd(a).shape == batch_shape

    def test_batch_validation_reports_offending_index(
        self, rng: np.random.Generator
    ) -> None:
        """A bad trial in a large batch must be locatable, not just flagged."""
        batch = random_spd(4, rng=rng, batch=10)
        batch[7] = np.diag([1.0, 1.0, 1.0, -1.0])
        with pytest.raises(NotPositiveDefiniteError, match=r"\(7,\)"):
            check_spd(batch, name="covariances")


# --------------------------------------------------------------------------- #
# 5. Output contract
# --------------------------------------------------------------------------- #


class TestOutputContract:
    """Every result satisfies the invariants the next layer assumes."""

    @pytest.mark.parametrize(
        "func", [logm_spd, expm_sym, sqrtm_spd, invsqrtm_spd, inv_spd]
    )
    def test_output_is_exactly_symmetric(self, func, rng: np.random.Generator) -> None:
        """Bitwise symmetry, not approximate.

        ``V f(w) V^T`` breaks exact symmetry at the 1e-16 level through
        round-off. Without the explicit symmetrisation on return, downstream
        validators would reject mathematically valid results.
        """
        result = func(random_spd(9, rng=rng))
        assert np.array_equal(result, result.T)

    @pytest.mark.parametrize("func", [expm_sym, sqrtm_spd, invsqrtm_spd, inv_spd])
    def test_definiteness_preserving_functions_return_spd(
        self, func, rng: np.random.Generator
    ) -> None:
        assert bool(is_spd(func(random_spd(9, rng=rng))))

    def test_outputs_are_float64(self, rng: np.random.Generator) -> None:
        """Float32 loses roughly half the digits of an AIRM geodesic distance."""
        a = random_spd(5, rng=rng).astype(np.float32)
        assert logm_spd(a).dtype == np.float64

    def test_integer_input_is_accepted(self) -> None:
        assert bool(is_spd(np.eye(3, dtype=np.int64)))

    def test_sqrtm_squares_back(self, rng: np.random.Generator) -> None:
        a = random_spd(8, rng=rng)
        root = sqrtm_spd(a)
        assert relative_error(root @ root, a) < spectral_error_bound(a)

    def test_invsqrtm_whitens(self, rng: np.random.Generator) -> None:
        """``A^-1/2 A A^-1/2 = I`` -- the whitening step inside AIRM distance."""
        a = random_spd(8, rng=rng)
        whitener = invsqrtm_spd(a)
        assert relative_error(
            whitener @ a @ whitener, np.eye(8)
        ) < spectral_error_bound(a)

    @pytest.mark.parametrize(
        ("power", "reference"),
        [(0.5, sqrtm_spd), (-0.5, invsqrtm_spd), (-1.0, inv_spd)],
    )
    def test_powm_agrees_with_named_functions(
        self, power: float, reference, rng: np.random.Generator
    ) -> None:
        a = random_spd(7, rng=rng)
        assert np.array_equal(powm_spd(a, power), reference(a))

    def test_powm_zero_gives_identity(self, rng: np.random.Generator) -> None:
        a = random_spd(6, rng=rng)
        assert relative_error(powm_spd(a, 0.0), np.eye(6)) < 1e-12

    def test_eigh_returns_ascending_orthonormal_basis(
        self, rng: np.random.Generator
    ) -> None:
        """The eigenvector convention Paper 4's encoding will depend on."""
        a = random_spd(6, rng=rng)
        eigenvalues, eigenvectors = eigh_symmetric(a)
        assert np.all(np.diff(eigenvalues) >= 0)
        assert relative_error(eigenvectors.T @ eigenvectors, np.eye(6)) < 1e-12
        reconstructed = (eigenvectors * eigenvalues) @ eigenvectors.T
        assert relative_error(reconstructed, a) < spectral_error_bound(a)


# --------------------------------------------------------------------------- #
# 6. Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    """Documented failure modes raise documented exceptions."""

    @pytest.mark.parametrize(
        ("bad", "exc", "pattern"),
        [
            (np.array([1.0, 2.0, 3.0]), NotSquareError, "at least 2 dimensions"),
            (np.zeros((3, 4)), NotSquareError, "square"),
            (np.zeros((2, 3, 4)), NotSquareError, "square"),
            (np.array([[1.0, 2.0], [0.0, 1.0]]), NotSymmetricError, "not symmetric"),
            (
                np.array([[1.0, 0.0], [0.0, -1.0]]),
                NotPositiveDefiniteError,
                "not positive definite",
            ),
            (np.zeros((3, 3)), NotPositiveDefiniteError, "not positive definite"),
            (np.array([[1.0, np.nan], [np.nan, 1.0]]), SPDError, "non-finite"),
            (np.array([[np.inf, 0.0], [0.0, 1.0]]), SPDError, "non-finite"),
        ],
    )
    def test_invalid_input_raises(self, bad, exc, pattern: str) -> None:
        with pytest.raises(exc, match=pattern):
            check_spd(bad, name="covariance")

    def test_error_message_names_the_argument(self) -> None:
        """A failure inside fold 7 of subject 4 must say *which* array broke."""
        with pytest.raises(SPDError, match="tangent_features"):
            check_spd(np.zeros((2, 2)), name="tangent_features")

    def test_pd_error_reports_eigenvalues_and_remedy(self) -> None:
        """The message must be actionable, not merely correct."""
        with pytest.raises(NotPositiveDefiniteError) as excinfo:
            check_spd(np.diag([1.0, 0.0]), name="cov")
        message = str(excinfo.value)
        assert "lambda_min" in message
        assert "regularize_spd" in message

    def test_nan_message_suggests_likely_causes(self) -> None:
        with pytest.raises(SPDError, match="flat"):
            check_spd(np.full((3, 3), np.nan))

    def test_symmetry_check_is_relative_not_absolute(self) -> None:
        """Asymmetry must be judged against the matrix's own scale.

        The two matrices below differ only by a factor of 1e10. A tolerance
        that calls one symmetric and the other not is unit-dependent, and
        would make validation results depend on the amplifier.
        """
        base = np.array([[1.0, 0.5], [0.5 + 1e-14, 1.0]])
        assert bool(is_symmetric(base))
        assert bool(is_symmetric(base * 1e10))
        assert bool(is_symmetric(base * 1e-10))

    def test_gross_asymmetry_is_caught_at_every_scale(self) -> None:
        base = np.array([[1.0, 0.9], [0.1, 1.0]])
        for scale in (1e-10, 1.0, 1e10):
            assert not bool(is_symmetric(base * scale))

    def test_validate_false_skips_checks(self) -> None:
        """The hot-loop escape hatch genuinely bypasses validation."""
        asymmetric = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(NotSymmetricError):
            logm_spd(asymmetric)
        result = logm_spd(asymmetric, validate=False)
        assert np.all(np.isfinite(result))

    def test_zero_matrix_counts_as_symmetric(self) -> None:
        """Guards the division by ``||A||_F`` in the symmetry test."""
        assert bool(is_symmetric(np.zeros((4, 4))))


# --------------------------------------------------------------------------- #
# 7. Explicit conditioning helpers
# --------------------------------------------------------------------------- #


class TestConditioningHelpers:
    """Repairs are opt-in, effective, and loud when they fire."""

    def test_regularize_makes_singular_matrix_spd(self) -> None:
        singular = np.diag([1.0, 1.0, 0.0])
        assert not bool(is_spd(singular))
        assert bool(is_spd(regularize_spd(singular, epsilon=1e-6)))

    def test_regularize_reduces_condition_number(
        self, rng: np.random.Generator
    ) -> None:
        a = random_spd(8, rng=rng, condition_number=1e10)
        assert condition_number(regularize_spd(a, epsilon=1e-4)) < condition_number(a)

    def test_regularize_with_zero_epsilon_is_identity(
        self, rng: np.random.Generator
    ) -> None:
        a = random_spd(5, rng=rng)
        assert np.array_equal(regularize_spd(a, epsilon=0.0), a)

    def test_shrinkage_preserves_trace(self, rng: np.random.Generator) -> None:
        """The property that distinguishes shrinkage from a ridge.

        A ridge inflates total signal power; shrinkage redistributes it. On
        short epochs the difference shows up directly in tangent-space feature
        magnitudes and therefore in classifier scale sensitivity.
        """
        a = random_spd(9, rng=rng, condition_number=1e6)
        for alpha in (0.0, 0.1, 0.5, 1.0):
            shrunk = shrink_toward_identity(a, alpha=alpha)
            assert np.trace(shrunk) == pytest.approx(np.trace(a), rel=1e-12)

    def test_shrinkage_endpoints(self, rng: np.random.Generator) -> None:
        a = random_spd(6, rng=rng)
        assert np.allclose(shrink_toward_identity(a, alpha=0.0), a)
        fully_shrunk = shrink_toward_identity(a, alpha=1.0)
        assert float(condition_number(fully_shrunk)) == pytest.approx(1.0)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_shrinkage_rejects_out_of_range_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            shrink_toward_identity(np.eye(3), alpha=alpha)

    def test_regularize_rejects_negative_epsilon(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            regularize_spd(np.eye(3), epsilon=-1.0)

    def test_projection_warns_when_it_modifies_data(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A repair that fires is a data-quality finding and must be logged.

        Silent repair is how unreproducible preprocessing enters a pipeline;
        the warning is what puts it into the run's provenance record.
        """
        with caplog.at_level(logging.WARNING, logger="geoq.geometry.spd"):
            repaired = project_to_spd(np.diag([1.0, 1.0, -0.5]), floor=1e-6)
        assert bool(is_spd(repaired))
        assert "floored" in caplog.text

    def test_projection_is_silent_on_valid_input(
        self, rng: np.random.Generator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="geoq.geometry.spd"):
            project_to_spd(random_spd(5, rng=rng, condition_number=10.0))
        assert caplog.text == ""

    def test_no_function_repairs_silently(self, rng: np.random.Generator) -> None:
        """The central policy: nothing conditions data unless asked."""
        singular = np.diag([1.0, 1.0, 0.0])
        for func in (logm_spd, sqrtm_spd, invsqrtm_spd, inv_spd):
            with pytest.raises(NotPositiveDefiniteError):
                func(singular)


# --------------------------------------------------------------------------- #
# 8. The synthetic generator itself
# --------------------------------------------------------------------------- #


class TestRandomSPD:
    """Every other test trusts this generator, so it is tested first-class."""

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_output_is_spd(self, n: int, rng: np.random.Generator) -> None:
        assert bool(np.all(is_spd(random_spd(n, rng=rng, batch=20))))

    def test_same_seed_gives_identical_output(self) -> None:
        a = random_spd(7, rng=np.random.default_rng(42), batch=3)
        b = random_spd(7, rng=np.random.default_rng(42), batch=3)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = random_spd(7, rng=np.random.default_rng(1))
        b = random_spd(7, rng=np.random.default_rng(2))
        assert not np.allclose(a, b)

    def test_orthogonal_basis_is_haar_unbiased(self) -> None:
        """QR without sign correction biases the eigenvector distribution.

        LAPACK's ``R`` has arbitrary diagonal signs; leaving them uncorrected
        makes the first eigenvector component systematically positive. That
        would quietly bias any test of an eigenvector-dependent encoding --
        exactly what Paper 4 builds.
        """
        generator = np.random.default_rng(7)
        samples = random_spd(4, rng=generator, batch=4000, condition_number=100.0)
        _, eigenvectors = np.linalg.eigh(samples)
        mean_first_component = float(np.mean(eigenvectors[:, 0, 0]))
        assert abs(mean_first_component) < 0.05

    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
    def test_scale_argument_shifts_spectrum(
        self, scale: float, rng: np.random.Generator
    ) -> None:
        a = random_spd(5, rng=rng, condition_number=10.0, scale=scale)
        assert float(np.max(np.linalg.eigvalsh(a))) == pytest.approx(
            10.0 * scale, rel=1e-9
        )

    def test_scalar_dimension_edge_case(self, rng: np.random.Generator) -> None:
        a = random_spd(1, rng=rng, condition_number=1e6)
        assert a.shape == (1, 1)
        assert bool(is_spd(a))

    @pytest.mark.parametrize(
        ("kwargs", "pattern"),
        [
            ({"n": 0}, "n must be"),
            ({"n": 3, "condition_number": 0.5}, "condition_number"),
            ({"n": 3, "scale": 0.0}, "scale"),
        ],
    )
    def test_invalid_arguments_raise(
        self, kwargs: dict, pattern: str, rng: np.random.Generator
    ) -> None:
        with pytest.raises(ValueError, match=pattern):
            random_spd(rng=rng, **kwargs)


# --------------------------------------------------------------------------- #
# 9. Property-based tests
# --------------------------------------------------------------------------- #


@st.composite
def spd_matrices(draw, min_dim: int = 2, max_dim: int = 10):
    """Hypothesis strategy generating SPD matrices across dimension and scale.

    Drawing the seed rather than the entries directly keeps every generated
    matrix exactly SPD, so the strategy explores the valid input space instead
    of wasting draws on rejected candidates.
    """
    n = draw(st.integers(min_value=min_dim, max_value=max_dim))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    log_kappa = draw(st.floats(min_value=0.0, max_value=8.0))
    log_scale = draw(st.floats(min_value=-10.0, max_value=10.0))
    return random_spd(
        n,
        rng=np.random.default_rng(seed),
        condition_number=10.0**log_kappa,
        scale=10.0**log_scale,
    )


class TestProperties:
    """Algebraic identities that must hold for arbitrary SPD inputs.

    Hand-written examples test the cases the author thought of. These test the
    cases the author did not.
    """

    @settings(
        max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices())
    def test_logm_expm_are_mutual_inverses(self, a) -> None:
        assert relative_error(expm_sym(logm_spd(a)), a) < spectral_error_bound(a)

    @settings(
        max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices())
    def test_all_functions_preserve_symmetry_exactly(self, a) -> None:
        for func in (logm_spd, sqrtm_spd, invsqrtm_spd, inv_spd):
            result = func(a)
            assert np.array_equal(result, result.T)

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices())
    def test_logdet_equals_slogdet(self, a) -> None:
        expected = float(np.linalg.slogdet(a)[1])
        # Absolute comparison is right here: logdet is already a logarithm, so
        # its natural error scale is additive, not multiplicative.
        assert abs(float(logdet_spd(a)) - expected) < 1e-6 * max(1.0, abs(expected))

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices())
    def test_inverse_has_reciprocal_spectrum(self, a) -> None:
        """``kappa(A^-1) == kappa(A)`` -- used when validating AIRM symmetry."""
        assert float(condition_number(inv_spd(a))) == pytest.approx(
            float(condition_number(a)), rel=1e-6
        )

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices(max_dim=8), p=st.floats(min_value=-2.0, max_value=2.0))
    def test_power_law_composition(self, a, p: float) -> None:
        """``(A^p)^(1/p) = A``, within the range where the identity survives.

        The round trip passes through ``A^p``, whose condition number is
        ``kappa ** |p|``, and back through ``A^(1/p)``, whose condition number
        is ``kappa ** |1/p|``. The worse of the two governs how much relative
        precision is destroyed, so the amplification exponent is
        ``max(|p|, 1/|p|)``. Measured at ``p = -2`` this model is tight:
        ``error / (eps * kappa ** 2)`` sits at 0.41-0.44 across two decades of
        conditioning, whereas ``error / (eps * kappa)`` runs away from 44 to
        4107 over the same range -- which is why a linear-in-kappa bound was
        wrong here even though it is correct for a single inversion.

        Two guards, for two different reasons:

        * ``|p| < 0.1`` is excluded because ``1/p`` then exceeds 10 and the
          intermediate is a power beyond anything this framework computes.
        * An effective condition number above ``1e8`` is excluded because
          fewer than eight significant digits survive the round trip there.
          Bounding such a case would test nothing: the tolerance would exceed
          unity and any implementation whatsoever would pass. Skipping is
          honest; a vacuous assertion is not.
        """
        if abs(p) < 0.1:
            return
        exponent = max(abs(p), 1.0 / abs(p))
        if float(condition_number(a)) ** exponent > 1e8:
            return
        assert relative_error(
            powm_spd(powm_spd(a, p), 1.0 / p), a
        ) < spectral_error_bound(a, kappa_power=exponent)

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices())
    def test_generated_matrices_always_validate(self, a) -> None:
        assert bool(is_spd(a))
        check_spd(a)

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices())
    def test_spd_funm_with_identity_returns_input(self, a) -> None:
        """The primitive itself is faithful, independent of any wrapper."""
        assert relative_error(
            spd_funm(a, lambda w: w, require_pd=True), a
        ) < spectral_error_bound(a)