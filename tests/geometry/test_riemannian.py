"""Tests for :mod:`geoq.geometry.riemannian`.

Organising principle
--------------------
Grouped by the geometric claim being defended, not by the function under test.
A failure should name the broken assumption, not the broken line.

Parity against pyRiemann lives in ``tests/regression/test_pyriemann_parity.py``
and is deliberately separate. Matching a reference implementation proves the
formulas were transcribed correctly; it does not prove they are *geometry*.
The tests here check the properties that make these objects a Riemannian
manifold at all -- metric axioms, affine invariance, isometry of transport,
the minimising property of the mean -- and those hold, or fail, independently
of what any external library computes.

Tolerance policy
----------------
Tolerances come from :func:`geoq.testing.spectral_error_bound`, which requires
the caller to declare an error model via ``kappa_power``. Distances and maps
are built from matrix logarithms of whitened products, which compress the
spectrum and are therefore conditioning-insensitive: ``kappa_power=0``. The
one place that is not true is noted where it occurs.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from geoq.geometry.riemannian import (
    METRICS,
    MeanResult,
    distance,
    distance_airm,
    distance_euclid,
    distance_logeuclid,
    distance_stein,
    exp_map,
    frechet_mean,
    geodesic,
    log_map,
    mean_euclid,
    mean_logeuclid,
    pairwise_distances,
    parallel_transport,
    whiten,
)
from geoq.geometry.spd import (
    NotPositiveDefiniteError,
    NotSymmetricError,
    invsqrtm_spd,
    is_spd,
    random_spd,
)
from geoq.testing import relative_error, spd_matrices, spectral_error_bound

#  Stein is a genuine metric but has no geodesic and no implemented mean, so
#  the metric-axiom tests include it while the geodesic tests do not.
CURVED_METRICS = ["airm", "logeuclid", "stein"]
ALL_METRICS = [*CURVED_METRICS, "euclid"]
GEODESIC_METRICS = ["airm", "logeuclid", "euclid"]


#  Every affine-invariant quantity here is computed by whitening: forming
#  ``A^-1/2 B A^-1/2``. Whitening inverts, and inversion amplifies relative
#  error by the condition number -- once for the whitener and again through
#  the whitened matrix's own spectrum. The effective conditioning of an AIRM
#  computation is therefore ``kappa ** 2``, not ``kappa``, which is why these
#  tests declare ``kappa_power=2.0`` where the spd-layer tests declare zero.
#
#  Measured asymmetry ``|d(a,b) - d(b,a)| / d`` for matrices of condition
#  ``kappa``: 284 eps at 1e2, 4.8e5 eps at 1e4, 4.7e13 eps at 1e8. The last is
#  a one-percent relative error. This is a property of the mathematics, not a
#  defect, and it is the reason a covariance pipeline on short EEG epochs
#  needs explicit shrinkage rather than raw sample covariances.
AIRM_KAPPA_POWER = 2.0

#  Property tests that assert a precise AIRM identity draw from a restricted
#  conditioning range. At kappa = 1e4 the effective conditioning is 1e8 and a
#  meaningful tolerance still exists; at the strategy default of 1e8 the
#  effective conditioning is 1e16 and no tolerance below unity is possible, so
#  the assertion would pass for any implementation whatsoever. The limit
#  itself is asserted separately in TestNumericalLimits.
AIRM_PROPERTY_MAX_LOG_KAPPA = 4.0

#  The Stein distance takes a square root of a difference of log-determinants.
#  For two nearly identical matrices that difference is pure cancellation at
#  the 1e-16 level, and the square root lifts it to roughly 1e-6 -- measured
#  peak excess over AIRM is ~125 * sqrt(eps) across n in [2, 22] and ten
#  orders of eigenvalue scale, and it does not shrink with dimension.
#
#  Stein therefore has an absolute resolution floor the other metrics lack:
#  below ~1e-6 it reports noise, not distance. Two consequences are encoded
#  below. Identity of indiscernibles still holds tightly, because the clamp on
#  the radicand makes d(A, A) exactly zero. But an inequality *between* Stein
#  and AIRM is only meaningful once the distance clears the floor, so the
#  property test skips below it rather than widening its tolerance to 1e-5 --
#  a bound that loose would be satisfied by almost any implementation and
#  would test nothing.
STEIN_SELF_DISTANCE_TOL = 1e-9
STEIN_MEANINGFUL_DISTANCE = 1e-5


def airm_tolerance(*matrices: np.ndarray, factor: float = 250.0) -> float:
    """Relative tolerance for a scalar derived from an AIRM computation.

    Takes the worst tolerance across all matrices involved, since the error is
    governed by the worst-conditioned input rather than by any one of them.

    Args:
        *matrices: The SPD matrices entering the computation.
        factor: Slack multiplier passed through to the bound.

    Returns:
        A relative tolerance.
    """
    return max(
        spectral_error_bound(m, factor=factor, kappa_power=AIRM_KAPPA_POWER)
        for m in matrices
    )


def riemannian_norm(base: np.ndarray, tangent: np.ndarray) -> float:
    """Norm of a tangent vector under the metric at ``base``.

    Defined as ``||base^-1/2 V base^-1/2||_F``. Tangent vectors at different
    base points are not comparable in the Frobenius norm; this is the norm
    that the geometry actually induces, and the one parallel transport is
    required to preserve.

    Args:
        base: SPD matrix of shape ``(n, n)``.
        tangent: Symmetric matrix of shape ``(n, n)``.

    Returns:
        The metric-induced norm.
    """
    whitener = invsqrtm_spd(base)
    return float(np.linalg.norm(whitener @ tangent @ whitener))


# --------------------------------------------------------------------------- #
# 1. Metric axioms
# --------------------------------------------------------------------------- #


class TestMetricAxioms:
    """Every supported metric must actually be a metric.

    Checked directly rather than assumed. A quantity that violates the
    triangle inequality is not a distance, and any kernel or nearest-centroid
    classifier built on it inherits behaviour that no amount of
    cross-validation will explain.
    """

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_non_negative(self, metric: str, rng: np.random.Generator) -> None:
        a = random_spd(6, rng=rng, batch=20)
        b = random_spd(6, rng=rng, batch=20)
        assert np.all(distance(a, b, metric=metric) >= 0.0)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_identity_of_indiscernibles(
        self, metric: str, rng: np.random.Generator
    ) -> None:
        a = random_spd(6, rng=rng, batch=10)
        assert np.all(distance(a, a, metric=metric) < 1e-7)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_positivity_for_distinct_points(
        self, metric: str, rng: np.random.Generator
    ) -> None:
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        assert float(distance(a, b, metric=metric)) > 1e-6

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_symmetry(self, metric: str, rng: np.random.Generator) -> None:
        a = random_spd(6, rng=rng, batch=15)
        b = random_spd(6, rng=rng, batch=15)
        forward = distance(a, b, metric=metric)
        backward = distance(b, a, metric=metric)
        assert np.allclose(forward, backward, rtol=1e-10)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_triangle_inequality(self, metric: str, rng: np.random.Generator) -> None:
        a = random_spd(5, rng=rng, batch=60)
        b = random_spd(5, rng=rng, batch=60)
        c = random_spd(5, rng=rng, batch=60)
        direct = distance(a, c, metric=metric)
        detour = distance(a, b, metric=metric) + distance(b, c, metric=metric)
        assert np.all(direct <= detour * (1.0 + 1e-9))


# --------------------------------------------------------------------------- #
# 2. Invariance -- the property that separates the metrics
# --------------------------------------------------------------------------- #


class TestInvariance:
    """Which congruences leave each metric unchanged.

    This is not a formality. Riemannian re-centring for cross-subject transfer
    works *because* AIRM is invariant to arbitrary invertible congruence: a
    subject's electrode impedance and head geometry act as an unknown ``W``,
    and an affine-invariant distance is blind to it. Log-Euclidean is invariant
    only to orthogonal congruence, so any transfer claim resting on it is
    resting on a weaker guarantee, and the paper must say so.
    """

    def test_airm_is_affine_invariant(self, rng: np.random.Generator) -> None:
        a = random_spd(7, rng=rng)
        b = random_spd(7, rng=rng)
        # A general invertible matrix, not orthogonal: this is the case that
        # distinguishes AIRM from every other metric here.
        w = rng.standard_normal((7, 7))
        transformed = float(distance_airm(w @ a @ w.T, w @ b @ w.T))
        original = float(distance_airm(a, b))
        assert relative_error(
            np.array(transformed), np.array(original)
        ) < spectral_error_bound(a, factor=1e4)

    def test_logeuclid_is_not_affine_invariant(self, rng: np.random.Generator) -> None:
        """The negative case, asserted rather than assumed.

        If this test ever passes, either the implementation collapsed to AIRM
        or the congruence was accidentally orthogonal. Both are silent errors
        that would invalidate any claimed distinction between the metrics.
        """
        a = random_spd(7, rng=rng)
        b = random_spd(7, rng=rng)
        w = rng.standard_normal((7, 7))
        transformed = float(distance_logeuclid(w @ a @ w.T, w @ b @ w.T))
        original = float(distance_logeuclid(a, b))
        assert abs(transformed - original) > 0.01 * original

    @pytest.mark.parametrize("metric", ["airm", "logeuclid", "euclid"])
    def test_orthogonal_invariance(self, metric: str, rng: np.random.Generator) -> None:
        """All three are invariant to rotation of the sensor basis.

        Physically: relabelling or rigidly rotating electrodes must not change
        any distance. A metric failing this would make results depend on
        channel ordering.
        """
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        rotated = float(distance(q @ a @ q.T, q @ b @ q.T, metric=metric))
        assert rotated == pytest.approx(float(distance(a, b, metric=metric)), rel=1e-9)

    def test_airm_is_scale_invariant_under_joint_scaling(
        self, rng: np.random.Generator
    ) -> None:
        """``d(cA, cB) = d(A, B)``: amplifier gain must not change distances."""
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        for c in (1e-12, 1e-6, 1e6):
            assert float(distance_airm(c * a, c * b)) == pytest.approx(
                float(distance_airm(a, b)), rel=1e-9
            )

    def test_airm_inversion_invariance(self, rng: np.random.Generator) -> None:
        """``d(A^-1, B^-1) = d(A, B)`` -- the whitened spectrum merely inverts."""
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        assert float(
            distance_airm(np.linalg.inv(a), np.linalg.inv(b))
        ) == pytest.approx(float(distance_airm(a, b)), rel=1e-8)


# --------------------------------------------------------------------------- #
# 3. Exp / Log maps
# --------------------------------------------------------------------------- #


class TestMaps:
    """The maps between the manifold and its tangent spaces."""

    def test_round_trip(self, rng: np.random.Generator) -> None:
        p = random_spd(8, rng=rng)
        q = random_spd(8, rng=rng, batch=12)
        assert relative_error(exp_map(p, log_map(p, q)), q) < spectral_error_bound(
            q, factor=1e3
        )

    def test_log_at_own_base_point_is_zero(self, rng: np.random.Generator) -> None:
        p = random_spd(8, rng=rng)
        assert float(np.abs(log_map(p, p)).max()) < 1e-9 * float(np.abs(p).max())

    def test_exp_of_zero_returns_base_point(self, rng: np.random.Generator) -> None:
        p = random_spd(8, rng=rng)
        assert relative_error(exp_map(p, np.zeros((8, 8))), p) < 1e-12

    def test_tangent_norm_equals_geodesic_distance(
        self, rng: np.random.Generator
    ) -> None:
        """``||Log_P(Q)||_P = d(P, Q)``.

        The defining consistency between the metric and the maps. If it fails,
        tangent-space features are not a faithful linearisation of the
        manifold and every TS+LDA result built on them is measuring something
        other than geodesic structure.
        """
        p = random_spd(7, rng=rng)
        q = random_spd(7, rng=rng)
        assert riemannian_norm(p, log_map(p, q)) == pytest.approx(
            float(distance_airm(p, q)), rel=1e-9
        )

    def test_tangent_vectors_may_be_indefinite(self, rng: np.random.Generator) -> None:
        """A tangent vector is a displacement, not a covariance.

        Guards against a plausible-looking future "fix" that validates tangent
        vectors as SPD, which would break the map for any Q with an eigenvalue
        below the base point's.
        """
        p = np.eye(4)
        q = np.diag([0.1, 0.2, 5.0, 8.0])
        eigenvalues = np.linalg.eigvalsh(log_map(p, q))
        assert eigenvalues[0] < 0 < eigenvalues[-1]

    def test_exp_map_accepts_indefinite_tangent(self, rng: np.random.Generator) -> None:
        p = random_spd(5, rng=rng)
        tangent = np.diag([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert bool(is_spd(exp_map(p, tangent)))

    def test_whiten_moves_reference_to_identity(self, rng: np.random.Generator) -> None:
        p = random_spd(6, rng=rng)
        assert relative_error(whiten(p, p), np.eye(6)) < 1e-11


# --------------------------------------------------------------------------- #
# 4. Geodesics
# --------------------------------------------------------------------------- #


class TestGeodesic:
    """Curves of shortest length, and the properties that identify them."""

    @pytest.mark.parametrize("metric", GEODESIC_METRICS)
    def test_endpoints(self, metric: str, rng: np.random.Generator) -> None:
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        assert relative_error(geodesic(a, b, 0.0, metric=metric), a) < 1e-11
        assert relative_error(geodesic(a, b, 1.0, metric=metric), b) < 1e-11

    def test_distance_grows_linearly_along_the_curve(
        self, rng: np.random.Generator
    ) -> None:
        """``d(A, gamma(t)) = t * d(A, B)``.

        The property that makes it a *geodesic* and not merely an
        interpolation. A curve joining the endpoints without this is not a
        shortest path.
        """
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        total = float(distance_airm(a, b))
        for t in (0.1, 0.25, 0.5, 0.75, 0.9):
            partial = float(distance_airm(a, geodesic(a, b, t)))
            assert partial == pytest.approx(t * total, rel=1e-8)

    def test_midpoint_is_the_two_matrix_mean(self, rng: np.random.Generator) -> None:
        """The only case where the AIRM Frechet mean has a closed form."""
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        assert (
            relative_error(geodesic(a, b, 0.5), frechet_mean(np.stack([a, b]))) < 1e-8
        )

    def test_reversal_symmetry(self, rng: np.random.Generator) -> None:
        a = random_spd(6, rng=rng)
        b = random_spd(6, rng=rng)
        assert relative_error(geodesic(a, b, 0.3), geodesic(b, a, 0.7)) < 1e-10

    def test_vectorised_t(self, rng: np.random.Generator) -> None:
        """A different interpolation parameter per matrix in a batch."""
        a = random_spd(5, rng=rng, batch=4)
        b = random_spd(5, rng=rng, batch=4)
        t = np.array([0.0, 0.25, 0.75, 1.0])
        result = geodesic(a, b, t)
        assert result.shape == (4, 5, 5)
        assert relative_error(result[0], a[0]) < 1e-11
        assert relative_error(result[3], b[3]) < 1e-11
        assert relative_error(result[1], geodesic(a[1], b[1], 0.25)) < 1e-11

    def test_extrapolation_stays_on_the_manifold(
        self, rng: np.random.Generator
    ) -> None:
        """``t`` outside [0, 1] is well defined and must remain SPD."""
        a = random_spd(5, rng=rng)
        b = random_spd(5, rng=rng)
        for t in (-0.5, 1.5, 2.0):
            assert bool(is_spd(geodesic(a, b, t)))

    def test_stein_geodesic_is_refused(self, rng: np.random.Generator) -> None:
        a = random_spd(4, rng=rng)
        b = random_spd(4, rng=rng)
        with pytest.raises(ValueError, match="no closed-form geodesic"):
            geodesic(a, b, 0.5, metric="stein")


# --------------------------------------------------------------------------- #
# 5. Parallel transport
# --------------------------------------------------------------------------- #


class TestParallelTransport:
    """Moving tangent vectors between base points without distorting them."""

    def test_is_an_isometry(self, rng: np.random.Generator) -> None:
        """The defining property: lengths are preserved.

        Transport is the correct formulation of cross-subject alignment. If it
        were not an isometry, moving a subject's features into a shared frame
        would rescale them, and the "alignment" would be introducing exactly
        the between-subject variance it is meant to remove.
        """
        p = random_spd(6, rng=rng)
        q = random_spd(6, rng=rng)
        for _ in range(10):
            tangent = log_map(p, random_spd(6, rng=rng))
            transported = parallel_transport(p, q, tangent)
            assert riemannian_norm(q, transported) == pytest.approx(
                riemannian_norm(p, tangent), rel=1e-9
            )

    def test_transport_to_self_is_the_identity(self, rng: np.random.Generator) -> None:
        p = random_spd(6, rng=rng)
        tangent = log_map(p, random_spd(6, rng=rng))
        assert relative_error(parallel_transport(p, p, tangent), tangent) < 1e-10

    def test_is_invertible(self, rng: np.random.Generator) -> None:
        p = random_spd(6, rng=rng)
        q = random_spd(6, rng=rng)
        tangent = log_map(p, random_spd(6, rng=rng))
        there_and_back = parallel_transport(q, p, parallel_transport(p, q, tangent))
        assert relative_error(there_and_back, tangent) < 1e-9

    def test_preserves_symmetry_exactly(self, rng: np.random.Generator) -> None:
        p = random_spd(6, rng=rng)
        q = random_spd(6, rng=rng)
        tangent = log_map(p, random_spd(6, rng=rng))
        result = parallel_transport(p, q, tangent)
        assert np.array_equal(result, result.T)

    def test_commutes_with_the_exp_map(self, rng: np.random.Generator) -> None:
        """Transporting then exponentiating preserves geodesic distance."""
        p = random_spd(5, rng=rng)
        q = random_spd(5, rng=rng)
        target = random_spd(5, rng=rng)
        tangent = log_map(p, target)
        moved = exp_map(q, parallel_transport(p, q, tangent))
        assert float(distance_airm(q, moved)) == pytest.approx(
            float(distance_airm(p, target)), rel=1e-8
        )


# --------------------------------------------------------------------------- #
# 6. Frechet means
# --------------------------------------------------------------------------- #


class TestFrechetMean:
    """The centre of mass, and the guarantees an MDM classifier needs."""

    @pytest.mark.parametrize("metric", ["airm", "logeuclid", "euclid"])
    def test_returns_spd(self, metric: str, rng: np.random.Generator) -> None:
        x = random_spd(7, rng=rng, batch=30)
        assert bool(is_spd(frechet_mean(x, metric=metric)))

    def test_minimises_sum_of_squared_distances(self, rng: np.random.Generator) -> None:
        """The definition, checked against perturbations in every direction.

        Convergence of the iteration is not the same claim as optimality of
        the fixed point. This test asserts the latter, which is what an MDM
        classifier actually relies on.
        """
        x = random_spd(6, rng=rng, batch=25)
        mean = frechet_mean(x)

        def objective(centre: np.ndarray) -> float:
            return float(
                np.sum(distance_airm(x, np.broadcast_to(centre, x.shape)) ** 2)
            )

        best = objective(mean)
        for step in (0.01, 0.05, 0.2):
            for k in (0, 7, 19):
                perturbed = exp_map(mean, step * log_map(mean, x[k]))
                assert objective(perturbed) > best

    def test_single_matrix_is_its_own_mean(self, rng: np.random.Generator) -> None:
        a = random_spd(5, rng=rng)
        assert relative_error(frechet_mean(a[None]), a) < 1e-11

    def test_identical_matrices(self, rng: np.random.Generator) -> None:
        """A degenerate but common case: an MDM class with one repeated trial."""
        a = random_spd(5, rng=rng)
        result = frechet_mean(np.stack([a] * 8), return_info=True)
        assert isinstance(result, MeanResult)
        assert result.converged
        assert relative_error(result.mean, a) < 1e-11

    def test_congruence_equivariance(self, rng: np.random.Generator) -> None:
        """``mean(W X W^T) = W mean(X) W^T``.

        Not checkable against a reference implementation -- both would share
        any error. It is the property that makes the AIRM mean the right
        centroid for data whose recording basis is unknown, and it follows
        from affine invariance rather than from the iteration converging.
        """
        x = random_spd(5, rng=rng, batch=20)
        w = rng.standard_normal((5, 5))
        transformed = frechet_mean(np.einsum("ij,bjk,lk->bil", w, x, w))
        expected = w @ frechet_mean(x) @ w.T
        assert relative_error(transformed, expected) < 1e-6

    def test_weights_shift_the_mean_toward_their_mass(
        self, rng: np.random.Generator
    ) -> None:
        x = random_spd(5, rng=rng, batch=10)
        weights = np.ones(10)
        weights[0] = 50.0
        weighted = frechet_mean(x, weights=weights)
        assert float(distance_airm(weighted, x[0])) < float(
            distance_airm(frechet_mean(x), x[0])
        )

    def test_uniform_weights_match_the_unweighted_mean(
        self, rng: np.random.Generator
    ) -> None:
        x = random_spd(5, rng=rng, batch=12)
        assert (
            relative_error(frechet_mean(x, weights=np.full(12, 3.7)), frechet_mean(x))
            < 1e-11
        )

    def test_init_does_not_change_the_fixed_point(
        self, rng: np.random.Generator
    ) -> None:
        """A different starting point must reach the same optimum."""
        x = random_spd(6, rng=rng, batch=20)
        default = frechet_mean(x)
        from_euclid = frechet_mean(x, init=mean_euclid(x))
        assert relative_error(from_euclid, default) < 1e-7

    def test_return_info_reports_convergence(self, rng: np.random.Generator) -> None:
        x = random_spd(6, rng=rng, batch=30)
        result = frechet_mean(x, return_info=True)
        assert isinstance(result, MeanResult)
        assert result.converged
        assert result.final_criterion < 1e-10
        assert len(result.history) == result.n_iter
        # Monotone descent is what the step-halving guard buys; without it the
        # iteration can oscillate while appearing to progress.
        assert result.history[-1] < result.history[0]

    def test_non_convergence_warns_and_reports(
        self, rng: np.random.Generator, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hitting the cap must be loud.

        A silently non-converged centroid inside a cross-validation fold
        produces a plausible-looking classifier that is wrong in a way
        accuracy alone cannot diagnose.
        """
        x = random_spd(6, rng=rng, batch=20)
        with caplog.at_level(logging.WARNING, logger="geoq.geometry.riemannian"):
            result = frechet_mean(x, max_iter=1, tol=1e-16, return_info=True)
        assert isinstance(result, MeanResult)
        assert not result.converged
        assert "did not converge" in caplog.text

    def test_closed_form_metrics_report_zero_iterations(
        self, rng: np.random.Generator
    ) -> None:
        x = random_spd(5, rng=rng, batch=10)
        for metric in ("logeuclid", "euclid"):
            result = frechet_mean(x, metric=metric, return_info=True)
            assert isinstance(result, MeanResult)
            assert result.n_iter == 0
            assert result.converged

    def test_logeuclid_mean_matches_its_definition(
        self, rng: np.random.Generator
    ) -> None:
        from geoq.geometry.spd import expm_sym, logm_spd

        x = random_spd(6, rng=rng, batch=15)
        expected = expm_sym(np.mean(logm_spd(x), axis=0))
        assert relative_error(mean_logeuclid(x), expected) < 1e-12

    def test_euclid_mean_is_the_arithmetic_mean(self, rng: np.random.Generator) -> None:
        x = random_spd(6, rng=rng, batch=15)
        assert relative_error(mean_euclid(x), np.mean(x, axis=0)) < 1e-14

    def test_airm_and_logeuclid_means_differ(self, rng: np.random.Generator) -> None:
        """If these agreed, the manifold structure would be doing no work."""
        x = random_spd(6, rng=rng, batch=20, condition_number=1e4)
        assert relative_error(frechet_mean(x), mean_logeuclid(x)) > 1e-6


# --------------------------------------------------------------------------- #
# 7. Pairwise distance matrices
# --------------------------------------------------------------------------- #


class TestPairwiseDistances:
    """The object every kernel method consumes."""

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_symmetric_with_exactly_zero_diagonal(
        self, metric: str, rng: np.random.Generator
    ) -> None:
        """Bitwise zero on the diagonal, not merely small.

        ``d(A, A)`` evaluates to ~1e-8 for ill-conditioned A through
        cancellation. A kernel built from a matrix with a non-zero diagonal is
        not positive semi-definite, and this is precisely the matrix Block 5.5
        compares quantum kernels against.
        """
        x = random_spd(6, rng=rng, batch=15, condition_number=1e8)
        d = pairwise_distances(x, metric=metric)
        assert np.array_equal(d, d.T)
        assert np.array_equal(np.diag(d), np.zeros(15))

    def test_matches_elementwise_computation(self, rng: np.random.Generator) -> None:
        x = random_spd(5, rng=rng, batch=12)
        d = pairwise_distances(x)
        for i, j in ((0, 1), (3, 9), (7, 11)):
            assert float(d[i, j]) == pytest.approx(
                float(distance_airm(x[i], x[j])), rel=1e-12
            )

    def test_cross_set_shape_and_values(self, rng: np.random.Generator) -> None:
        x = random_spd(5, rng=rng, batch=6)
        y = random_spd(5, rng=rng, batch=9)
        d = pairwise_distances(x, y)
        assert d.shape == (6, 9)
        assert float(d[2, 5]) == pytest.approx(
            float(distance_airm(x[2], y[5])), rel=1e-12
        )

    def test_self_pairing_matches_cross_pairing(self, rng: np.random.Generator) -> None:
        """The triangular fast path must agree with the general path."""
        x = random_spd(5, rng=rng, batch=8)
        assert np.allclose(pairwise_distances(x), pairwise_distances(x, x), atol=1e-12)

    def test_rejects_wrong_rank(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match=r"shape \(m, n, n\)"):
            pairwise_distances(random_spd(5, rng=rng))

    def test_rejects_mismatched_dimensions(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="same matrix dimension"):
            pairwise_distances(
                random_spd(5, rng=rng, batch=3), random_spd(6, rng=rng, batch=3)
            )


# --------------------------------------------------------------------------- #
# 8. Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    """Configuration errors fail loudly and early."""

    def test_unknown_metric_raises(self, rng: np.random.Generator) -> None:
        """A typo in a YAML file must not silently change the geometry."""
        a = random_spd(4, rng=rng)
        with pytest.raises(ValueError, match="Unknown metric"):
            distance(a, a, metric="reimannian")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown metric"):
            frechet_mean(a[None], metric="euclidian")  # type: ignore[arg-type]

    def test_error_message_lists_available_metrics(
        self, rng: np.random.Generator
    ) -> None:
        a = random_spd(4, rng=rng)
        with pytest.raises(ValueError) as excinfo:
            distance(a, a, metric="airm2")  # type: ignore[arg-type]
        assert all(name in str(excinfo.value) for name in METRICS)

    def test_stein_mean_is_refused_rather_than_approximated(
        self, rng: np.random.Generator
    ) -> None:
        """Silently returning the AIRM mean under a Stein label would be wrong."""
        x = random_spd(4, rng=rng, batch=5)
        with pytest.raises(ValueError, match="distinct fixed-point iteration"):
            frechet_mean(x, metric="stein")

    def test_non_spd_input_rejected(self, rng: np.random.Generator) -> None:
        a = random_spd(4, rng=rng)
        singular = np.diag([1.0, 1.0, 1.0, 0.0])
        with pytest.raises(NotPositiveDefiniteError):
            distance_airm(a, singular)
        with pytest.raises(NotPositiveDefiniteError):
            log_map(singular, a)

    def test_asymmetric_tangent_rejected(self, rng: np.random.Generator) -> None:
        p = random_spd(3, rng=rng)
        with pytest.raises(NotSymmetricError):
            exp_map(p, np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    @pytest.mark.parametrize(
        ("weights", "pattern"),
        [
            (np.ones(3), r"shape \(5,\)"),
            (np.array([1.0, -1.0, 1.0, 1.0, 1.0]), "non-negative"),
            (np.zeros(5), "positive value"),
        ],
    )
    def test_invalid_weights_rejected(
        self, weights: np.ndarray, pattern: str, rng: np.random.Generator
    ) -> None:
        x = random_spd(4, rng=rng, batch=5)
        with pytest.raises(ValueError, match=pattern):
            frechet_mean(x, weights=weights)

    def test_mean_rejects_wrong_rank(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match=r"shape \(m, n, n\)"):
            frechet_mean(random_spd(4, rng=rng))


# --------------------------------------------------------------------------- #
# 9. Property-based tests
# --------------------------------------------------------------------------- #


class TestProperties:
    """Identities that must hold for arbitrary SPD inputs."""

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        a=spd_matrices(max_dim=8, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA),
        b=spd_matrices(max_dim=8, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA),
    )
    def test_distance_is_symmetric_and_non_negative(self, a, b) -> None:
        if a.shape != b.shape:
            return
        tolerance = airm_tolerance(a, b)
        for metric in ALL_METRICS:
            forward = float(distance(a, b, metric=metric))
            assert forward >= 0.0
            assert forward == pytest.approx(
                float(distance(b, a, metric=metric)),
                rel=tolerance,
                abs=STEIN_SELF_DISTANCE_TOL,
            )

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        a=spd_matrices(max_dim=8, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA),
        b=spd_matrices(max_dim=8, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA),
    )
    def test_log_exp_round_trip(self, a, b) -> None:
        if a.shape != b.shape:
            return
        assert relative_error(exp_map(a, log_map(a, b)), b) < airm_tolerance(a, b)

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        a=spd_matrices(max_dim=6, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA),
        b=spd_matrices(max_dim=6, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA),
    )
    def test_tangent_norm_equals_distance(self, a, b) -> None:
        if a.shape != b.shape:
            return
        expected = float(distance_airm(a, b))
        if expected < 1e-8:
            return
        assert riemannian_norm(a, log_map(a, b)) == pytest.approx(
            expected, rel=airm_tolerance(a, b)
        )

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        a=spd_matrices(max_dim=6),
        b=spd_matrices(max_dim=6),
        t=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_geodesic_stays_on_the_manifold(self, a, b, t: float) -> None:
        if a.shape != b.shape:
            return
        assert bool(is_spd(geodesic(a, b, t)))

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices(max_dim=6), b=spd_matrices(max_dim=6))
    def test_stein_is_bounded_by_airm(self, a, b) -> None:
        """The Stein divergence never exceeds the geodesic distance.

        A known inequality (Sra 2012). It is the cheapest available check that
        the two curved metrics are describing the same geometry rather than
        diverging through an implementation error in one of them.
        """
        if a.shape != b.shape:
            return
        airm = float(distance_airm(a, b))
        if airm < STEIN_MEANINGFUL_DISTANCE:
            return
        assert float(distance_stein(a, b)) <= airm * (1.0 + 1e-9)

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices(max_dim=6, max_log_kappa=AIRM_PROPERTY_MAX_LOG_KAPPA))
    def test_distance_to_self_is_zero(self, a) -> None:
        #  Not exactly zero for AIRM: d(A, A) is computed by whitening A with
        #  its own inverse square root, and the result differs from the
        #  identity by ~eps * kappa ** 2. The tolerance therefore has to be
        #  conditioning-aware, floored at the flat tolerance that the
        #  closed-form metrics satisfy.
        tolerance = max(STEIN_SELF_DISTANCE_TOL, airm_tolerance(a))
        for metric in ALL_METRICS:
            assert float(distance(a, a, metric=metric)) < tolerance

    @settings(
        max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(a=spd_matrices(max_dim=6), b=spd_matrices(max_dim=6))
    def test_euclid_distance_matches_frobenius(self, a, b) -> None:
        if a.shape != b.shape:
            return
        assert float(distance_euclid(a, b)) == pytest.approx(
            float(np.linalg.norm(a - b)), rel=1e-12
        )