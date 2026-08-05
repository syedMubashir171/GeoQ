"""Tests for :mod:`geoq.geometry.tangent`.

The central claim
-----------------
``||tangent_space(X, P)||_2 == distance_airm(P, X)``.

Everything else in this module is bookkeeping around that identity. It is what
makes a linear classifier on tangent-space features a classifier on the
*geometry* rather than on an arbitrary coordinate chart, and it is the reason
the TS+LDA baseline works at all.

The identity holds only because of the whitening step. ``TestIsometry``
asserts it, and ``test_unwhitened_map_is_not_an_isometry`` asserts the
converse -- that skipping whitening breaks it. The second test exists because
the unwhitened version produces plausible-looking features that classify
plausibly well; the error is silent, and measured here at 0.36 absolute
against a true distance, it is far too large to be dismissed as numerical
noise but far too subtle to notice in an accuracy table.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from geoq.geometry.riemannian import distance_airm, frechet_mean, log_map
from geoq.geometry.spd import (
    NotPositiveDefiniteError,
    NotSymmetricError,
    is_spd,
    random_spd,
)
from geoq.geometry.tangent import (
    OFF_DIAGONAL_SCALE,
    matrix_dimension,
    tangent_space,
    untangent_space,
    unvectorize,
    vector_dimension,
    vectorize,
)
from geoq.testing import relative_error, spd_matrices

#  22 is BCI Competition IV 2a's channel count; 2 is the Paper-4
#  proof-of-concept case.
DIMENSIONS = [2, 3, 6, 22]

#  The tangent map whitens, and whitening inverts, so its effective
#  conditioning is kappa ** 2 -- the same amplification established for the
#  Riemannian layer. Property tests therefore draw from a restricted
#  conditioning range where a meaningful tolerance still exists.
PROPERTY_MAX_LOG_KAPPA = 4.0


@pytest.fixture
def reference_point(rng: np.random.Generator) -> np.ndarray:
    """A realistic reference: the Frechet mean of a set, as a pipeline uses."""
    return frechet_mean(random_spd(6, rng=rng, batch=30))


# --------------------------------------------------------------------------- #
# 1. Dimension arithmetic
# --------------------------------------------------------------------------- #


class TestDimensions:
    """Vector and matrix dimensions convert without ambiguity."""

    @pytest.mark.parametrize(
        ("n", "d"), [(1, 1), (2, 3), (3, 6), (4, 10), (6, 21), (22, 253)]
    )
    def test_known_values(self, n: int, d: int) -> None:
        assert vector_dimension(n) == d
        assert matrix_dimension(d) == n

    @pytest.mark.parametrize("n", range(1, 40))
    def test_round_trip(self, n: int) -> None:
        assert matrix_dimension(vector_dimension(n)) == n

    @pytest.mark.parametrize("d", [2, 4, 5, 7, 8, 9, 254])
    def test_invalid_length_rejected(self, d: int) -> None:
        """A sliced or concatenated feature matrix must fail loudly.

        Silently reshaping an invalid length would produce a matrix filled
        with misaligned entries -- wrong, but structurally valid, and
        therefore undetectable downstream.
        """
        with pytest.raises(ValueError, match="does not correspond"):
            matrix_dimension(d)

    def test_error_message_suggests_the_nearest_valid_length(self) -> None:
        with pytest.raises(ValueError, match="nearest valid length is 253"):
            matrix_dimension(254)

    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_non_positive_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            vector_dimension(bad)
        with pytest.raises(ValueError, match=">= 1"):
            matrix_dimension(bad)


# --------------------------------------------------------------------------- #
# 2. Vectorisation
# --------------------------------------------------------------------------- #


class TestVectorization:
    """Flattening preserves the Frobenius norm and is exactly invertible."""

    def test_explicit_convention(self) -> None:
        """Pins the ordering and scaling against a hand-computed example.

        Written out literally rather than derived, because this is the
        convention that must match pyRiemann. A test that recomputed the
        expected value using the same index expressions as the implementation
        would pass even if both were wrong.
        """
        s = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        root_two = OFF_DIAGONAL_SCALE
        expected = np.array(
            [1.0, 2.0 * root_two, 3.0 * root_two, 4.0, 5.0 * root_two, 6.0]
        )
        assert np.allclose(vectorize(s), expected)

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_is_an_isometry(self, n: int, rng: np.random.Generator) -> None:
        """``||vec(S)||_2 == ||S||_F``.

        This is what the sqrt(2) buys. Without it each off-diagonal entry is
        counted once rather than twice and the norm is wrong by a factor that
        depends on how much energy sits off the diagonal -- that is, on the
        spatial correlation structure of the EEG.
        """
        s = np.array([np.diag(np.ones(n))] * 0 + [rng.standard_normal((n, n))])
        s = 0.5 * (s + np.swapaxes(s, -1, -2))
        assert float(np.linalg.norm(vectorize(s))) == pytest.approx(
            float(np.linalg.norm(s)), rel=1e-14
        )

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_round_trip(self, n: int, rng: np.random.Generator) -> None:
        s = rng.standard_normal((7, n, n))
        s = 0.5 * (s + np.swapaxes(s, -1, -2))
        assert np.allclose(unvectorize(vectorize(s)), s, atol=1e-15)

    def test_accepts_indefinite_input(self, rng: np.random.Generator) -> None:
        """Tangent vectors are displacements and are routinely indefinite."""
        s = np.diag([-3.0, -1.0, 0.0, 2.0])
        assert np.allclose(unvectorize(vectorize(s)), s)

    def test_rejects_asymmetric_input(self) -> None:
        with pytest.raises(NotSymmetricError):
            vectorize(np.array([[1.0, 2.0], [0.0, 1.0]]))

    @pytest.mark.parametrize("batch", [(), (1,), (5,), (3, 4)])
    def test_batch_shapes(
        self, batch: tuple[int, ...], rng: np.random.Generator
    ) -> None:
        s = rng.standard_normal((*batch, 5, 5))
        s = 0.5 * (s + np.swapaxes(s, -1, -2))
        vectorised = vectorize(s)
        assert vectorised.shape == (*batch, 15)
        assert unvectorize(vectorised).shape == (*batch, 5, 5)

    def test_unvectorize_output_is_exactly_symmetric(
        self, rng: np.random.Generator
    ) -> None:
        v = rng.standard_normal(21)
        result = unvectorize(v)
        assert np.array_equal(result, result.T)


# --------------------------------------------------------------------------- #
# 3. The isometry contract
# --------------------------------------------------------------------------- #


class TestIsometry:
    """The identity the whole module exists to provide."""

    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_feature_norm_equals_geodesic_distance(
        self, n: int, rng: np.random.Generator
    ) -> None:
        reference = frechet_mean(random_spd(n, rng=rng, batch=20))
        x = random_spd(n, rng=rng, batch=25)
        features = tangent_space(x, reference)
        distances = distance_airm(np.broadcast_to(reference, x.shape), x)
        assert np.allclose(np.linalg.norm(features, axis=-1), distances, rtol=1e-9)

    def test_reference_maps_to_the_origin(self, reference_point: np.ndarray) -> None:
        assert (
            float(np.abs(tangent_space(reference_point, reference_point)).max()) < 1e-10
        )

    def test_unwhitened_map_is_not_an_isometry(
        self, reference_point: np.ndarray, rng: np.random.Generator
    ) -> None:
        """The mistake this module is built to prevent, asserted explicitly.

        ``vec(Log_P(X))`` without whitening yields features that look
        reasonable and classify reasonably, but whose Euclidean geometry is
        not the manifold's. Measured here the discrepancy reaches ~0.36
        against true distances of order 1 -- far above numerical noise, and
        invisible in any accuracy table.

        If this test ever fails, whitening has been silently introduced into
        the naive path or removed from the real one; either way the isometry
        claim above would no longer mean what it says.
        """
        x = random_spd(6, rng=rng, batch=30)
        naive = vectorize(log_map(np.broadcast_to(reference_point, x.shape), x))
        distances = distance_airm(np.broadcast_to(reference_point, x.shape), x)
        discrepancy = np.abs(np.linalg.norm(naive, axis=-1) - distances).max()
        assert discrepancy > 1e-3 * float(distances.mean())

    def test_distances_between_feature_vectors_approximate_geodesics(
        self, reference_point: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Euclidean distance in feature space tracks geodesic distance.

        Exact only at the reference point; the approximation degrades with
        distance from it, which is why a pipeline re-centres per subject. The
        loose tolerance here is the honest statement of that: 25 percent, not
        machine precision. A tighter assertion would be false.
        """
        x = random_spd(6, rng=rng, batch=20)
        # Keep the set tight around the reference so the linearisation holds.
        x = np.stack(
            [frechet_mean(np.stack([reference_point, m]), metric="airm") for m in x]
        )
        features = tangent_space(x, reference_point)
        for i, j in ((0, 1), (3, 7), (12, 19)):
            euclidean = float(np.linalg.norm(features[i] - features[j]))
            geodesic_distance = float(distance_airm(x[i], x[j]))
            assert euclidean == pytest.approx(geodesic_distance, rel=0.25)


# --------------------------------------------------------------------------- #
# 4. Round trip and the inverse map
# --------------------------------------------------------------------------- #


class TestInverseMap:
    """``untangent_space`` recovers the manifold points."""

    @pytest.mark.parametrize("metric", ["airm", "logeuclid"])
    @pytest.mark.parametrize("n", DIMENSIONS)
    def test_round_trip(self, metric: str, n: int, rng: np.random.Generator) -> None:
        reference = frechet_mean(random_spd(n, rng=rng, batch=20))
        x = random_spd(n, rng=rng, batch=15)
        recovered = untangent_space(
            tangent_space(x, reference, metric=metric), reference, metric=metric
        )
        assert relative_error(recovered, x) < 1e-9

    @pytest.mark.parametrize("metric", ["airm", "logeuclid"])
    def test_output_is_spd(
        self, metric: str, reference_point: np.ndarray, rng: np.random.Generator
    ) -> None:
        v = rng.standard_normal((10, 21))
        assert bool(np.all(is_spd(untangent_space(v, reference_point, metric=metric))))

    def test_origin_maps_to_the_reference(self, reference_point: np.ndarray) -> None:
        assert (
            relative_error(
                untangent_space(np.zeros(21), reference_point), reference_point
            )
            < 1e-12
        )

    def test_metric_mismatch_changes_the_result(
        self, reference_point: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Forward and inverse must use the same metric.

        Mixing them produces a valid SPD matrix that is simply the wrong one,
        so this asserts the two paths are genuinely distinct rather than
        silently equivalent.
        """
        x = random_spd(6, rng=rng, batch=5)
        features = tangent_space(x, reference_point, metric="airm")
        mismatched = untangent_space(features, reference_point, metric="logeuclid")
        assert relative_error(mismatched, x) > 1e-6


# --------------------------------------------------------------------------- #
# 5. Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    """Misuse fails loudly, especially misuse that would leak or mislead."""

    def test_batched_reference_rejected(self, rng: np.random.Generator) -> None:
        """A per-sample reference would make the coordinates meaningless.

        Every feature vector would sit in a different tangent space, so their
        entries would not be comparable and no classifier trained on them
        would mean anything. Broadcasting silently is the dangerous
        alternative.
        """
        x = random_spd(5, rng=rng, batch=8)
        with pytest.raises(ValueError, match="single matrix"):
            tangent_space(x, random_spd(5, rng=rng, batch=8))

    @pytest.mark.parametrize("metric", ["stein", "euclid", "reimannian"])
    def test_unsupported_metric_rejected(
        self, metric: str, reference_point: np.ndarray, rng: np.random.Generator
    ) -> None:
        x = random_spd(6, rng=rng, batch=3)
        with pytest.raises(ValueError, match=r"[Uu]nsupported metric"):
            tangent_space(x, reference_point, metric=metric)  # type: ignore[arg-type]

    def test_non_spd_reference_rejected(self, rng: np.random.Generator) -> None:
        x = random_spd(4, rng=rng, batch=3)
        with pytest.raises(NotPositiveDefiniteError):
            tangent_space(x, np.diag([1.0, 1.0, 1.0, 0.0]))

    def test_non_spd_input_rejected(self, rng: np.random.Generator) -> None:
        reference = random_spd(4, rng=rng)
        singular = np.stack([np.diag([1.0, 1.0, 1.0, 0.0])])
        with pytest.raises(NotPositiveDefiniteError):
            tangent_space(singular, reference)


# --------------------------------------------------------------------------- #
# 6. Property-based tests
# --------------------------------------------------------------------------- #


class TestProperties:
    """Identities holding for arbitrary SPD inputs."""

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(s=spd_matrices(max_dim=10))
    def test_vectorization_is_an_isometry(self, s) -> None:
        assert float(np.linalg.norm(vectorize(s))) == pytest.approx(
            float(np.linalg.norm(s)), rel=1e-13
        )

    @settings(
        max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(s=spd_matrices(max_dim=10))
    def test_vectorization_round_trip(self, s) -> None:
        assert relative_error(unvectorize(vectorize(s)), s) < 1e-13

    @settings(
        max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        x=spd_matrices(max_dim=8, max_log_kappa=PROPERTY_MAX_LOG_KAPPA),
        reference=spd_matrices(max_dim=8, max_log_kappa=PROPERTY_MAX_LOG_KAPPA),
    )
    def test_feature_norm_equals_distance(self, x, reference) -> None:
        if x.shape != reference.shape:
            return
        expected = float(distance_airm(reference, x))
        if expected < 1e-8:
            return
        norm = float(np.linalg.norm(tangent_space(x, reference)))
        assert norm == pytest.approx(expected, rel=1e-6)

    @settings(
        max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        x=spd_matrices(max_dim=8, max_log_kappa=PROPERTY_MAX_LOG_KAPPA),
        reference=spd_matrices(max_dim=8, max_log_kappa=PROPERTY_MAX_LOG_KAPPA),
    )
    def test_map_round_trip(self, x, reference) -> None:
        if x.shape != reference.shape:
            return
        recovered = untangent_space(tangent_space(x, reference), reference)
        assert relative_error(recovered, x) < 1e-6
