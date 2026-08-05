"""Tests for :mod:`geoq.models.classical.mdm`.

What is being defended
----------------------
MDM is the baseline every quantum claim in this thesis will be measured
against, so its correctness has to be established more carefully than its
simplicity suggests. Four properties matter:

* **The centroids are the class Frechet means.** Not approximately, not
  something close -- the classifier is defined by them, and if they are wrong
  every reported accuracy is wrong in a way no amount of cross-validation
  reveals.
* **Decisions are geometric, not coordinate-dependent.** Under the
  affine-invariant metric, congruence-transforming every trial must leave
  predictions unchanged. That is what makes MDM meaningful across subjects
  with different head geometry and electrode impedance, and it is a property
  no reference implementation can confirm for you.
* **Leakage is structurally impossible.** The centroids come from ``fit`` and
  are reused verbatim; nothing is recomputed at prediction time.
* **Failures are visible.** A non-converged centroid displaces a decision
  boundary by an unknown amount, and the resulting accuracy drop looks like a
  modelling result rather than a numerical one.

Constructing separable data
---------------------------
Random SPD matrices are not separable, so an accuracy test on them measures
nothing. The fixture below places two clusters at distinct points on the
manifold and interpolates each trial part of the way toward a random matrix,
which produces genuinely separable classes with controllable overlap.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the 'ml' extra: pip install -e '.[ml]'")

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import NotFittedError

from geoq.features.covariance import Covariances
from geoq.geometry.riemannian import (
    distance,
    distance_airm,
    frechet_mean,
    geodesic,
)
from geoq.geometry.spd import NotPositiveDefiniteError, is_spd, random_spd
from geoq.models.classical.mdm import MDM
from geoq.testing import relative_error

N_PER_CLASS = 30
N_CHANNELS = 6


@pytest.fixture
def separable(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two genuinely separable classes of SPD matrices.

    Each class is a cloud around its own point on the manifold, produced by
    interpolating a quarter of the way from that point toward a random matrix.
    """
    anchors = [random_spd(N_CHANNELS, rng=rng) for _ in range(2)]
    trials = [
        geodesic(anchor, random_spd(N_CHANNELS, rng=rng), 0.25)
        for anchor in anchors
        for _ in range(N_PER_CLASS)
    ]
    x = np.stack(trials)
    y = np.repeat([0, 1], N_PER_CLASS)
    return x, y


@pytest.fixture
def three_class(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Three separable classes, matching motor-imagery paradigms."""
    anchors = [random_spd(5, rng=rng) for _ in range(3)]
    trials = [
        geodesic(anchor, random_spd(5, rng=rng), 0.2)
        for anchor in anchors
        for _ in range(20)
    ]
    return np.stack(trials), np.repeat([0, 1, 2], 20)


# --------------------------------------------------------------------------- #
# 1. The centroids
# --------------------------------------------------------------------------- #


class TestCentroids:
    """The learned objects are exactly the class Frechet means."""

    @pytest.mark.parametrize("metric", ["airm", "logeuclid", "euclid"])
    def test_centroid_is_the_class_mean(
        self, metric: str, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        fitted = MDM(metric=metric).fit(x, y)
        for index, label in enumerate(fitted.classes_):
            expected = frechet_mean(x[y == label], metric=metric)
            assert relative_error(fitted.centroids_[index], expected) < 1e-7

    def test_centroids_are_spd(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = separable
        assert bool(np.all(is_spd(MDM().fit(x, y).centroids_)))

    def test_centroid_order_matches_classes(self, rng: np.random.Generator) -> None:
        """Ordering is load-bearing: predictions index ``classes_`` directly.

        A silent misalignment would produce a classifier that is systematically
        wrong while looking entirely healthy, and on a two-class problem it
        would report near-zero accuracy rather than an obvious error.
        """
        x = random_spd(4, rng=rng, batch=40)
        y = np.array(["left"] * 20 + ["right"] * 20)
        fitted = MDM().fit(x, y)
        assert list(fitted.classes_) == ["left", "right"]
        assert relative_error(fitted.centroids_[0], frechet_mean(x[y == "left"])) < 1e-7

    def test_shape(self, three_class: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = three_class
        fitted = MDM().fit(x, y)
        assert fitted.centroids_.shape == (3, 5, 5)

    def test_sample_weight_shifts_the_centroid(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        weights = np.ones(x.shape[0])
        weights[0] = 100.0
        weighted = MDM().fit(x, y, sample_weight=weights)
        plain = MDM().fit(x, y)
        assert float(distance_airm(weighted.centroids_[0], x[0])) < float(
            distance_airm(plain.centroids_[0], x[0])
        )

    def test_uniform_sample_weight_is_a_no_op(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        weighted = MDM().fit(x, y, sample_weight=np.full(x.shape[0], 2.5))
        plain = MDM().fit(x, y)
        assert relative_error(weighted.centroids_, plain.centroids_) < 1e-10


# --------------------------------------------------------------------------- #
# 2. Geometric invariance
# --------------------------------------------------------------------------- #


class TestInvariance:
    """Decisions depend on the geometry, not on the coordinate system."""

    def test_predictions_are_congruence_invariant(
        self, separable: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """Transforming every trial by ``W`` must not change any prediction.

        This is the property that makes MDM usable across subjects: an unknown
        invertible ``W`` stands in for differences in head geometry, electrode
        impedance, and amplifier gain. No reference implementation can confirm
        this for you -- both would share any error -- so it is asserted
        directly.
        """
        x, y = separable
        transform = rng.standard_normal((N_CHANNELS, N_CHANNELS))
        transformed = np.einsum("ij,bjk,lk->bil", transform, x, transform)
        assert np.array_equal(
            MDM().fit(x, y).predict(x),
            MDM().fit(transformed, y).predict(transformed),
        )

    def test_predictions_are_scale_invariant(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Recording units must not change decisions."""
        x, y = separable
        for scale in (1e-9, 1e6):
            assert np.array_equal(
                MDM().fit(x, y).predict(x),
                MDM().fit(x * scale, y).predict(x * scale),
            )

    def test_logeuclid_is_not_congruence_invariant(
        self, separable: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """The negative control.

        Only AIRM is affine-invariant. If the log-Euclidean classifier also
        passed the invariance test, the metric parameter would not be
        selecting a different geometry and the previous test would prove
        nothing.
        """
        x, y = separable
        transform = rng.standard_normal((N_CHANNELS, N_CHANNELS))
        transformed = np.einsum("ij,bjk,lk->bil", transform, x, transform)
        base = MDM(metric="logeuclid").fit(x, y).transform(x)
        moved = MDM(metric="logeuclid").fit(transformed, y).transform(transformed)
        assert relative_error(moved, base) > 1e-6


# --------------------------------------------------------------------------- #
# 3. Prediction
# --------------------------------------------------------------------------- #


class TestPrediction:
    """Nearest-centroid assignment, distances, and scores."""

    def test_separable_classes_are_learned(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        assert MDM().fit(x, y).score(x, y) > 0.9

    def test_multiclass(self, three_class: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = three_class
        fitted = MDM().fit(x, y)
        assert fitted.transform(x).shape == (60, 3)
        assert fitted.score(x, y) > 0.9

    def test_prediction_is_argmin_of_distances(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        fitted = MDM().fit(x, y)
        expected = fitted.classes_[np.argmin(fitted.transform(x), axis=1)]
        assert np.array_equal(fitted.predict(x), expected)

    def test_a_centroid_is_classified_as_its_own_class(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The sanity check that catches an index misalignment immediately."""
        x, y = separable
        fitted = MDM().fit(x, y)
        assert np.array_equal(fitted.predict(fitted.centroids_), fitted.classes_)

    def test_distance_to_own_centroid_is_zero(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        fitted = MDM().fit(x, y)
        diagonal = np.diag(fitted.transform(fitted.centroids_))
        assert np.all(diagonal < 1e-8)

    def test_labels_preserve_their_dtype(self, rng: np.random.Generator) -> None:
        """String labels must come back as strings, not as indices."""
        x = random_spd(4, rng=rng, batch=30)
        y = np.array(["rest"] * 15 + ["move"] * 15)
        predictions = MDM().fit(x, y).predict(x)
        assert predictions.dtype.kind in "US"
        assert set(np.unique(predictions)) <= {"rest", "move"}

    def test_non_consecutive_integer_labels(self, rng: np.random.Generator) -> None:
        """BCI datasets often label classes 7, 8, 9 rather than 0, 1, 2."""
        x = random_spd(4, rng=rng, batch=30)
        y = np.repeat([7, 9], 15)
        assert set(np.unique(MDM().fit(x, y).predict(x))) <= {7, 9}


class TestMetricIsRespected:
    """The configured metric governs the distances, not only the centroids.

    Added after a mutation test: replacing the metric inside ``transform``
    with a hardcoded ``"euclid"``, while leaving centroid estimation correct,
    was caught by none of the original tests. The parameter would have
    silently stopped affecting decisions -- and since Paper 4 varies the
    metric as an experimental factor, that failure would have invalidated the
    comparison it exists to make.
    """

    @pytest.mark.parametrize("metric", ["airm", "logeuclid", "euclid"])
    def test_transform_matches_the_named_distance(
        self, metric: str, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        fitted = MDM(metric=metric).fit(x, y)
        distances = fitted.transform(x)
        for index in range(fitted.classes_.shape[0]):
            expected = distance(
                x,
                np.broadcast_to(fitted.centroids_[index], x.shape),
                metric=metric,
            )
            assert np.allclose(distances[:, index], expected, rtol=1e-10)

    def test_metrics_produce_different_distances(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """If they agreed, the metric parameter would be decorative."""
        x, y = separable
        airm = MDM(metric="airm").fit(x, y).transform(x)
        euclid = MDM(metric="euclid").fit(x, y).transform(x)
        logeuclid = MDM(metric="logeuclid").fit(x, y).transform(x)
        assert relative_error(euclid, airm) > 1e-3
        assert relative_error(logeuclid, airm) > 1e-6

    def test_metric_affects_at_least_some_decisions(
        self, rng: np.random.Generator
    ) -> None:
        """On overlapping classes the geometry must change the boundary.

        Well-separated classes are classified identically under every metric,
        so a test on separable data cannot detect a metric that is being
        ignored. Overlap is what makes the choice observable.
        """
        anchors = [random_spd(5, rng=rng) for _ in range(2)]
        trials = [
            geodesic(anchor, random_spd(5, rng=rng), 0.85)
            for anchor in anchors
            for _ in range(40)
        ]
        x = np.stack(trials)
        y = np.repeat([0, 1], 40)
        airm = MDM(metric="airm").fit(x, y).predict(x)
        euclid = MDM(metric="euclid").fit(x, y).predict(x)
        assert not np.array_equal(airm, euclid)


class TestPredictProba:
    """Softmax over negative squared distances."""

    def test_rows_are_stochastic(
        self, three_class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = three_class
        proba = MDM().fit(x, y).predict_proba(x)
        assert proba.shape == (60, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0)

    def test_argmax_agrees_with_predict(
        self, three_class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = three_class
        fitted = MDM().fit(x, y)
        assert np.array_equal(
            fitted.classes_[np.argmax(fitted.predict_proba(x), axis=1)],
            fitted.predict(x),
        )

    def test_stable_when_every_centroid_is_distant(
        self, separable: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """An outlier trial must not produce NaN.

        The softmax subtracts each row's maximum before exponentiating. Without
        that, a trial far from every centroid underflows the whole row to zero
        and the normalisation divides by zero -- a NaN that propagates
        silently into any averaged metric downstream.
        """
        x, y = separable
        fitted = MDM().fit(x, y)
        outlier = random_spd(N_CHANNELS, rng=rng, condition_number=1e10)[None]
        proba = fitted.predict_proba(outlier)
        assert np.all(np.isfinite(proba))
        assert float(proba.sum()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 4. Leakage
# --------------------------------------------------------------------------- #


class TestLeakagePrevention:
    """Centroids come from ``fit`` and are reused verbatim."""

    def test_predict_is_a_pure_function_of_the_centroids(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Batch composition must not affect any individual prediction."""
        x, y = separable
        fitted = MDM().fit(x, y)
        all_at_once = fitted.transform(x)
        one_at_a_time = np.vstack([fitted.transform(trial[None]) for trial in x])
        assert np.array_equal(all_at_once, one_at_a_time)

    def test_centroids_differ_from_the_full_data_version(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        train = np.arange(0, x.shape[0], 2)
        fold = MDM().fit(x[train], y[train])
        whole = MDM().fit(x, y)
        assert relative_error(fold.centroids_, whole.centroids_) > 1e-6

    def test_refitting_replaces_the_centroids(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """State must not accumulate across folds."""
        x, y = separable
        classifier = MDM()
        classifier.fit(x[::2], y[::2])
        first = classifier.centroids_.copy()
        classifier.fit(x[1::2], y[1::2])
        assert relative_error(classifier.centroids_, first) > 1e-8

    def test_cross_validation_matches_manual_folds(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Scores from the cross-validator equal hand-computed fold scores.

        If ``fit`` were seeing test data, the cross-validator's scores would
        exceed the manual ones. Equality is the evidence it is not.
        """
        x, y = separable
        splitter = StratifiedKFold(4, shuffle=True, random_state=0)
        automatic = cross_val_score(MDM(), x, y, cv=splitter)
        manual = np.array(
            [
                MDM().fit(x[train], y[train]).score(x[test], y[test])
                for train, test in splitter.split(x, y)
            ]
        )
        assert np.allclose(automatic, manual)


# --------------------------------------------------------------------------- #
# 5. Estimator contract and pipelines
# --------------------------------------------------------------------------- #


class TestEstimatorContract:
    """Compliance with the scikit-learn classifier interface."""

    def test_get_params_round_trips(self) -> None:
        classifier = MDM(metric="logeuclid", tol=1e-8, max_iter=50)
        params = classifier.get_params()
        assert params["metric"] == "logeuclid"
        assert MDM(**params).get_params() == params

    def test_clone_drops_fitted_state(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        cloned = clone(MDM(metric="euclid").fit(x, y))
        assert cloned.metric == "euclid"
        assert not hasattr(cloned, "centroids_")

    def test_fit_returns_self(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = separable
        classifier = MDM()
        assert classifier.fit(x, y) is classifier

    def test_predict_before_fit_raises(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = separable
        with pytest.raises(NotFittedError):
            MDM().predict(x)

    def test_fitted_attributes(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = separable
        fitted = MDM().fit(x, y)
        assert fitted.n_channels_ == N_CHANNELS
        assert fitted.n_features_in_ == N_CHANNELS
        assert fitted.centroid_converged_.all()
        assert fitted.centroid_n_iter_.shape == (2,)


class TestPipelineIntegration:
    """The Covariances to MDM baseline, end to end."""

    def test_pipeline_from_raw_epochs(self, rng: np.random.Generator) -> None:
        """The full classical baseline the syllabus's Block 2 asks for."""
        epochs = rng.standard_normal((40, 8, 400))
        epochs[20:] *= np.linspace(0.5, 2.0, 8)[None, :, None]
        labels = np.repeat([0, 1], 20)
        pipeline = make_pipeline(Covariances(estimator="oas"), MDM())
        pipeline.fit(epochs, labels)
        assert pipeline.predict(epochs).shape == (40,)

    def test_cross_val_score(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = separable
        scores = cross_val_score(MDM(), x, y, cv=StratifiedKFold(4))
        assert scores.shape == (4,)
        assert scores.mean() > 0.8

    def test_metric_settable_through_the_pipeline(self) -> None:
        pipeline = make_pipeline(Covariances(), MDM())
        pipeline.set_params(mdm__metric="logeuclid")
        assert pipeline.named_steps["mdm"].metric == "logeuclid"


# --------------------------------------------------------------------------- #
# 6. Validation and failure modes
# --------------------------------------------------------------------------- #


class TestValidation:
    """Misuse fails at the point of misuse."""

    def test_stein_is_refused_with_an_explanation(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Stein has a distance but no implemented mean.

        Substituting the AIRM mean would produce a classifier that reports one
        geometry and uses another -- a result that cannot be reproduced by
        anyone reading the configuration.
        """
        x, y = separable
        with pytest.raises(ValueError, match="no implemented"):
            MDM(metric="stein").fit(x, y)

    @pytest.mark.parametrize(
        ("kwargs", "pattern"),
        [
            ({"metric": "riemann"}, "metric must be one of"),
            ({"tol": 0.0}, "positive finite"),
            ({"max_iter": 0}, "positive integer"),
        ],
    )
    def test_invalid_parameters_rejected_at_fit(
        self,
        kwargs: dict,
        pattern: str,
        separable: tuple[np.ndarray, np.ndarray],
    ) -> None:
        x, y = separable
        with pytest.raises(ValueError, match=pattern):
            MDM(**kwargs).fit(x, y)

    def test_single_class_rejected(self, rng: np.random.Generator) -> None:
        """An unstratified split can produce a single-class fold."""
        x = random_spd(4, rng=rng, batch=20)
        with pytest.raises(ValueError, match="at least two classes"):
            MDM().fit(x, np.zeros(20, dtype=int))

    def test_length_mismatch_rejected(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        with pytest.raises(ValueError, match="labels"):
            MDM().fit(x, y[:-5])

    def test_two_dimensional_input_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="n_trials, n_channels, n_channels"):
            MDM().fit(rng.standard_normal((30, 16)), np.repeat([0, 1], 15))

    def test_error_suggests_the_covariances_step(
        self, rng: np.random.Generator
    ) -> None:
        with pytest.raises(ValueError, match="Covariances step"):
            MDM().fit(rng.standard_normal((30, 16)), np.repeat([0, 1], 15))

    def test_non_spd_input_rejected(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = separable
        bad = x.copy()
        bad[3] = np.diag(np.r_[np.ones(N_CHANNELS - 1), 0.0])
        with pytest.raises(NotPositiveDefiniteError):
            MDM().fit(bad, y)

    def test_channel_mismatch_rejected(
        self, separable: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        x, y = separable
        fitted = MDM().fit(x, y)
        with pytest.raises(ValueError, match="Expected 6 channels"):
            fitted.predict(random_spd(9, rng=rng, batch=4))

    @pytest.mark.parametrize(
        ("weights", "pattern"),
        [
            (np.ones(5), r"shape \(60,\)"),
            (-np.ones(60), "non-negative"),
        ],
    )
    def test_invalid_sample_weight_rejected(
        self,
        weights: np.ndarray,
        pattern: str,
        separable: tuple[np.ndarray, np.ndarray],
    ) -> None:
        x, y = separable
        with pytest.raises(ValueError, match=pattern):
            MDM().fit(x, y, sample_weight=weights)

    def test_non_convergence_is_reported(
        self,
        separable: tuple[np.ndarray, np.ndarray],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A displaced centroid must not pass silently.

        The resulting accuracy drop is indistinguishable from a modelling
        result unless the numerical cause is logged.
        """
        x, y = separable
        with caplog.at_level(logging.WARNING, logger="geoq.models.classical.mdm"):
            fitted = MDM(max_iter=1, tol=1e-16).fit(x, y)
        assert not fitted.centroid_converged_.all()
        assert "did not converge" in caplog.text

    def test_non_convergence_warning_can_be_silenced(
        self,
        separable: tuple[np.ndarray, np.ndarray],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        x, y = separable
        with caplog.at_level(logging.WARNING, logger="geoq.models.classical.mdm"):
            MDM(max_iter=1, tol=1e-16, warn_on_non_convergence=False).fit(x, y)
        assert "MDM:" not in caplog.text
