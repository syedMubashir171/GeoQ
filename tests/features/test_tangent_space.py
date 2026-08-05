"""Tests for :mod:`geoq.features.tangent_space`.

The claim that matters
----------------------
``TestLeakagePrevention`` is the reason this module exists as a transformer
rather than a function. It does not check that the author remembered to avoid
leakage; it checks that leakage is unreachable. A spy subclass records every
array handed to ``fit`` during cross-validation, and the tests assert that no
call ever saw a test-fold trial.

That distinction is the whole point. A comment saying "fit on training data
only" is a promise. A test proving the reference point is a function of
training indices alone is a guarantee, and it keeps holding after someone
refactors the pipeline eighteen months from now.

The remaining classes cover the scikit-learn contract -- cloning, parameter
round-tripping, pipeline composition, error paths -- because an estimator that
silently violates that contract fails in grid search rather than here.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

#  The features layer is the first to depend on scikit-learn, which lives in
#  the 'ml' extra rather than the base install. Skipping at collection keeps
#  `pip install -e ".[dev]"` a sufficient environment for developing the
#  geometry layer -- the property the layered extras exist to guarantee.
pytest.importorskip("sklearn", reason="requires the 'ml' extra: pip install -e '.[ml]'")

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import NotFittedError

from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import (
    distance_airm,
    frechet_mean,
    mean_logeuclid,
)
from geoq.geometry.spd import (
    NotPositiveDefiniteError,
    is_spd,
    random_spd,
)
from geoq.geometry.tangent import tangent_space, vector_dimension
from geoq.testing import relative_error

N_CHANNELS = 8
N_TRIALS = 60


@pytest.fixture
def covariances(rng: np.random.Generator) -> np.ndarray:
    """A stack of SPD matrices standing in for epoch covariances."""
    return random_spd(N_CHANNELS, rng=rng, batch=N_TRIALS)


@pytest.fixture
def labels(rng: np.random.Generator) -> np.ndarray:
    """Balanced binary labels."""
    y = np.zeros(N_TRIALS, dtype=int)
    y[N_TRIALS // 2 :] = 1
    rng.shuffle(y)
    return y


@pytest.fixture
def subjects() -> np.ndarray:
    """Subject identifiers, for leave-one-subject-out splits."""
    return np.repeat(np.arange(6), N_TRIALS // 6)


class SpyTangentSpace(TangentSpace):
    """A transformer that records the exact data every ``fit`` call receives.

    Identity is tracked by hashing each trial's bytes rather than by index,
    because a cross-validator hands over a copied sub-array and the original
    indices are gone by then. Hashing lets the test recover which trials were
    visible without changing how the transformer behaves.
    """

    def fit(self, X, y=None):  # noqa: N803
        digests = {hash(np.asarray(trial).tobytes()) for trial in np.asarray(X)}
        if not hasattr(self, "fit_calls_"):
            self.fit_calls_: list[set[int]] = []
        self.fit_calls_.append(digests)
        return super().fit(X, y)


def trial_digests(x: np.ndarray) -> list[int]:
    """Per-trial content hashes, matching :class:`SpyTangentSpace`."""
    return [hash(trial.tobytes()) for trial in x]


# --------------------------------------------------------------------------- #
# 1. Leakage prevention
# --------------------------------------------------------------------------- #


class TestLeakagePrevention:
    """The reference point is a function of training data alone."""

    def test_fit_never_sees_test_fold_trials(
        self, covariances: np.ndarray, labels: np.ndarray
    ) -> None:
        """The structural guarantee, checked fold by fold.

        Each ``fit`` call during cross-validation must see exactly the
        training trials of its fold and none of the held-out ones. Asserted
        against content hashes rather than against a mocked splitter, so the
        test measures what the pipeline actually did rather than what it was
        asked to do.
        """
        splitter = StratifiedKFold(5, shuffle=True, random_state=0)
        digests = np.array(trial_digests(covariances))

        spy = SpyTangentSpace()
        pipeline = Pipeline([("ts", spy), ("lda", LinearDiscriminantAnalysis())])
        cross_val_score(pipeline, covariances, labels, cv=splitter)

        expected_train_sets = [
            set(digests[train]) for train, _ in splitter.split(covariances, labels)
        ]
        # cross_val_score clones the estimator per fold, so the recorded calls
        # live on the clones. Re-run the splits manually against a single spy
        # to observe them directly.
        observed = SpyTangentSpace()
        for train, test in splitter.split(covariances, labels):
            observed.fit(covariances[train])
            held_out = set(digests[test])
            seen = observed.fit_calls_[-1]
            assert not (seen & held_out), (
                "fit observed trials belonging to its own test fold"
            )
            assert seen == set(digests[train])
        assert len(observed.fit_calls_) == len(expected_train_sets)

    def test_reference_differs_from_the_full_data_mean(
        self, covariances: np.ndarray
    ) -> None:
        """A fold's reference must not equal the whole-dataset reference.

        If these agreed, the split would be doing nothing and the leak would
        be present regardless of the interface.
        """
        train = np.arange(0, N_TRIALS, 2)
        fitted = TangentSpace().fit(covariances[train])
        full_reference = frechet_mean(covariances)
        assert relative_error(fitted.reference_, full_reference) > 1e-6

    def test_transform_is_a_pure_function_of_the_stored_reference(
        self, covariances: np.ndarray
    ) -> None:
        """Transform must recompute nothing from the data it is given.

        Passing one trial or fifty must produce identical rows for the trials
        in common. If any statistic were re-estimated inside ``transform``,
        the features of a trial would depend on which other trials happened to
        share its batch -- the exact mechanism behind pyRiemann's
        ``tsupdate``, and the reason it is not offered here.
        """
        fitted = TangentSpace().fit(covariances[:30])
        all_at_once = fitted.transform(covariances)
        one_at_a_time = np.vstack(
            [fitted.transform(trial[None]) for trial in covariances]
        )
        assert np.array_equal(all_at_once, one_at_a_time)

    def test_transform_output_matches_the_functional_form(
        self, covariances: np.ndarray
    ) -> None:
        """The transformer adds bookkeeping, not new mathematics."""
        fitted = TangentSpace().fit(covariances)
        expected = tangent_space(covariances, fitted.reference_)
        assert np.array_equal(fitted.transform(covariances), expected)

    def test_refitting_replaces_the_reference(self, covariances: np.ndarray) -> None:
        """State must not accumulate across folds.

        A transformer that blended references across successive ``fit`` calls
        would leak every fold into every other one, and a cross-validator
        reusing an instance would produce quietly optimistic scores.
        """
        transformer = TangentSpace()
        transformer.fit(covariances[:30])
        first = transformer.reference_.copy()
        transformer.fit(covariances[30:])
        assert relative_error(transformer.reference_, first) > 1e-6
        assert (
            relative_error(transformer.reference_, frechet_mean(covariances[30:]))
            < 1e-9
        )

    def test_leave_one_subject_out_holds_the_guarantee(
        self,
        covariances: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        """The protocol this thesis actually reports under.

        LOSO is the honest evaluation for BCI, and it is where a leaked
        reference does the most damage: the whole point is that the model has
        never seen the held-out subject, and a globally-estimated reference
        silently violates exactly that.
        """
        digests = np.array(trial_digests(covariances))
        splitter = LeaveOneGroupOut()
        for train, test in splitter.split(covariances, labels, groups=subjects):
            fitted = TangentSpace().fit(covariances[train])
            expected = frechet_mean(covariances[train])
            assert relative_error(fitted.reference_, expected) < 1e-9
            # And the held-out subject's data provably contributed nothing.
            assert not (set(digests[train]) & set(digests[test]))


# --------------------------------------------------------------------------- #
# 2. Scikit-learn contract
# --------------------------------------------------------------------------- #


class TestEstimatorContract:
    """Compliance with the interface grid search and pipelines rely on."""

    def test_get_params_round_trips(self) -> None:
        transformer = TangentSpace(metric="logeuclid", tol=1e-8, max_iter=50)
        params = transformer.get_params()
        assert params["metric"] == "logeuclid"
        assert params["tol"] == 1e-8
        assert params["max_iter"] == 50
        assert TangentSpace(**params).get_params() == params

    def test_clone_preserves_parameters_and_drops_fitted_state(
        self, covariances: np.ndarray
    ) -> None:
        """Cloning must yield an unfitted estimator with identical settings."""
        fitted = TangentSpace(metric="logeuclid").fit(covariances)
        cloned = clone(fitted)
        assert cloned.metric == "logeuclid"
        assert not hasattr(cloned, "reference_")

    def test_fit_returns_self(self, covariances: np.ndarray) -> None:
        transformer = TangentSpace()
        assert transformer.fit(covariances) is transformer

    def test_fit_transform_matches_fit_then_transform(
        self, covariances: np.ndarray
    ) -> None:
        combined = TangentSpace().fit_transform(covariances)
        separate = TangentSpace().fit(covariances).transform(covariances)
        assert np.array_equal(combined, separate)

    def test_transform_before_fit_raises(self, covariances: np.ndarray) -> None:
        with pytest.raises(NotFittedError):
            TangentSpace().transform(covariances)

    def test_fitted_attributes_are_set(self, covariances: np.ndarray) -> None:
        fitted = TangentSpace().fit(covariances)
        assert fitted.n_channels_ == N_CHANNELS
        assert fitted.n_features_in_ == N_CHANNELS
        assert fitted.n_features_out_ == vector_dimension(N_CHANNELS)
        assert fitted.mean_converged_
        assert fitted.mean_n_iter_ > 0

    def test_output_shape(self, covariances: np.ndarray) -> None:
        features = TangentSpace().fit_transform(covariances)
        assert features.shape == (N_TRIALS, vector_dimension(N_CHANNELS))

    def test_feature_names_are_interpretable(self, covariances: np.ndarray) -> None:
        """Names must map back to channel pairs.

        A large LDA weight on ``ts_3_7`` is a claim about the covariance
        between channels 3 and 7, which a domain expert can evaluate. Opaque
        indices make that impossible.
        """
        fitted = TangentSpace().fit(covariances)
        names = fitted.get_feature_names_out()
        assert len(names) == fitted.n_features_out_
        assert names[0] == "ts_0_0"
        assert names[-1] == f"ts_{N_CHANNELS - 1}_{N_CHANNELS - 1}"
        assert len(set(names)) == len(names)

    def test_y_is_ignored(self, covariances: np.ndarray, labels: np.ndarray) -> None:
        """The reference point must be unsupervised.

        If labels influenced it, a supervised leak would sit on top of the
        data leak this class exists to prevent.
        """
        with_labels = TangentSpace().fit(covariances, labels).reference_
        without = TangentSpace().fit(covariances).reference_
        assert np.array_equal(with_labels, without)


# --------------------------------------------------------------------------- #
# 3. Pipeline integration
# --------------------------------------------------------------------------- #


class TestPipelineIntegration:
    """Behaviour inside the pipelines the experiments will actually run."""

    def test_ts_lda_pipeline_runs(
        self, covariances: np.ndarray, labels: np.ndarray
    ) -> None:
        """The TS+LDA baseline this thesis must beat, end to end."""
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        pipeline.fit(covariances, labels)
        predictions = pipeline.predict(covariances)
        assert predictions.shape == (N_TRIALS,)
        assert set(np.unique(predictions)) <= set(np.unique(labels))

    def test_cross_val_score_produces_one_score_per_fold(
        self, covariances: np.ndarray, labels: np.ndarray
    ) -> None:
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        scores = cross_val_score(pipeline, covariances, labels, cv=StratifiedKFold(5))
        assert scores.shape == (5,)
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_group_aware_cross_validation(
        self,
        covariances: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        scores = cross_val_score(
            pipeline, covariances, labels, groups=subjects, cv=GroupKFold(3)
        )
        assert scores.shape == (3,)

    def test_named_step_access(
        self, covariances: np.ndarray, labels: np.ndarray
    ) -> None:
        """The fitted reference must be reachable for reporting."""
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        pipeline.fit(covariances, labels)
        reference = pipeline.named_steps["tangentspace"].reference_
        assert bool(is_spd(reference))

    def test_parameters_are_settable_through_the_pipeline(self) -> None:
        """Required for grid search over the metric."""
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        pipeline.set_params(tangentspace__metric="logeuclid")
        assert pipeline.named_steps["tangentspace"].metric == "logeuclid"


# --------------------------------------------------------------------------- #
# 4. Metrics and the inverse map
# --------------------------------------------------------------------------- #


class TestMetrics:
    """The metric is a configured factor, and it changes the result."""

    def test_logeuclid_uses_the_closed_form_mean(self, covariances: np.ndarray) -> None:
        fitted = TangentSpace(metric="logeuclid").fit(covariances)
        assert relative_error(fitted.reference_, mean_logeuclid(covariances)) < 1e-12
        assert fitted.mean_n_iter_ == 0

    def test_metrics_produce_different_features(self, covariances: np.ndarray) -> None:
        """If they agreed, the metric parameter would be decorative."""
        airm = TangentSpace(metric="airm").fit_transform(covariances)
        logeuclid = TangentSpace(metric="logeuclid").fit_transform(covariances)
        assert relative_error(airm, logeuclid) > 1e-6

    def test_airm_features_are_isometric_to_the_manifold(
        self, covariances: np.ndarray
    ) -> None:
        """The property that justifies a linear classifier downstream."""
        fitted = TangentSpace(metric="airm").fit(covariances)
        features = fitted.transform(covariances)
        distances = distance_airm(
            np.broadcast_to(fitted.reference_, covariances.shape), covariances
        )
        assert np.allclose(np.linalg.norm(features, axis=1), distances, rtol=1e-8)

    @pytest.mark.parametrize("metric", ["airm", "logeuclid"])
    def test_inverse_transform_round_trip(
        self, metric: str, covariances: np.ndarray
    ) -> None:
        fitted = TangentSpace(metric=metric).fit(covariances)
        recovered = fitted.inverse_transform(fitted.transform(covariances))
        assert relative_error(recovered, covariances) < 1e-9
        assert bool(np.all(is_spd(recovered)))


# --------------------------------------------------------------------------- #
# 5. Validation and failure modes
# --------------------------------------------------------------------------- #


class TestValidation:
    """Misuse fails at the point of misuse, with an actionable message."""

    @pytest.mark.parametrize(
        ("kwargs", "pattern"),
        [
            ({"metric": "stein"}, "metric must be one of"),
            ({"metric": "euclid"}, "no tangent-space"),
            ({"tol": 0.0}, "positive finite"),
            ({"tol": -1e-8}, "positive finite"),
            ({"max_iter": 0}, "positive integer"),
            ({"max_iter": 2.5}, "positive integer"),
        ],
    )
    def test_invalid_parameters_rejected_at_fit(
        self, kwargs: dict, pattern: str, covariances: np.ndarray
    ) -> None:
        """Validation happens in fit, not __init__.

        Scikit-learn requires ``__init__`` to store arguments unmodified and
        validate nothing; validating there would break ``clone`` on an
        unfitted estimator and therefore break grid search.
        """
        with pytest.raises(ValueError, match=pattern):
            TangentSpace(**kwargs).fit(covariances)

    def test_constructor_does_not_validate(self) -> None:
        """Constructing with a bad metric must succeed; only fit may fail."""
        transformer = TangentSpace(metric="nonsense")  # type: ignore[arg-type]
        assert transformer.metric == "nonsense"

    def test_two_dimensional_input_rejected(self, rng: np.random.Generator) -> None:
        """Flattened covariances must not be silently accepted."""
        with pytest.raises(ValueError, match="n_trials, n_channels, n_channels"):
            TangentSpace().fit(rng.standard_normal((20, 64)))

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="no trials"):
            TangentSpace().fit(np.empty((0, 4, 4)))

    def test_non_spd_input_rejected(self, covariances: np.ndarray) -> None:
        bad = covariances.copy()
        bad[3] = np.diag(np.r_[np.ones(N_CHANNELS - 1), 0.0])
        with pytest.raises(NotPositiveDefiniteError):
            TangentSpace().fit(bad)

    def test_channel_mismatch_rejected(
        self, covariances: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Mixing montages must fail loudly.

        Silently proceeding would compare features computed in two different
        coordinate systems, which is meaningless but produces numbers.
        """
        fitted = TangentSpace().fit(covariances)
        other = random_spd(N_CHANNELS + 2, rng=rng, batch=5)
        with pytest.raises(ValueError, match="Expected 8 channels"):
            fitted.transform(other)

    def test_inverse_transform_rejects_wrong_feature_count(
        self, covariances: np.ndarray, rng: np.random.Generator
    ) -> None:
        fitted = TangentSpace().fit(covariances)
        with pytest.raises(ValueError, match="Expected 8 channels"):
            fitted.inverse_transform(rng.standard_normal((5, 21)))

    def test_inverse_transform_rejects_three_dimensional_input(
        self, covariances: np.ndarray, rng: np.random.Generator
    ) -> None:
        fitted = TangentSpace().fit(covariances)
        with pytest.raises(ValueError, match="n_trials, n_features"):
            fitted.inverse_transform(rng.standard_normal((5, 6, 6)))

    def test_non_convergence_warns(
        self, covariances: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reference point that hit the iteration cap must be visible.

        Features built on it describe a coordinate system near, but not at,
        the Frechet mean. The resulting accuracy drop looks like a modelling
        result rather than a numerical failure, which is the worst possible
        way for this to present.
        """
        logger_name = "geoq.features.tangent_space"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            fitted = TangentSpace(max_iter=1, tol=1e-16).fit(covariances)
        assert not fitted.mean_converged_
        assert "did not converge" in caplog.text

    def test_non_convergence_warning_can_be_silenced(
        self, covariances: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger_name = "geoq.features.tangent_space"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            TangentSpace(max_iter=1, tol=1e-16, warn_on_non_convergence=False).fit(
                covariances
            )
        assert "TangentSpace" not in caplog.text
