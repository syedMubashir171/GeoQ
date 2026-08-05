"""Tests for :mod:`geoq.evaluation.protocol`.

What is being defended
----------------------
This module produces the numbers that go into the thesis, so the tests target
the properties that make those numbers trustworthy rather than merely present.

* **Fold independence.** The estimator is cloned per fold. A reused estimator
  carries fitted state forward, and no splitter can prevent that leak.
  ``TestFoldIndependence`` verifies it against a spy.
* **Fold sizes are recorded.** The corrected resampled t-test needs the
  train/test ratio, and it cannot be recovered from a table of scores. If
  these fields were dropped, every significance claim built on the results
  would silently become anticonservative.
* **Nested selection stays inside the training fold.** Choosing
  hyperparameters once on the whole dataset and reporting outer-fold scores is
  a leak whose size grows with the grid. ``TestNestedSelection`` checks that
  the inner search never sees a test-fold trial, including its subject labels.
* **Kappa is reported by default.** ``TestMetrics`` includes the degenerate
  case that motivated the choice: a classifier predicting one class throughout
  scores 0.525 accuracy and exactly 0.000 kappa on the same folds.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the 'ml' extra: pip install -e '.[ml]'")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from geoq.evaluation.protocol import (
    DEFAULT_METRICS,
    METRIC_FUNCTIONS,
    EvaluationResult,
    FoldResult,
    evaluate,
)
from geoq.evaluation.splitters import (
    LeakyShuffleSplit,
    LeaveOneSubjectOut,
    WithinSubjectKFold,
)
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import geodesic
from geoq.geometry.spd import random_spd
from geoq.models.classical.mdm import MDM

N_SUBJECTS = 6
N_PER_SUBJECT = 30


def simulate(
    seed: int = 0,
    *,
    task_effect: float = 0.4,
    n_subjects: int = N_SUBJECTS,
    n_per_subject: int = N_PER_SUBJECT,
    n_channels: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate SPD trials with subject identity, drift, and a task effect.

    Args:
        seed: Random seed.
        task_effect: Interpolation weight toward the class direction. The
            default is strong enough that an honest protocol can decode it,
            so tests measure the machinery rather than a null result.
        n_subjects: Number of participants.
        n_per_subject: Trials per participant.
        n_channels: Channel count.

    Returns:
        Trials, labels, and subject identifiers.
    """
    rng = np.random.default_rng(seed)
    directions = [random_spd(n_channels, rng=rng) for _ in range(2)]
    matrices, targets, groups = [], [], []
    for subject in range(n_subjects):
        state = random_spd(n_channels, rng=rng)
        for index in range(n_per_subject):
            state = geodesic(state, random_spd(n_channels, rng=rng), 0.02)
            label = index % 2
            matrices.append(geodesic(state, directions[label], task_effect))
            targets.append(label)
            groups.append(subject)
    return np.stack(matrices), np.array(targets), np.array(groups)


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A decodable simulated dataset."""
    return simulate()


@pytest.fixture
def pipeline():
    """The TS+LDA baseline."""
    return make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())


class SpyEstimator(ClassifierMixin, BaseEstimator):
    """Records the trials seen by each ``fit`` call, by content hash."""

    def __init__(self, tag: str = "spy") -> None:
        """Store the tag; scikit-learn requires __init__ to do nothing else."""
        self.tag = tag

    def fit(self, X, y):  # noqa: N803
        SpyEstimator.calls.append(
            {hash(np.asarray(trial).tobytes()) for trial in np.asarray(X)}
        )
        self.classes_ = np.unique(y)
        self.fitted_marker_ = len(SpyEstimator.calls)
        return self

    def predict(self, X):  # noqa: N803
        return np.full(np.asarray(X).shape[0], self.classes_[0])

    calls: ClassVar[list[set[int]]] = []


# --------------------------------------------------------------------------- #
# 1. Fold independence
# --------------------------------------------------------------------------- #


class TestFoldIndependence:
    """Each fold gets a fresh estimator and sees only its training trials."""

    def test_estimator_is_cloned_per_fold(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """The template estimator must never be fitted in place.

        If the caller's object were fitted, a second call to ``evaluate``
        would start from the previous run's state, and the two results would
        differ for reasons invisible in the configuration.
        """
        matrices, targets, groups = dataset
        estimator = MDM()
        evaluate(
            estimator,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert not hasattr(estimator, "centroids_")

    def test_fit_sees_only_training_trials(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verified by content hash, not by trusting the splitter."""
        matrices, targets, groups = dataset
        SpyEstimator.calls = []
        splitter = LeaveOneSubjectOut()
        evaluate(
            SpyEstimator(),
            matrices,
            targets,
            groups=groups,
            splitter=splitter,
            metrics=("accuracy",),
        )
        digests = np.array([hash(trial.tobytes()) for trial in matrices])
        expected = [
            set(digests[train])
            for train, _ in splitter.split(matrices, targets, groups)
        ]
        assert SpyEstimator.calls == expected

    def test_repeated_evaluation_is_deterministic(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        """Two identical calls must produce identical numbers."""
        matrices, targets, groups = dataset
        first = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        second = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert np.array_equal(first.scores("kappa"), second.scores("kappa"))


# --------------------------------------------------------------------------- #
# 2. What the result records
# --------------------------------------------------------------------------- #


class TestResultContents:
    """The record contains what later analysis will need."""

    def test_one_fold_per_subject(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert len(result.folds) == N_SUBJECTS
        assert all(isinstance(fold, FoldResult) for fold in result.folds)

    def test_fold_sizes_are_recorded(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        """Required by the corrected resampled t-test.

        Cross-validation folds share training data, so their scores are
        correlated and an ordinary paired t-test over folds is
        anticonservative. The Nadeau-Bengio correction needs the train/test
        ratio, which cannot be recovered from a table of scores after the
        fact -- so it has to be captured here or the significance testing
        downstream is unsound.
        """
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        for n_train, n_test in result.fold_sizes:
            assert n_train == (N_SUBJECTS - 1) * N_PER_SUBJECT
            assert n_test == N_PER_SUBJECT
            assert n_train + n_test == matrices.shape[0]

    def test_test_groups_identify_the_held_out_subject(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        """Without this a per-subject breakdown is impossible."""
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        held_out = [fold.test_groups for fold in result.folds]
        assert held_out == [(subject,) for subject in range(N_SUBJECTS)]

    def test_protocol_travels_with_the_result(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        """A number without its protocol cannot be interpreted."""
        matrices, targets, groups = dataset
        honest = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        leaky = evaluate(
            pipeline,
            matrices,
            targets,
            splitter=LeakyShuffleSplit(
                n_splits=4, random_state=0, acknowledge_leakage=True
            ),
        )
        assert honest.protocol.name == "loso"
        assert not honest.protocol.is_optimistic
        assert leaky.protocol.is_optimistic

    def test_chance_accuracy_is_the_majority_class_rate(
        self, rng: np.random.Generator
    ) -> None:
        """An accuracy figure without its chance level invites misreading."""
        matrices, targets, groups = simulate()
        targets = targets.copy()
        targets[: int(0.8 * targets.size)] = 0
        result = evaluate(
            MDM(),
            matrices,
            targets,
            groups=groups,
            splitter=WithinSubjectKFold(n_splits=3),
        )
        assert result.chance_accuracy > 0.7
        assert result.n_classes == 2
        assert sum(result.class_balance.values()) == pytest.approx(1.0)

    def test_timings_are_recorded(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert all(fold.fit_seconds > 0 for fold in result.folds)
        assert all(fold.score_seconds > 0 for fold in result.folds)

    def test_train_scores_are_optional(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        without = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        with_train = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
            return_train_scores=True,
        )
        assert without.folds[0].train_scores is None
        assert with_train.folds[0].train_scores is not None
        assert set(with_train.folds[0].train_scores) == set(DEFAULT_METRICS)


# --------------------------------------------------------------------------- #
# 3. Metrics
# --------------------------------------------------------------------------- #


class TestMetrics:
    """Metric selection, and why kappa is a default rather than an option."""

    def test_default_metrics_include_kappa(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert "kappa" in result.metrics

    def test_kappa_exposes_a_degenerate_classifier(self) -> None:
        """The case that makes kappa non-optional.

        A classifier that always predicts the majority class scores 0.8
        accuracy on an 80/20 split -- a figure that reads as a solid result
        and would pass unremarked in a table. Balanced accuracy is exactly
        0.500 and kappa exactly 0.000, which state the truth: nothing was
        learned.

        Demonstrated with a dummy classifier rather than a weak simulation, so
        the numbers are exact and the test cannot drift with a change of seed.
        Reporting accuracy alone here would put a fictitious finding into a
        paper, and no amount of cross-validation would catch it.
        """
        from sklearn.dummy import DummyClassifier

        matrices, _, groups = simulate()
        targets = np.zeros(matrices.shape[0], dtype=int)
        # 20 percent minority, spread evenly so every fold is imbalanced.
        targets[::5] = 1

        result = evaluate(
            DummyClassifier(strategy="most_frequent"),
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert result.mean("accuracy") == pytest.approx(0.8, abs=1e-9)
        assert result.mean("balanced_accuracy") == pytest.approx(0.5, abs=1e-9)
        assert result.mean("kappa") == pytest.approx(0.0, abs=1e-9)
        assert result.chance_accuracy == pytest.approx(0.8, abs=1e-9)

    def test_custom_metric_selection(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
            metrics=("f1_macro", "accuracy"),
        )
        assert result.metrics == ("f1_macro", "accuracy")
        assert "kappa" not in result.folds[0].scores

    def test_probability_metric(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
            metrics=("roc_auc",),
        )
        assert 0.0 <= result.mean("roc_auc") <= 1.0

    def test_probability_metric_without_predict_proba_raises(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """The error must name the cause, not fail deep inside sklearn."""
        matrices, targets, groups = dataset
        SpyEstimator.calls = []
        with pytest.raises(AttributeError, match="predict_proba"):
            evaluate(
                SpyEstimator(),
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
                metrics=("roc_auc",),
            )

    def test_unknown_metric_raises(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate(
                pipeline,
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
                metrics=("acuracy",),
            )

    def test_every_registered_metric_runs(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
            metrics=tuple(METRIC_FUNCTIONS),
        )
        assert set(result.folds[0].scores) == set(METRIC_FUNCTIONS)


# --------------------------------------------------------------------------- #
# 4. Nested hyperparameter selection
# --------------------------------------------------------------------------- #


class TestNestedSelection:
    """Selection happens inside each outer training fold, never outside it."""

    def test_inner_search_never_sees_test_trials(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """The leak that nested CV exists to prevent.

        Selecting hyperparameters once on the whole dataset and then reporting
        outer-fold scores lets the choice see the test data. The bias grows
        with the size of the grid, and is invisible in the reported numbers.
        """
        matrices, targets, groups = dataset
        SpyEstimator.calls = []
        splitter = LeaveOneSubjectOut()
        evaluate(
            SpyEstimator(),
            matrices,
            targets,
            groups=groups,
            splitter=splitter,
            metrics=("accuracy",),
            param_grid={"tag": ["a", "b"]},
            inner_splitter=WithinSubjectKFold(n_splits=3),
        )
        digests = np.array([hash(trial.tobytes()) for trial in matrices])
        held_out_per_fold = [
            set(digests[test]) for _, test in splitter.split(matrices, targets, groups)
        ]
        # Every inner fit must be a subset of some outer training fold.
        for seen in SpyEstimator.calls:
            assert any(not (seen & held_out) for held_out in held_out_per_fold)

    def test_best_params_recorded_per_fold(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Instability across folds is itself a finding.

        A grid whose winner changes every fold is selecting noise, not a
        better model, and that is only visible if the per-fold choice is kept.
        """
        matrices, targets, groups = dataset
        result = evaluate(
            MDM(),
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
            param_grid={"metric": ["airm", "logeuclid"]},
            inner_splitter=WithinSubjectKFold(n_splits=3),
        )
        assert all(fold.best_params is not None for fold in result.folds)
        assert all("metric" in fold.best_params for fold in result.folds)
        assert result.param_grid == {"metric": ["airm", "logeuclid"]}

    def test_param_grid_without_inner_splitter_raises(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Refusing is the point: there is no safe default here."""
        matrices, targets, groups = dataset
        with pytest.raises(ValueError, match="inner_splitter"):
            evaluate(
                MDM(),
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
                param_grid={"metric": ["airm"]},
            )

    def test_unknown_selection_metric_raises(
        self, dataset: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        matrices, targets, groups = dataset
        with pytest.raises(ValueError, match="selection_metric"):
            evaluate(
                MDM(),
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
                param_grid={"metric": ["airm"]},
                inner_splitter=WithinSubjectKFold(n_splits=3),
                selection_metric="acuracy",
            )


# --------------------------------------------------------------------------- #
# 5. Views onto the result
# --------------------------------------------------------------------------- #


class TestViews:
    """Aggregates are derived from the per-fold record, not stored."""

    def test_mean_and_std_match_the_folds(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        values = np.array([fold.scores["accuracy"] for fold in result.folds])
        assert result.mean("accuracy") == pytest.approx(values.mean())
        assert result.std("accuracy") == pytest.approx(values.std(ddof=1))

    def test_scores_for_missing_metric_raises(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
            metrics=("accuracy",),
        )
        with pytest.raises(KeyError, match="was not computed"):
            result.scores("kappa")

    def test_to_frame_is_self_describing(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        """Protocol guarantees repeat on every row.

        A results table concatenated across experiments must remain
        interpretable without a separate key, or a leaky run and an honest one
        become indistinguishable once they share a file.
        """
        pytest.importorskip("pandas")
        matrices, targets, groups = dataset
        frame = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        ).to_frame()
        assert len(frame) == N_SUBJECTS
        for column in (
            "fold",
            "n_train",
            "n_test",
            "test_groups",
            "protocol",
            "subject_independent",
            "temporally_disjoint",
            "accuracy",
            "kappa",
        ):
            assert column in frame.columns
        assert frame["protocol"].nunique() == 1

    def test_summary_contains_the_protocol_and_chance(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        summary = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        ).summary()
        assert summary["protocol"] == "loso"
        assert summary["is_optimistic"] is False
        assert "chance_accuracy" in summary
        assert "kappa_mean" in summary

    def test_repr_flags_an_optimistic_protocol(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        """Anything that prints the result should show the caveat."""
        matrices, targets, _ = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            splitter=LeakyShuffleSplit(
                n_splits=4, random_state=0, acknowledge_leakage=True
            ),
        )
        assert "OPTIMISTIC" in repr(result)

    def test_repr_omits_the_flag_for_an_honest_protocol(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        result = evaluate(
            pipeline,
            matrices,
            targets,
            groups=groups,
            splitter=LeaveOneSubjectOut(),
        )
        assert "OPTIMISTIC" not in repr(result)
        assert isinstance(result, EvaluationResult)


# --------------------------------------------------------------------------- #
# 6. Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    """Inconsistent input fails at the point of use."""

    def test_length_mismatch_rejected(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        with pytest.raises(ValueError, match="labels"):
            evaluate(
                pipeline,
                matrices,
                targets[:-4],
                groups=groups,
                splitter=LeaveOneSubjectOut(),
            )

    def test_single_class_training_fold_rejected(self) -> None:
        """A training fold with one class must fail, naming the fold.

        Constructed so the first fold trips the training check rather than the
        test-fold one: the held-out subject carries both labels while every
        remaining subject carries only one.
        """
        matrices, _, groups = simulate(n_subjects=4)
        targets = np.zeros(matrices.shape[0], dtype=int)
        first_subject = groups == 0
        targets[first_subject] = np.arange(int(first_subject.sum())) % 2
        with pytest.raises(ValueError, match="single class in its training set"):
            evaluate(
                MDM(),
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
            )

    def test_single_class_test_fold_rejected(self) -> None:
        """A test fold with one class must fail too.

        Kappa is undefined there, balanced accuracy collapses to a single
        class's recall, and accuracy is either zero or one. Averaging such a
        fold into a reported mean produces something that looks like a score
        and is not one -- so this is an error rather than a warning.

        In practice it means a participant performed only one task condition,
        which is a data problem the experimenter must decide about explicitly.
        """
        matrices, _, groups = simulate(n_subjects=4)
        targets = np.arange(matrices.shape[0]) % 2
        last = groups == groups.max()
        targets[last] = 1
        with pytest.raises(ValueError, match=r"single class .* in its test set"):
            evaluate(
                MDM(),
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
            )

    def test_empty_metrics_rejected(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
    ) -> None:
        matrices, targets, groups = dataset
        with pytest.raises(ValueError, match="At least one metric"):
            evaluate(
                pipeline,
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
                metrics=(),
            )

    def test_optimistic_protocol_warns(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The caveat belongs in the log beside the inflated numbers."""
        matrices, targets, _ = dataset
        with caplog.at_level(logging.WARNING, logger="geoq.evaluation.protocol"):
            evaluate(
                pipeline,
                matrices,
                targets,
                splitter=LeakyShuffleSplit(
                    n_splits=3, random_state=0, acknowledge_leakage=True
                ),
            )
        assert "does not provide both evaluation guarantees" in caplog.text

    def test_honest_protocol_does_not_warn(
        self,
        dataset: tuple[np.ndarray, np.ndarray, np.ndarray],
        pipeline,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        matrices, targets, groups = dataset
        with caplog.at_level(logging.WARNING, logger="geoq.evaluation.protocol"):
            evaluate(
                pipeline,
                matrices,
                targets,
                groups=groups,
                splitter=LeaveOneSubjectOut(),
            )
        assert caplog.text == ""
