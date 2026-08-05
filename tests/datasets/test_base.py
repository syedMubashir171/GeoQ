"""Tests for :mod:`geoq.datasets.base`.

What is being defended
----------------------
* **Alignment.** ``TestValidation`` checks that a label or subject vector out
  of step with the trials is rejected. This is the quietest catastrophic bug
  in the framework: every evaluation guarantee continues to hold formally
  while measuring nothing, because the splitter is grouping by a subject
  vector that no longer describes the data.
* **The generator contains what it claims.** ``TestSyntheticStructure``
  verifies each of the three ingredients independently -- subject identity is
  decodable, adjacent trials are more similar than distant ones, and
  covariance conditioning matches the real regime. Every leakage and
  calibration measurement in this framework rests on the generator, so a
  generator that quietly lacked one of them would invalidate those results
  while all their tests still passed.
* **A true null is really null.** ``test_zero_task_effect_is_undecodable``
  pins the case that false-positive rates are measured on.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from geoq.datasets.base import (
    DATASETS,
    EEGDataset,
    load_dataset,
    make_synthetic_eeg,
    register_dataset,
)


def small(**overrides: Any) -> EEGDataset:
    """A quick synthetic dataset for structural tests."""
    defaults = {
        "n_subjects": 4,
        "n_trials_per_subject": 20,
        "n_channels": 8,
        "n_times": 128,
        "seed": 0,
    }
    return make_synthetic_eeg(**{**defaults, **overrides})


@pytest.fixture
def dataset() -> EEGDataset:
    """A small synthetic dataset."""
    return small()


def valid_arrays(
    n_trials: int = 12, n_channels: int = 4, n_times: int = 64
) -> dict[str, Any]:
    """Minimal valid constructor arguments."""
    rng = np.random.default_rng(0)
    return {
        "epochs": rng.standard_normal((n_trials, n_channels, n_times)),
        "labels": np.tile([0, 1], n_trials // 2),
        "subjects": np.repeat([1, 2], n_trials // 2),
        "sampling_rate": 250.0,
    }


# --------------------------------------------------------------------------- #
# 1. Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    """Invariants enforced at construction."""

    def test_valid_dataset_constructs(self) -> None:
        assert EEGDataset(**valid_arrays()).n_trials == 12

    def test_two_dimensional_epochs_rejected(self) -> None:
        arrays = valid_arrays()
        arrays["epochs"] = arrays["epochs"].reshape(12, -1)
        with pytest.raises(ValueError, match="n_trials, n_channels, n_times"):
            EEGDataset(**arrays)

    def test_transposed_epochs_are_diagnosed(self) -> None:
        """The error must name the convention, not just the shape.

        A ``(n_trials, n_times, n_channels)`` array has the right rank and the
        wrong meaning; without the hint it produces a covariance of the wrong
        size several layers later.
        """
        arrays = valid_arrays(n_channels=4, n_times=256)
        arrays["epochs"] = np.swapaxes(arrays["epochs"], 1, 2)
        with pytest.raises(ValueError) as excinfo:
            EEGDataset(**arrays)
        assert "needs transposing" in str(excinfo.value)

    def test_non_finite_rejected_with_the_trial_index(self) -> None:
        arrays = valid_arrays()
        arrays["epochs"][7, 2, 30] = np.nan
        with pytest.raises(ValueError, match="trial 7"):
            EEGDataset(**arrays)

    @pytest.mark.parametrize("field", ["labels", "subjects"])
    def test_misaligned_vector_rejected(self, field: str) -> None:
        """The quietest catastrophic bug this container prevents.

        A subject vector shorter or longer than the trials leaves every
        subject-independence check formally passing while the splitter groups
        by identifiers that no longer describe the data.
        """
        arrays = valid_arrays()
        arrays[field] = arrays[field][:-2]
        with pytest.raises(ValueError, match="align with epochs"):
            EEGDataset(**arrays)

    def test_misaligned_sessions_rejected(self) -> None:
        arrays = valid_arrays()
        arrays["sessions"] = np.zeros(5)
        with pytest.raises(ValueError, match="sessions must be empty"):
            EEGDataset(**arrays)

    def test_empty_sessions_allowed(self) -> None:
        assert EEGDataset(**valid_arrays()).sessions.size == 0

    def test_single_class_rejected(self) -> None:
        arrays = valid_arrays()
        arrays["labels"] = np.zeros(12, dtype=int)
        with pytest.raises(ValueError, match="Only one class"):
            EEGDataset(**arrays)

    def test_no_trials_rejected(self) -> None:
        arrays = valid_arrays()
        arrays["epochs"] = np.zeros((0, 4, 64))
        arrays["labels"] = np.array([])
        arrays["subjects"] = np.array([])
        with pytest.raises(ValueError, match="no trials"):
            EEGDataset(**arrays)

    @pytest.mark.parametrize("rate", [0.0, -250.0])
    def test_non_positive_sampling_rate_rejected(self, rate: float) -> None:
        with pytest.raises(ValueError, match="sampling_rate"):
            EEGDataset(**{**valid_arrays(), "sampling_rate": rate})

    def test_channel_name_count_must_match(self) -> None:
        with pytest.raises(ValueError, match="channel_names"):
            EEGDataset(**valid_arrays(), channel_names=("A", "B"))

    def test_rank_deficient_epochs_rejected(self) -> None:
        """``n_times <= n_channels`` makes the SPD manifold undefined."""
        with pytest.raises(ValueError, match="does not exceed n_channels"):
            EEGDataset(**valid_arrays(n_channels=64, n_times=64))

    def test_short_epochs_warn_but_are_accepted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Conditioning risk is a fact about the data, not a defect."""
        with caplog.at_level(logging.WARNING, logger="geoq.datasets.base"):
            EEGDataset(**valid_arrays(n_channels=40, n_times=60))
        assert "ill-conditioned" in caplog.text

    def test_ample_epochs_do_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="geoq.datasets.base"):
            EEGDataset(**valid_arrays())
        assert caplog.text == ""

    def test_epochs_are_coerced_to_float64(self) -> None:
        arrays = valid_arrays()
        arrays["epochs"] = arrays["epochs"].astype(np.float32)
        assert EEGDataset(**arrays).epochs.dtype == np.float64

    def test_is_frozen(self, dataset: EEGDataset) -> None:
        with pytest.raises(AttributeError):
            dataset.name = "changed"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 2. Properties and selection
# --------------------------------------------------------------------------- #


class TestProperties:
    """Derived quantities used across the framework."""

    def test_shape_properties(self, dataset: EEGDataset) -> None:
        assert dataset.n_trials == 80
        assert dataset.n_channels == 8
        assert dataset.n_times == 128
        assert dataset.n_subjects == 4
        assert len(dataset) == 80

    def test_duration_and_ratio(self, dataset: EEGDataset) -> None:
        assert dataset.duration == pytest.approx(128 / 250.0)
        assert dataset.samples_per_channel == pytest.approx(16.0)

    def test_class_balance_sums_to_one(self, dataset: EEGDataset) -> None:
        assert sum(dataset.class_balance.values()) == pytest.approx(1.0)
        assert dataset.chance_accuracy == pytest.approx(0.5)

    def test_repr_is_informative(self, dataset: EEGDataset) -> None:
        text = repr(dataset)
        assert "80 trials" in text and "8ch" in text and "4 subjects" in text

    def test_summary_contains_interpretive_context(self, dataset: EEGDataset) -> None:
        """Chance level and conditioning ratio are what make results readable.

        An accuracy without the first is uninterpretable; a geodesic distance
        without the second has unknown precision.
        """
        summary = dataset.summary()
        assert summary["chance_accuracy"] == pytest.approx(0.5)
        assert summary["samples_per_channel"] == pytest.approx(16.0)
        assert summary["trials_per_subject"] == {str(s): 20 for s in range(1, 5)}
        assert summary["synthetic"] is True


class TestSelection:
    """Subsetting preserves alignment and metadata."""

    def test_subset_by_mask(self, dataset: EEGDataset) -> None:
        subset = dataset.subset(dataset.subjects == 2)
        assert subset.n_trials == 20
        assert set(subset.subjects.tolist()) == {2}
        assert subset.sampling_rate == dataset.sampling_rate
        assert subset.channel_names == dataset.channel_names

    def test_subset_keeps_labels_aligned(self, dataset: EEGDataset) -> None:
        """The property that makes subsetting safe.

        Trials, labels and subjects must be reindexed together; slicing one
        without the others is exactly the misalignment the container rejects
        at construction.
        """
        index = np.array([5, 1, 40, 22])
        subset = dataset.subset(index)
        assert np.array_equal(subset.labels, dataset.labels[index])
        assert np.array_equal(subset.subjects, dataset.subjects[index])
        assert np.array_equal(subset.epochs, dataset.epochs[index])

    def test_select_subjects(self, dataset: EEGDataset) -> None:
        subset = dataset.select_subjects([1, 3])
        assert subset.n_subjects == 2
        assert sorted(set(subset.subjects.tolist())) == [1, 3]

    def test_missing_subject_rejected(self, dataset: EEGDataset) -> None:
        """A missing subject must be an error.

        Silently returning fewer subjects would change the evaluation without
        changing the configuration.
        """
        with pytest.raises(ValueError, match="not present"):
            dataset.select_subjects([1, 99])

    def test_by_subject_covers_everything_once(self, dataset: EEGDataset) -> None:
        pieces = list(dataset.by_subject())
        assert [subject for subject, _ in pieces] == [1, 2, 3, 4]
        assert sum(len(piece) for _, piece in pieces) == dataset.n_trials


# --------------------------------------------------------------------------- #
# 3. The synthetic generator
# --------------------------------------------------------------------------- #


class TestSyntheticBasics:
    """Shape, determinism, and defaults."""

    def test_defaults_match_bci_iv_2a(self) -> None:
        """So a pipeline developed on simulation meets no surprises on real data."""
        data = make_synthetic_eeg()
        assert data.n_subjects == 9
        assert data.n_channels == 22
        assert data.n_times == 500
        assert data.sampling_rate == 250.0
        assert data.duration == pytest.approx(2.0)

    def test_deterministic(self) -> None:
        first = small(seed=7)
        second = small(seed=7)
        assert np.array_equal(first.epochs, second.epochs)
        assert np.array_equal(first.labels, second.labels)

    def test_different_seeds_differ(self) -> None:
        assert not np.allclose(small(seed=1).epochs, small(seed=2).epochs)

    def test_classes_are_balanced(self, dataset: EEGDataset) -> None:
        """Every subject has equal class counts.

        Imbalance would make accuracy and balanced accuracy diverge for
        reasons unrelated to whatever property a test is checking.
        """
        for _, subject_data in dataset.by_subject():
            counts = np.unique(subject_data.labels, return_counts=True)[1]
            assert counts.min() == counts.max()

    def test_multiclass(self) -> None:
        data = small(n_classes=4, n_trials_per_subject=24)
        assert data.classes.size == 4

    def test_sessions_are_assigned(self, dataset: EEGDataset) -> None:
        assert set(dataset.sessions.tolist()) == {0, 1}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_subjects": 0},
            {"n_classes": 1},
            {"task_effect": -0.1},
            {"source_decay": 0.0},
            {"n_trials_per_subject": 1},
        ],
    )
    def test_invalid_arguments_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            small(**kwargs)


class TestSyntheticStructure:
    """Each of the three ingredients is verified independently.

    Every leakage and calibration measurement in this framework rests on this
    generator. One missing ingredient would invalidate those results while
    their own tests continued to pass, because the property under test would
    simply have nothing to detect.
    """

    def test_subject_identity_is_decodable(self) -> None:
        """Ingredient one: the signal a leaky split exploits.

        Without a subject-specific component, a shuffled split and a
        subject-independent one give the same answer and the leak cannot be
        measured at all.
        """
        pytest.importorskip("sklearn")
        from sklearn.model_selection import cross_val_score

        from geoq.features.covariance import Covariances
        from geoq.models.classical.mdm import MDM

        data = small(n_subjects=4, n_trials_per_subject=24, task_effect=0.0)
        covariances = Covariances(
            estimator="oas", audit_conditioning=False
        ).fit_transform(data.epochs)
        accuracy = cross_val_score(MDM(), covariances, data.subjects, cv=3).mean()
        assert accuracy > 0.6

    def test_adjacent_trials_are_more_similar_than_distant_ones(self) -> None:
        """Ingredient two: the autocorrelation that makes shuffling unsound.

        ``drift`` is the innovation weight of the AR(1) state, so the
        autoregressive coefficient is ``1 - drift`` and *smaller* values give
        stronger temporal structure. Measured at ``drift = 0.05``, adjacent
        trials sit about half the geodesic distance apart that trials thirty
        apart do.

        The epochs are deliberately long and the averages taken over many
        pairs: at 128 samples on eight channels the per-trial covariance is
        noisy enough to swamp a lag-one difference entirely, which is a
        property of covariance estimation rather than of the generator.
        """
        from geoq.features.covariance import Covariances
        from geoq.geometry.riemannian import distance_airm

        data = small(
            n_subjects=1,
            n_trials_per_subject=100,
            n_times=256,
            drift=0.05,
            task_effect=0.0,
        )
        covariances = Covariances(
            estimator="oas", audit_conditioning=False
        ).fit_transform(data.epochs)

        adjacent = np.mean(
            [
                float(distance_airm(covariances[i], covariances[i + 1]))
                for i in range(90)
            ]
        )
        distant = np.mean(
            [
                float(distance_airm(covariances[i], covariances[i + 30]))
                for i in range(60)
            ]
        )
        assert adjacent < 0.8 * distant

    def test_drift_parameter_controls_autocorrelation(self) -> None:
        """The parameter moves the quantity it names, in the stated direction.

        Faster drift leaves less structure between neighbouring trials.
        """
        from geoq.features.covariance import Covariances
        from geoq.geometry.riemannian import distance_airm

        def adjacent_distance(drift: float) -> float:
            data = small(
                n_subjects=1,
                n_trials_per_subject=60,
                n_times=256,
                drift=drift,
                task_effect=0.0,
            )
            covariances = Covariances(
                estimator="oas", audit_conditioning=False
            ).fit_transform(data.epochs)
            return float(
                np.mean(
                    [
                        float(distance_airm(covariances[i], covariances[i + 1]))
                        for i in range(50)
                    ]
                )
            )

        assert adjacent_distance(0.02) < adjacent_distance(0.5)

    def test_conditioning_matches_the_real_regime(self) -> None:
        """Ingredient three: spatial correlation from volume conduction.

        White noise would give well-conditioned covariances and hide every
        conditioning problem real EEG has. Measured at the 22-channel default,
        the sample covariance reaches a median condition number above ``1e4``
        while a shrinkage estimator stays near ``1e3``.
        """
        from geoq.features.covariance import Covariances
        from geoq.geometry.spd import condition_number

        data = make_synthetic_eeg(n_subjects=2, n_trials_per_subject=20)
        options = {"audit_conditioning": False}
        plain = condition_number(
            Covariances(estimator="scm", **options).fit_transform(data.epochs)
        )
        shrunk = condition_number(
            Covariances(estimator="oas", **options).fit_transform(data.epochs)
        )
        assert np.median(plain) > 1e4
        assert np.median(plain) > 10 * np.median(shrunk)

    def test_task_effect_is_decodable_across_subjects(self) -> None:
        """The dataset must support a genuine generalisation result."""
        pytest.importorskip("sklearn")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline

        from geoq.evaluation.protocol import evaluate
        from geoq.evaluation.splitters import LeaveOneSubjectOut
        from geoq.features.covariance import Covariances
        from geoq.features.tangent_space import TangentSpace

        data = small(n_subjects=5, n_trials_per_subject=40, task_effect=0.5)
        pipeline = make_pipeline(
            Covariances(estimator="oas", audit_conditioning=False),
            TangentSpace(),
            LinearDiscriminantAnalysis(),
        )
        result = evaluate(
            pipeline,
            data.epochs,
            data.labels,
            groups=data.subjects,
            splitter=LeaveOneSubjectOut(),
        )
        assert result.mean("kappa") > 0.3

    def test_zero_task_effect_is_undecodable(self) -> None:
        """The true null that false-positive rates are measured on.

        If a residual effect leaked in, every calibration measurement in the
        statistics layer would be quantifying detection rather than error.
        """
        pytest.importorskip("sklearn")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline

        from geoq.evaluation.protocol import evaluate
        from geoq.evaluation.splitters import LeaveOneSubjectOut
        from geoq.features.covariance import Covariances
        from geoq.features.tangent_space import TangentSpace

        data = small(n_subjects=5, n_trials_per_subject=40, task_effect=0.0)
        pipeline = make_pipeline(
            Covariances(estimator="oas", audit_conditioning=False),
            TangentSpace(),
            LinearDiscriminantAnalysis(),
        )
        result = evaluate(
            pipeline,
            data.epochs,
            data.labels,
            groups=data.subjects,
            splitter=LeaveOneSubjectOut(),
        )
        assert abs(result.mean("kappa")) < 0.15

    def test_stronger_task_effect_is_easier(self) -> None:
        """Monotonicity: the parameter must control what it claims to."""
        pytest.importorskip("sklearn")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline

        from geoq.features.covariance import Covariances
        from geoq.features.tangent_space import TangentSpace

        pipeline = make_pipeline(
            Covariances(estimator="oas", audit_conditioning=False),
            TangentSpace(),
            LinearDiscriminantAnalysis(),
        )
        scores = []
        for effect in (0.1, 0.8):
            data = small(n_subjects=3, n_trials_per_subject=40, task_effect=effect)
            scores.append(
                cross_val_score(pipeline, data.epochs, data.labels, cv=3).mean()
            )
        assert scores[1] > scores[0]


# --------------------------------------------------------------------------- #
# 4. Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    """Datasets are selected by configuration name."""

    def test_synthetic_is_registered(self) -> None:
        assert "synthetic" in DATASETS

    def test_load_by_name(self) -> None:
        data = load_dataset("synthetic", n_subjects=2, n_trials_per_subject=10)
        assert isinstance(data, EEGDataset)
        assert data.n_subjects == 2

    def test_unknown_name_raises(self) -> None:
        """A typo must not quietly change which recordings an experiment used."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("bci_iv_2a_typo")

    def test_error_mentions_unimported_adapters(self) -> None:
        """The most likely cause of a missing name is a missing import."""
        with pytest.raises(ValueError) as excinfo:
            load_dataset("nonexistent")
        assert "imported" in str(excinfo.value)

    def test_duplicate_registration_rejected(self) -> None:
        """Otherwise a config would mean different things by import order."""
        with pytest.raises(ValueError, match="already registered"):

            @register_dataset("synthetic")
            def _duplicate() -> EEGDataset:  # pragma: no cover
                return small()

    def test_registration_and_cleanup(self) -> None:
        @register_dataset("test_only_dataset")
        def _loader(**kwargs: Any) -> EEGDataset:
            return small(**kwargs)

        try:
            assert load_dataset("test_only_dataset").n_subjects == 4
        finally:
            DATASETS.pop("test_only_dataset", None)
