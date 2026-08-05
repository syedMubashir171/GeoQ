"""Tests for :mod:`geoq.statistics.significance`.

What is being defended
----------------------
* **The permutation scheme is the test.** ``TestPermutationScheme`` shows that
  permuting globally instead of within subject turns a correct ``p = 0.96``
  into ``p = 0.099`` on data containing no task signal at all. The scheme is
  not a detail; it determines the answer.
* **p is never zero.** ``TestPValueEstimator`` pins the Phipson-Smyth
  estimator and the resolution floor, so a model that beats every permutation
  is reported as bounded rather than certain.
* **Chance is not 0.5.** ``TestChanceLevel`` checks the binomial upper tail
  against exact values. At twenty test trials a random classifier reaches
  0.700 five percent of the time, which is above many published per-subject
  BCI accuracies.
* **The bootstrap's limitation is stated.** Resampling folds treats them as
  independent when they are not, so the interval is too narrow. That is
  documented in the module and asserted here by comparing against the
  corrected standard error.

Permutation tests are slow -- ``n_permutations`` times the fold count, model
fits -- so these tests use small permutation counts and tiny datasets. They
verify behaviour, not statistical power.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from geoq.statistics.significance import (
    BootstrapResult,
    PermutationResult,
    bootstrap_ci,
    empirical_chance_level,
    permutation_test,
    permute_labels,
)

pytest_sklearn = pytest.importorskip(
    "sklearn", reason="requires the 'ml' extra: pip install -e '.[ml]'"
)


def make_data(
    seed: int,
    *,
    n_subjects: int = 5,
    n_per_subject: int = 24,
    n_channels: int = 4,
    task_effect: float = 0.0,
    subject_priors: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SPD trials with optional task signal and per-subject class priors.

    Args:
        seed: Random seed.
        n_subjects: Number of participants.
        n_per_subject: Trials per participant.
        n_channels: Channel count.
        task_effect: Interpolation toward the class direction. Zero gives a
            true null: labels carry no feature information.
        subject_priors: Per-subject probability of the positive class.

    Returns:
        Trials, labels, and subject identifiers.
    """
    from geoq.geometry.riemannian import geodesic
    from geoq.geometry.spd import random_spd

    rng = np.random.default_rng(seed)
    directions = [random_spd(n_channels, rng=rng) for _ in range(2)]
    priors = subject_priors or [0.5] * n_subjects

    matrices, targets, groups = [], [], []
    for subject in range(n_subjects):
        anchor = random_spd(n_channels, rng=rng)
        for _ in range(n_per_subject):
            label = int(rng.random() < priors[subject])
            base = geodesic(anchor, random_spd(n_channels, rng=rng), 0.3)
            matrices.append(
                base
                if task_effect == 0.0
                else geodesic(base, directions[label], task_effect)
            )
            targets.append(label)
            groups.append(subject)
    return np.stack(matrices), np.array(targets), np.array(groups)


# --------------------------------------------------------------------------- #
# 1. Chance level
# --------------------------------------------------------------------------- #


class TestChanceLevel:
    """The theoretical 0.5 is the null's mean, not its upper tail."""

    @pytest.mark.parametrize(
        ("n_samples", "expected"),
        [(20, 0.700), (40, 0.625), (80, 0.5875), (288, 0.5486)],
    )
    def test_known_values(self, n_samples: int, expected: float) -> None:
        """Checked against the exact binomial quantile.

        At twenty test trials -- an ordinary single-subject fold -- a random
        classifier reaches 0.700 five percent of the time. A per-subject
        accuracy of 0.65 in a results table is therefore not evidence of
        anything, and reporting it as above chance is a common error.
        """
        assert empirical_chance_level(n_samples, 2) == pytest.approx(expected, abs=1e-3)

    def test_matches_the_binomial_quantile(self) -> None:
        computed = empirical_chance_level(100, 2, alpha=0.05)
        assert computed == pytest.approx(stats.binom.ppf(0.95, 100, 0.5) / 100)

    def test_approaches_the_theoretical_level(self) -> None:
        """Only asymptotically: the excess shrinks like one over root n."""
        assert empirical_chance_level(20, 2) - 0.5 > 0.15
        assert empirical_chance_level(10000, 2) - 0.5 < 0.01

    def test_decreases_monotonically_with_sample_size(self) -> None:
        levels = [empirical_chance_level(n, 2) for n in (20, 40, 80, 160, 320)]
        assert levels == sorted(levels, reverse=True)

    def test_four_class_is_lower(self) -> None:
        assert empirical_chance_level(100, 4) < empirical_chance_level(100, 2)
        assert empirical_chance_level(100, 4) > 0.25

    def test_stricter_alpha_raises_the_threshold(self) -> None:
        assert empirical_chance_level(100, 2, alpha=0.01) > empirical_chance_level(
            100, 2, alpha=0.05
        )

    @pytest.mark.parametrize(
        ("n_samples", "n_classes", "alpha"),
        [(0, 2, 0.05), (100, 1, 0.05), (100, 2, 0.0), (100, 2, 1.0)],
    )
    def test_invalid_arguments_rejected(
        self, n_samples: int, n_classes: int, alpha: float
    ) -> None:
        with pytest.raises(ValueError):
            empirical_chance_level(n_samples, n_classes, alpha=alpha)


# --------------------------------------------------------------------------- #
# 2. The permutation scheme
# --------------------------------------------------------------------------- #


class TestPermuteLabels:
    """Shuffling preserves what it must and destroys what it must."""

    def test_within_group_preserves_per_group_balance(
        self, rng: np.random.Generator
    ) -> None:
        labels = np.tile([0, 1], 30)
        groups = np.repeat(np.arange(6), 10)
        permuted = permute_labels(labels, rng, groups=groups, within_groups=True)
        for group in np.unique(groups):
            assert permuted[groups == group].sum() == labels[groups == group].sum()

    def test_global_destroys_per_group_balance(self, rng: np.random.Generator) -> None:
        """The negative control for the test above."""
        labels = np.repeat([0, 1], 30)
        groups = np.repeat(np.arange(6), 10)
        permuted = permute_labels(labels, rng, groups=groups, within_groups=False)
        original = [labels[groups == g].mean() for g in range(6)]
        shuffled = [permuted[groups == g].mean() for g in range(6)]
        assert original != shuffled

    def test_overall_balance_always_preserved(self, rng: np.random.Generator) -> None:
        labels = np.tile([0, 1, 1], 20)
        groups = np.repeat(np.arange(6), 10)
        for within in (True, False):
            permuted = permute_labels(labels, rng, groups=groups, within_groups=within)
            assert np.array_equal(np.sort(permuted), np.sort(labels))

    def test_within_groups_without_groups_rejected(
        self, rng: np.random.Generator
    ) -> None:
        """The default must not silently degrade to a global shuffle."""
        with pytest.raises(ValueError, match="requires `groups`"):
            permute_labels(np.tile([0, 1], 10), rng, within_groups=True)

    def test_shape_mismatch_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            permute_labels(np.zeros(10), rng, groups=np.zeros(7), within_groups=True)

    def test_reproducible(self) -> None:
        labels = np.tile([0, 1], 20)
        groups = np.repeat(np.arange(4), 10)
        first = permute_labels(labels, np.random.default_rng(5), groups=groups)
        second = permute_labels(labels, np.random.default_rng(5), groups=groups)
        assert np.array_equal(first, second)


class TestPermutationScheme:
    """The scheme determines the answer, on a true null."""

    def test_global_permutation_manufactures_significance(self) -> None:
        """The central result of this module.

        The data contains no task signal. Each subject has a different class
        prior, and the protocol is within-subject, so that prior is learnable
        from a subject's own training trials -- which is why the observed
        score sits above 0.5 despite the null.

        Permuting within subject preserves those priors, so the null model
        enjoys the same advantage and the comparison is fair: null mean 0.646,
        observed 0.583, p = 0.96. Permuting globally destroys them, dropping
        the null to 0.497, and the same observed score is suddenly compared
        against a baseline no model in the experiment ever faced: p = 0.099.

        A slightly stronger prior would push that below 0.05, producing a
        publishable false positive from pure noise. This is why the scheme is
        recorded in the result rather than left implicit.
        """
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(
            3, subject_priors=[0.1, 0.2, 0.5, 0.8, 0.9]
        )
        arguments = {
            "splitter": WithinSubjectKFold(n_splits=3),
            "groups": groups,
            "metric": "accuracy",
            "n_permutations": 150,
            "random_state": 0,
        }
        within = permutation_test(
            MDM(), matrices, targets, within_groups=True, **arguments
        )
        globally = permutation_test(
            MDM(), matrices, targets, within_groups=False, **arguments
        )

        assert within.observed == pytest.approx(globally.observed)
        assert within.null_mean > globally.null_mean + 0.10
        assert within.p_value > 0.5
        assert globally.p_value < within.p_value / 2

    def test_scheme_is_recorded(self) -> None:
        """Without it the p-value cannot be interpreted."""
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(1)
        result = permutation_test(
            MDM(),
            matrices,
            targets,
            splitter=WithinSubjectKFold(n_splits=3),
            groups=groups,
            n_permutations=20,
        )
        assert result.within_groups is True
        assert result.to_dict()["within_groups"] is True

    def test_within_groups_default_requires_groups(self) -> None:
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, _ = make_data(1)
        with pytest.raises(ValueError, match="requires `groups`"):
            permutation_test(
                MDM(),
                matrices,
                targets,
                splitter=WithinSubjectKFold(n_splits=3),
                n_permutations=5,
            )


# --------------------------------------------------------------------------- #
# 3. The p-value estimator
# --------------------------------------------------------------------------- #


class TestPValueEstimator:
    """Phipson-Smyth: p is bounded below by the permutation count."""

    @staticmethod
    def _result(observed: float, null: np.ndarray) -> PermutationResult:
        at_least = int(np.sum(null >= observed))
        return PermutationResult(
            observed=observed,
            null_scores=null,
            p_value=(1 + at_least) / (1 + null.size),
            n_permutations=null.size,
            metric="accuracy",
            within_groups=True,
            null_mean=float(null.mean()),
            null_std=float(null.std(ddof=1)),
        )

    def test_never_zero(self) -> None:
        """A model beating every permutation is bounded, not certain.

        Reporting ``p = 0`` claims infinite evidence from a finite sample.
        """
        null = np.full(999, 0.5)
        result = self._result(0.9, null)
        assert result.p_value > 0
        assert result.p_value == pytest.approx(1 / 1000)

    def test_resolution_floor_is_reported(self) -> None:
        result = self._result(0.9, np.full(99, 0.5))
        assert result.minimum_achievable_p == pytest.approx(1 / 100)
        assert result.at_resolution_limit

    def test_not_at_limit_when_some_permutations_win(self) -> None:
        null = np.concatenate([np.full(90, 0.5), np.full(10, 0.95)])
        result = self._result(0.9, null)
        assert not result.at_resolution_limit
        assert result.p_value == pytest.approx(11 / 101)

    def test_summary_flags_the_limit(self) -> None:
        assert "resolution limit" in self._result(0.9, np.full(99, 0.5)).summary()
        null = np.concatenate([np.full(50, 0.5), np.full(50, 0.95)])
        assert "resolution limit" not in self._result(0.9, null).summary()

    def test_p_is_one_when_the_model_is_worst(self) -> None:
        result = self._result(0.1, np.full(99, 0.5))
        assert result.p_value == pytest.approx(1.0)


class TestPermutationBehaviour:
    """End-to-end behaviour on signal and on noise."""

    def test_detects_a_strong_effect(self) -> None:
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(0, task_effect=0.6)
        result = permutation_test(
            MDM(),
            matrices,
            targets,
            splitter=WithinSubjectKFold(n_splits=3),
            groups=groups,
            n_permutations=99,
            random_state=0,
        )
        assert result.observed > 0.75
        assert result.p_value < 0.05
        assert result.null_mean == pytest.approx(0.5, abs=0.1)

    def test_p_value_from_the_real_code_path_is_never_zero(self) -> None:
        """Exercises the estimator inside ``permutation_test``, not a stand-in.

        Added after a mutation test: removing the Phipson-Smyth ``+1`` was
        caught by none of the tests above, because they all constructed a
        ``PermutationResult`` by hand and never ran the arithmetic in the
        module. A strong effect makes the observed score beat every
        permutation, so the p-value lands exactly on the floor -- which is the
        only case where the missing ``+1`` shows.
        """
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(0, task_effect=0.9)
        result = permutation_test(
            MDM(),
            matrices,
            targets,
            splitter=WithinSubjectKFold(n_splits=3),
            groups=groups,
            n_permutations=49,
            random_state=0,
        )
        assert float(np.sum(result.null_scores >= result.observed)) == 0.0
        assert result.p_value > 0.0
        assert result.p_value == pytest.approx(1 / 50)
        assert result.at_resolution_limit

    def test_does_not_detect_pure_noise(self) -> None:
        """A true null must not be rejected."""
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(11, task_effect=0.0)
        result = permutation_test(
            MDM(),
            matrices,
            targets,
            splitter=WithinSubjectKFold(n_splits=3),
            groups=groups,
            n_permutations=99,
            random_state=0,
        )
        assert result.p_value > 0.05

    def test_null_distribution_is_returned(self) -> None:
        """Needed to plot the null, which is how a reader checks the test."""
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(1)
        result = permutation_test(
            MDM(),
            matrices,
            targets,
            splitter=WithinSubjectKFold(n_splits=3),
            groups=groups,
            n_permutations=25,
        )
        assert result.null_scores.shape == (25,)
        assert result.null_std > 0

    def test_reproducible(self) -> None:
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(1)
        arguments = {
            "splitter": WithinSubjectKFold(n_splits=3),
            "groups": groups,
            "n_permutations": 20,
            "random_state": 42,
        }
        first = permutation_test(MDM(), matrices, targets, **arguments)
        second = permutation_test(MDM(), matrices, targets, **arguments)
        assert np.array_equal(first.null_scores, second.null_scores)

    def test_zero_permutations_rejected(self) -> None:
        from geoq.evaluation.splitters import WithinSubjectKFold
        from geoq.models.classical.mdm import MDM

        matrices, targets, groups = make_data(1)
        with pytest.raises(ValueError, match="n_permutations must be positive"):
            permutation_test(
                MDM(),
                matrices,
                targets,
                splitter=WithinSubjectKFold(n_splits=3),
                groups=groups,
                n_permutations=0,
            )


# --------------------------------------------------------------------------- #
# 4. Bootstrap
# --------------------------------------------------------------------------- #


class TestBootstrap:
    """Interval estimation without a normality assumption."""

    @pytest.fixture
    def scores(self) -> np.ndarray:
        return np.array([0.62, 0.71, 0.55, 0.68, 0.59, 0.74, 0.66, 0.61, 0.70])

    @pytest.mark.parametrize("method", ["percentile", "bca"])
    def test_interval_brackets_the_estimate(
        self, method: str, scores: np.ndarray
    ) -> None:
        result = bootstrap_ci(scores, method=method, n_resamples=2000)
        assert isinstance(result, BootstrapResult)
        assert result.lower < result.estimate < result.upper
        assert result.contains(result.estimate)

    def test_estimate_is_the_sample_statistic(self, scores: np.ndarray) -> None:
        result = bootstrap_ci(scores, n_resamples=500)
        assert result.estimate == pytest.approx(scores.mean())

    def test_wider_confidence_gives_a_wider_interval(self, scores: np.ndarray) -> None:
        narrow = bootstrap_ci(scores, confidence_level=0.80, n_resamples=3000)
        wide = bootstrap_ci(scores, confidence_level=0.99, n_resamples=3000)
        assert wide.width > narrow.width

    def test_noisier_data_gives_a_wider_interval(
        self, rng: np.random.Generator
    ) -> None:
        quiet = bootstrap_ci(rng.normal(0.6, 0.01, 20), n_resamples=2000)
        noisy = bootstrap_ci(rng.normal(0.6, 0.15, 20), n_resamples=2000)
        assert noisy.width > quiet.width

    def test_approximate_coverage(self, rng: np.random.Generator) -> None:
        """Roughly nominal on well-behaved independent data.

        Loose bounds on purpose: exact coverage is not the claim, and a tight
        assertion would be measuring the seed.
        """
        covered = 0
        for trial in range(200):
            sample = np.random.default_rng(trial).normal(0.6, 0.05, 25)
            covered += bootstrap_ci(
                sample, n_resamples=400, random_state=trial
            ).contains(0.6)
        assert 0.85 < covered / 200 < 1.0

    def test_narrower_than_the_corrected_standard_error(
        self, scores: np.ndarray
    ) -> None:
        """The documented limitation, asserted rather than only described.

        Resampling folds treats them as independent when they share training
        data, so the interval is too narrow -- in the same direction and for
        the same reason as an uncorrected t-test. It describes the spread of
        fold scores; it is not an inference tool.
        """
        from geoq.statistics.comparison import correction_factor

        result = bootstrap_ci(scores, n_resamples=3000)
        naive_error = scores.std(ddof=1) / np.sqrt(scores.size)
        factor = correction_factor(8 * 288, 288, scores.size)
        corrected_error = np.sqrt(factor * scores.var(ddof=1))
        assert result.standard_error == pytest.approx(naive_error, rel=0.25)
        assert result.width < 2 * 1.96 * corrected_error

    def test_bca_reports_bias(self, rng: np.random.Generator) -> None:
        result = bootstrap_ci(rng.normal(0.6, 0.05, 30), n_resamples=2000)
        assert abs(result.bias) < 0.02
        assert result.standard_error > 0

    def test_custom_statistic(self, scores: np.ndarray) -> None:
        result = bootstrap_ci(
            scores, statistic=np.median, method="percentile", n_resamples=2000
        )
        assert result.estimate == pytest.approx(np.median(scores))

    def test_reproducible(self, scores: np.ndarray) -> None:
        first = bootstrap_ci(scores, n_resamples=500, random_state=3)
        second = bootstrap_ci(scores, n_resamples=500, random_state=3)
        assert first.lower == second.lower
        assert first.upper == second.upper

    def test_bca_handles_a_degenerate_sample(self) -> None:
        """Constant data makes the BCa bias correction undefined.

        Falling back to percentile limits is the honest response; an infinite
        normal quantile would otherwise propagate NaN into the interval.
        """
        result = bootstrap_ci(np.full(10, 0.6), method="bca", n_resamples=500)
        assert np.isfinite(result.lower)
        assert np.isfinite(result.upper)
        assert result.width == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("values", "kwargs", "pattern"),
        [
            (np.array([0.5]), {}, "at least two"),
            (np.array([[0.5, 0.6]]), {}, "one-dimensional"),
            (np.array([0.5, np.nan]), {}, "non-finite"),
            (np.array([0.5, 0.6]), {"confidence_level": 1.0}, "strictly in"),
            (np.array([0.5, 0.6]), {"method": "normal"}, "percentile.*bca"),
            (np.array([0.5, 0.6]), {"method": "bca"}, "at least three"),
        ],
    )
    def test_invalid_input_rejected(
        self, values: np.ndarray, kwargs: dict, pattern: str
    ) -> None:
        with pytest.raises(ValueError, match=pattern):
            bootstrap_ci(values, **kwargs)

    def test_to_dict_excludes_resamples(self, scores: np.ndarray) -> None:
        """A results file must not carry ten thousand intermediate numbers."""
        record = bootstrap_ci(scores, n_resamples=500).to_dict()
        assert "resamples" not in record
        assert set(record) >= {"estimate", "ci_lower", "ci_upper", "method"}
