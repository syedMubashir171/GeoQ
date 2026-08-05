"""Tests for :mod:`geoq.statistics.comparison`.

What is being defended
----------------------
This module produces the inferential claims in the thesis, so the tests check
statistical behaviour rather than only code paths.

* **The correction is real and correctly signed.** ``TestCorrection`` verifies
  the inflation factor against its closed form and confirms the corrected test
  is strictly more conservative than the naive one on identical data.
* **Calibration is measured, not assumed.** ``TestCalibration`` runs both tests
  under a simulated true null and reports their false-positive rates. It
  asserts the corrected test improves on the naive one under fold correlation
  and, deliberately, does *not* assert that it reaches the nominal rate --
  because it does not, and a test encoding that false expectation would have
  to be quietly relaxed later.
* **Equivalence is separated from non-significance.** ``TestEquivalence`` and
  ``TestVerdict`` pin the distinction the thesis's negative results depend on:
  failing to detect a difference supports nothing on its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from geoq.statistics.comparison import (
    ComparisonResult,
    compare,
    corrected_resampled_ttest,
    correction_factor,
    hedges_g,
    holm_bonferroni,
    minimum_detectable_effect,
    tost_equivalence,
)

N_FOLDS = 9
N_TRAIN = 8 * 288
N_TEST = 288


def naive_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Ordinary paired t-test, for comparison against the corrected one."""
    differences = a - b
    n = differences.size
    statistic = differences.mean() / np.sqrt(differences.var(ddof=1) / n)
    return float(statistic), float(2.0 * stats.t.sf(abs(statistic), n - 1))


@pytest.fixture
def paired_scores(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two score vectors with a small consistent difference."""
    base = 0.60 + rng.normal(0, 0.05, N_FOLDS)
    return base, base - 0.03 + rng.normal(0, 0.01, N_FOLDS)


# --------------------------------------------------------------------------- #
# 1. The correction itself
# --------------------------------------------------------------------------- #


class TestCorrection:
    """The Nadeau-Bengio inflation factor and its consequences."""

    def test_matches_the_closed_form(self) -> None:
        assert correction_factor(800, 100, 10) == pytest.approx(0.1 + 100 / 800)

    def test_loso_nine_subjects(self) -> None:
        """The configuration this thesis actually reports under.

        With nine subjects the factor is 0.236 against a naive 0.111, so the
        standard error is 1.46 times larger. That ratio is what moves
        borderline p-values across the conventional threshold.
        """
        factor = correction_factor(N_TRAIN, N_TEST, N_FOLDS)
        assert factor == pytest.approx(1 / 9 + 1 / 8)
        assert np.sqrt(factor / (1 / 9)) == pytest.approx(1.458, abs=0.01)

    def test_factor_grows_as_test_folds_grow(self) -> None:
        """Larger test folds mean more training overlap, hence more inflation."""
        small = correction_factor(900, 100, 10)
        large = correction_factor(500, 500, 10)
        assert large > small

    @pytest.mark.parametrize(
        ("n_train", "n_test", "n_folds"),
        [(0, 10, 5), (10, 0, 5), (-1, 10, 5), (100, 10, 1)],
    )
    def test_invalid_arguments_rejected(
        self, n_train: int, n_test: int, n_folds: int
    ) -> None:
        with pytest.raises(ValueError):
            correction_factor(n_train, n_test, n_folds)

    def test_corrected_test_is_strictly_more_conservative(
        self, paired_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """On identical data the corrected p-value must be larger.

        The direction is the point. A correction that made results *more*
        significant would be a sign error, and the resulting inference would
        be wrong in the dangerous direction.
        """
        a, b = paired_scores
        _, naive_p = naive_ttest(a, b)
        statistic, corrected_p, _ = corrected_resampled_ttest(
            a, b, n_train=N_TRAIN, n_test=N_TEST
        )
        assert corrected_p > naive_p
        assert abs(statistic) < abs(naive_ttest(a, b)[0])

    def test_standard_error_ratio_matches_the_factor(
        self, paired_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = paired_scores
        _, _, corrected_se = corrected_resampled_ttest(
            a, b, n_train=N_TRAIN, n_test=N_TEST
        )
        differences = a - b
        naive_se = np.sqrt(differences.var(ddof=1) / N_FOLDS)
        expected = np.sqrt(correction_factor(N_TRAIN, N_TEST, N_FOLDS) * N_FOLDS)
        assert corrected_se / naive_se == pytest.approx(expected)

    def test_identical_scores_give_p_one(self) -> None:
        """Zero variance and zero difference is no evidence, not certainty.

        Returning ``p = 0`` here -- which a naive implementation does, by
        dividing zero by zero -- would report perfect certainty of a
        difference that is exactly zero.
        """
        scores = np.linspace(0.5, 0.7, N_FOLDS)
        statistic, p_value, standard_error = corrected_resampled_ttest(
            scores, scores.copy(), n_train=N_TRAIN, n_test=N_TEST
        )
        assert statistic == 0.0
        assert p_value == 1.0
        assert standard_error == 0.0

    def test_constant_nonzero_difference(self) -> None:
        """A perfectly consistent difference is infinitely significant.

        Degenerate but correct: with zero variance across folds there is no
        sampling uncertainty left. It is flagged rather than smoothed over,
        because in practice it means the folds are not independent replicates
        at all.
        """
        a = np.linspace(0.5, 0.7, N_FOLDS)
        statistic, p_value, _ = corrected_resampled_ttest(
            a, a - 0.05, n_train=N_TRAIN, n_test=N_TEST
        )
        assert np.isposinf(statistic)
        assert p_value == 0.0

    def test_antisymmetry(self, paired_scores) -> None:
        """Swapping the models flips the sign but not the p-value."""
        a, b = paired_scores
        forward = corrected_resampled_ttest(a, b, n_train=N_TRAIN, n_test=N_TEST)
        backward = corrected_resampled_ttest(b, a, n_train=N_TRAIN, n_test=N_TEST)
        assert forward[0] == pytest.approx(-backward[0])
        assert forward[1] == pytest.approx(backward[1])


# --------------------------------------------------------------------------- #
# 2. Calibration, measured
# --------------------------------------------------------------------------- #


class TestCalibration:
    """False-positive rates under a simulated true null."""

    @staticmethod
    def _false_positive_rates(
        tau: float, *, trials: int = 1500, seed: int = 7
    ) -> tuple[float, float]:
        """Return naive and corrected false-positive rates.

        Per-fold differences share a component of standard deviation ``tau``,
        modelling the correlation induced by folds sharing training data. The
        true mean difference is zero, so every rejection is a false positive.

        Args:
            tau: Standard deviation of the shared component.
            trials: Number of simulated experiments.
            seed: Random seed.

        Returns:
            Tuple of naive and corrected false-positive rates.
        """
        rng = np.random.default_rng(seed)
        naive_hits = corrected_hits = 0
        for _ in range(trials):
            shared = rng.normal(0.0, tau) if tau > 0 else 0.0
            differences = shared + rng.normal(0.0, 0.02, N_FOLDS)
            a = 0.6 + differences / 2
            b = 0.6 - differences / 2
            naive_hits += naive_ttest(a, b)[1] < 0.05
            corrected_hits += (
                corrected_resampled_ttest(a, b, n_train=N_TRAIN, n_test=N_TEST)[1]
                < 0.05
            )
        return naive_hits / trials, corrected_hits / trials

    def test_naive_is_calibrated_when_folds_are_independent(self) -> None:
        """Establishes the baseline: the naive test is not broken in general."""
        naive, _ = self._false_positive_rates(tau=0.0)
        assert 0.03 < naive < 0.07

    def test_correction_is_conservative_when_folds_are_independent(self) -> None:
        """The cost of the correction, stated rather than hidden.

        With genuinely independent folds the corrected test rejects far less
        often than nominal, so it loses power. That is the price of protecting
        against the correlated case, and it is worth knowing when a null
        result is being interpreted.
        """
        naive, corrected = self._false_positive_rates(tau=0.0)
        assert corrected < naive
        assert corrected < 0.02

    @pytest.mark.parametrize("tau", [0.01, 0.02])
    def test_correction_reduces_false_positives_under_correlation(
        self, tau: float
    ) -> None:
        """The case the correction exists for."""
        naive, corrected = self._false_positive_rates(tau=tau)
        assert naive > 0.15
        assert corrected < naive * 0.75

    def test_correction_does_not_fully_restore_the_nominal_rate(self) -> None:
        """Asserted deliberately, because it is true.

        At ``tau = 0.02`` the corrected false-positive rate is around 0.31,
        far above the nominal 0.05. Writing a test that expected 0.05 would
        encode a false belief about the method and would have to be silently
        relaxed the first time it failed. The honest consequence is that a
        single p-value is weak evidence regardless of correction, which is why
        the module also provides effect sizes and equivalence tests.
        """
        _, corrected = self._false_positive_rates(tau=0.02)
        assert corrected > 0.05


# --------------------------------------------------------------------------- #
# 3. Effect size
# --------------------------------------------------------------------------- #


class TestEffectSize:
    """Hedges' g and its small-sample correction."""

    def test_shrinks_cohens_d(self, paired_scores) -> None:
        """At nine folds the correction is about ten percent, not negligible."""
        a, b = paired_scores
        differences = a - b
        cohen = differences.mean() / differences.std(ddof=1)
        g = hedges_g(a, b)
        assert abs(g) < abs(cohen)
        assert g / cohen == pytest.approx(1 - 3 / (4 * (N_FOLDS - 1) - 1))

    def test_correction_vanishes_with_many_folds(
        self, rng: np.random.Generator
    ) -> None:
        """The bias is a small-sample artefact and must disappear at scale."""
        a = rng.normal(0.6, 0.05, 500)
        b = a - 0.02 + rng.normal(0, 0.01, 500)
        differences = a - b
        cohen = differences.mean() / differences.std(ddof=1)
        assert hedges_g(a, b) == pytest.approx(cohen, rel=0.01)

    def test_sign_follows_the_direction(self, paired_scores) -> None:
        a, b = paired_scores
        assert hedges_g(a, b) > 0
        assert hedges_g(b, a) < 0

    def test_zero_when_differences_have_no_spread(self) -> None:
        a = np.linspace(0.5, 0.7, N_FOLDS)
        assert hedges_g(a, a - 0.05) == 0.0

    def test_is_scale_free(self, rng: np.random.Generator) -> None:
        """Multiplying both score vectors must not change the effect size."""
        a = rng.normal(0.6, 0.05, N_FOLDS)
        b = a - 0.02 + rng.normal(0, 0.01, N_FOLDS)
        assert hedges_g(a * 10, b * 10) == pytest.approx(hedges_g(a, b))


# --------------------------------------------------------------------------- #
# 4. Equivalence
# --------------------------------------------------------------------------- #


class TestEquivalence:
    """TOST, and why it is not the complement of a significance test."""

    def test_identical_models_are_equivalent(self, rng: np.random.Generator) -> None:
        base = 0.6 + rng.normal(0, 0.03, N_FOLDS)
        near = base + rng.normal(0, 0.002, N_FOLDS)
        p_value = tost_equivalence(
            base, near, bound=0.02, n_train=N_TRAIN, n_test=N_TEST
        )
        assert p_value < 0.05

    def test_clearly_different_models_are_not_equivalent(
        self, rng: np.random.Generator
    ) -> None:
        base = 0.6 + rng.normal(0, 0.01, N_FOLDS)
        worse = base - 0.15
        p_value = tost_equivalence(
            base, worse, bound=0.02, n_train=N_TRAIN, n_test=N_TEST
        )
        assert p_value > 0.05

    def test_a_wider_bound_makes_equivalence_easier(self, paired_scores) -> None:
        """Which is why the bound must be justified before seeing the data."""
        a, b = paired_scores
        tight = tost_equivalence(a, b, bound=0.01, n_train=N_TRAIN, n_test=N_TEST)
        loose = tost_equivalence(a, b, bound=0.20, n_train=N_TRAIN, n_test=N_TEST)
        assert loose < tight

    def test_noisy_data_cannot_establish_equivalence(
        self, rng: np.random.Generator
    ) -> None:
        """The property that makes TOST honest.

        With enough noise, neither a difference nor equivalence can be
        demonstrated. A method that returned equivalence here would let any
        underpowered study claim two models perform the same.
        """
        a = 0.6 + rng.normal(0, 0.25, N_FOLDS)
        b = 0.6 + rng.normal(0, 0.25, N_FOLDS)
        difference_p = corrected_resampled_ttest(a, b, n_train=N_TRAIN, n_test=N_TEST)[
            1
        ]
        equivalence_p = tost_equivalence(
            a, b, bound=0.02, n_train=N_TRAIN, n_test=N_TEST
        )
        assert difference_p > 0.05
        assert equivalence_p > 0.05

    def test_uses_the_corrected_standard_error(self, rng: np.random.Generator) -> None:
        """An uncorrected TOST would declare equivalence too readily.

        That is the more dangerous direction here: it turns an underpowered
        null into a positive claim of sameness.
        """
        base = 0.6 + rng.normal(0, 0.02, N_FOLDS)
        near = base + rng.normal(0, 0.004, N_FOLDS)
        corrected = tost_equivalence(
            base, near, bound=0.01, n_train=N_TRAIN, n_test=N_TEST
        )
        # A very large training set drives the correction toward the naive
        # case, which must give a smaller (more permissive) p-value.
        almost_naive = tost_equivalence(base, near, bound=0.01, n_train=10**7, n_test=1)
        assert corrected > almost_naive

    @pytest.mark.parametrize("bound", [0.0, -0.01])
    def test_non_positive_bound_rejected(self, bound: float, paired_scores) -> None:
        a, b = paired_scores
        with pytest.raises(ValueError, match="must be positive"):
            tost_equivalence(a, b, bound=bound, n_train=N_TRAIN, n_test=N_TEST)


class TestMinimumDetectableEffect:
    """The number that makes a null result interpretable."""

    def test_scales_with_noise(self, rng: np.random.Generator) -> None:
        quiet = rng.normal(0.6, 0.01, N_FOLDS)
        noisy = rng.normal(0.6, 0.20, N_FOLDS)
        small = minimum_detectable_effect(
            quiet, quiet - 0.02, n_train=N_TRAIN, n_test=N_TEST
        )
        large = minimum_detectable_effect(
            noisy,
            noisy * 0 + rng.normal(0.6, 0.20, N_FOLDS),
            n_train=N_TRAIN,
            n_test=N_TEST,
        )
        assert large > small

    def test_a_detected_difference_exceeds_the_mde(self, paired_scores) -> None:
        """Internal consistency: what was detected was detectable."""
        a, b = paired_scores
        _, p_value, _ = corrected_resampled_ttest(a, b, n_train=N_TRAIN, n_test=N_TEST)
        mde = minimum_detectable_effect(a, b, n_train=N_TRAIN, n_test=N_TEST)
        if p_value < 0.05:
            assert abs(np.mean(a - b)) > mde * 0.5

    def test_higher_power_requires_a_larger_effect(self, paired_scores) -> None:
        a, b = paired_scores
        at_80 = minimum_detectable_effect(
            a, b, n_train=N_TRAIN, n_test=N_TEST, power=0.8
        )
        at_95 = minimum_detectable_effect(
            a, b, n_train=N_TRAIN, n_test=N_TEST, power=0.95
        )
        assert at_95 > at_80


# --------------------------------------------------------------------------- #
# 5. Verdicts
# --------------------------------------------------------------------------- #


class TestVerdict:
    """The four outcomes, kept distinct."""

    @staticmethod
    def _result(**overrides) -> ComparisonResult:
        defaults = {
            "mean_difference": 0.0,
            "statistic": 0.0,
            "p_value": 0.5,
            "degrees_of_freedom": 8,
            "corrected_standard_error": 0.01,
            "naive_standard_error": 0.007,
            "confidence_interval": (-0.02, 0.02),
            "effect_size": 0.0,
            "n_folds": 9,
            "minimum_detectable_effect": 0.03,
        }
        return ComparisonResult(**{**defaults, **overrides})

    def test_non_significance_alone_is_inconclusive(self) -> None:
        """The error this whole module exists to prevent.

        A non-significant difference with no equivalence test supports
        nothing. It is equally consistent with the models being identical and
        with the study having been far too small to tell.
        """
        verdict = self._result(p_value=0.42).verdict()
        assert "Inconclusive" in verdict
        assert "0.03" in verdict  # the MDE must be quoted
        assert not self._result(p_value=0.42).equivalent

    def test_equivalence_requires_a_test(self) -> None:
        assert not self._result(p_value=0.9).equivalent
        assert self._result(
            p_value=0.9, equivalence_bound=0.02, equivalence_p_value=0.01
        ).equivalent

    def test_difference_detected(self) -> None:
        verdict = self._result(p_value=0.001, mean_difference=0.08).verdict()
        assert "Difference detected" in verdict

    def test_equivalence_established(self) -> None:
        verdict = self._result(
            p_value=0.8, equivalence_bound=0.02, equivalence_p_value=0.004
        ).verdict()
        assert "Practically equivalent" in verdict

    def test_contradiction_is_flagged(self) -> None:
        """Significant and equivalent at once means the bound is too loose."""
        verdict = self._result(
            p_value=0.001, equivalence_bound=0.5, equivalence_p_value=0.001
        ).verdict()
        assert "Contradictory" in verdict

    def test_to_dict_is_flat_and_complete(self) -> None:
        record = self._result().to_dict()
        for key in (
            "p_value",
            "se_corrected",
            "se_naive",
            "hedges_g",
            "minimum_detectable_effect",
            "significant",
            "equivalent",
        ):
            assert key in record


# --------------------------------------------------------------------------- #
# 6. Multiple comparisons
# --------------------------------------------------------------------------- #


class TestHolmBonferroni:
    """Step-down family-wise correction."""

    def test_known_example(self) -> None:
        adjusted, rejected = holm_bonferroni([0.001, 0.012, 0.04, 0.06, 0.20])
        assert adjusted[0] == pytest.approx(0.005)
        assert adjusted[1] == pytest.approx(0.048)
        assert rejected.tolist() == [True, True, False, False, False]

    def test_preserves_input_order(self) -> None:
        adjusted, _ = holm_bonferroni([0.20, 0.001, 0.04])
        assert adjusted[1] < adjusted[2] < adjusted[0]

    def test_is_monotone(self, rng: np.random.Generator) -> None:
        """An adjusted p-value must never decrease as the raw one increases."""
        raw = np.sort(rng.uniform(0, 1, 20))
        adjusted, _ = holm_bonferroni(raw)
        assert np.all(np.diff(adjusted) >= -1e-12)

    def test_never_less_powerful_than_bonferroni(
        self, rng: np.random.Generator
    ) -> None:
        """Holm dominates Bonferroni, so there is no reason to use the latter."""
        raw = rng.uniform(0, 0.2, 15)
        adjusted, _ = holm_bonferroni(raw)
        assert np.all(adjusted <= np.minimum(raw * raw.size, 1.0) + 1e-12)

    def test_single_test_is_unchanged(self) -> None:
        adjusted, rejected = holm_bonferroni([0.03])
        assert adjusted[0] == pytest.approx(0.03)
        assert rejected[0]

    def test_capped_at_one(self) -> None:
        adjusted, _ = holm_bonferroni([0.6, 0.7, 0.8])
        assert np.all(adjusted <= 1.0)

    @pytest.mark.parametrize("bad", [[], [-0.1], [1.5]])
    def test_invalid_input_rejected(self, bad: list) -> None:
        with pytest.raises(ValueError):
            holm_bonferroni(bad)

    def test_eighteen_comparisons_under_the_null(
        self, rng: np.random.Generator
    ) -> None:
        """The thesis's realistic family size.

        One quantum model against three classical baselines across six
        datasets is eighteen tests. Under a true null roughly one uncorrected
        p-value below 0.05 is expected by chance; the correction must
        suppress it, or that single result becomes a reported finding.
        """
        false_positives = 0
        for _ in range(400):
            raw = rng.uniform(0, 1, 18)
            _, rejected = holm_bonferroni(raw)
            false_positives += bool(rejected.any())
        assert false_positives / 400 < 0.08


# --------------------------------------------------------------------------- #
# 7. The end-to-end comparison
# --------------------------------------------------------------------------- #


class TestCompare:
    """Comparing two EvaluationResult objects.

    Skipped without the ``ml`` extra: :mod:`geoq.evaluation.protocol` needs
    scikit-learn, while the rest of this module needs only SciPy. Keeping the
    statistics layer testable on a bare install is what lets the inference
    code be developed and verified independently of the modelling stack.
    """

    pytestmark = pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("sklearn") is None,
        reason="requires the 'ml' extra: pip install -e '.[ml]'",
    )

    @staticmethod
    def _fake_result(scores: np.ndarray, protocol, metric: str = "kappa"):
        from geoq.evaluation.protocol import EvaluationResult, FoldResult

        folds = tuple(
            FoldResult(
                fold=index,
                scores={metric: float(value)},
                n_train=N_TRAIN,
                n_test=N_TEST,
                test_groups=(index,),
            )
            for index, value in enumerate(scores)
        )
        return EvaluationResult(
            folds=folds,
            protocol=protocol,
            estimator_repr="fake",
            metrics=(metric,),
            n_samples=N_TRAIN + N_TEST,
            n_classes=2,
            chance_accuracy=0.5,
        )

    @pytest.fixture
    def protocol(self):
        from geoq.evaluation.splitters import LeaveOneSubjectOut

        return LeaveOneSubjectOut().info

    def test_end_to_end(self, protocol, rng: np.random.Generator) -> None:
        base = 0.4 + rng.normal(0, 0.05, N_FOLDS)
        # Fold differences must vary, or the comparison lands on the
        # degenerate zero-variance path and there is no standard error to
        # compare.
        other = base - 0.06 + rng.normal(0, 0.01, N_FOLDS)
        result = compare(
            self._fake_result(base, protocol),
            self._fake_result(other, protocol),
            equivalence_bound=0.02,
        )
        assert result.n_folds == N_FOLDS
        assert result.mean_difference == pytest.approx(
            float(np.mean(base - other)), abs=1e-9
        )
        assert result.corrected_standard_error > result.naive_standard_error
        assert result.equivalence_p_value is not None
        assert np.isfinite(result.minimum_detectable_effect)

    def test_confidence_interval_brackets_the_mean(
        self, protocol, rng: np.random.Generator
    ) -> None:
        base = 0.4 + rng.normal(0, 0.05, N_FOLDS)
        result = compare(
            self._fake_result(base, protocol),
            self._fake_result(base - 0.03 + rng.normal(0, 0.01, N_FOLDS), protocol),
        )
        lower, upper = result.confidence_interval
        assert lower < result.mean_difference < upper

    def test_mismatched_protocols_refused(self, rng: np.random.Generator) -> None:
        """Comparing across protocols confounds the model with the evaluation."""
        from geoq.evaluation.splitters import (
            LeaveOneSubjectOut,
            WithinSubjectKFold,
        )

        base = 0.4 + rng.normal(0, 0.05, N_FOLDS)
        with pytest.raises(ValueError, match="different protocols"):
            compare(
                self._fake_result(base, LeaveOneSubjectOut().info),
                self._fake_result(base, WithinSubjectKFold().info),
            )

    def test_mismatched_fold_counts_refused(
        self, protocol, rng: np.random.Generator
    ) -> None:
        with pytest.raises(ValueError, match="different fold sizes"):
            compare(
                self._fake_result(rng.normal(0.5, 0.05, N_FOLDS), protocol),
                self._fake_result(rng.normal(0.5, 0.05, N_FOLDS - 2), protocol),
            )

    def test_missing_metric_raises(self, protocol, rng: np.random.Generator) -> None:
        base = rng.normal(0.5, 0.05, N_FOLDS)
        with pytest.raises(KeyError):
            compare(
                self._fake_result(base, protocol),
                self._fake_result(base, protocol),
                metric="accuracy",
            )
