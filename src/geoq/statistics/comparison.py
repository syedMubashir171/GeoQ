"""Statistical comparison of cross-validated results.

Turning two :class:`~geoq.evaluation.protocol.EvaluationResult` objects into a
defensible claim. This is the layer the thesis's negative results depend on
most: "the quantum model is not better" is only publishable if the statistics
distinguish *no detected difference* from *evidence of no difference*, and
that distinction is the whole reason this module exists.

Why an ordinary t-test over folds is wrong
------------------------------------------
Cross-validation folds share training data. In leave-one-subject-out with nine
subjects, any two folds have seven of nine subjects in common, so their scores
are strongly correlated. The usual paired t-test assumes independence, and
under this violation its variance estimate is too small: it declares
differences significant that would not survive a fresh sample. Nadeau and
Bengio (2003) derived the correction, which inflates the variance by
``1/k + n_test/n_train``. Every test in this module uses it, including the
equivalence test -- a TOST built on an uncorrected standard error would be
anticonservative in exactly the same way, and would let a "we proved
equivalence" claim rest on the same error.

What the correction does and does not fix
-----------------------------------------
It is a partial remedy. Simulated under a true null with nine folds, varying
the standard deviation ``tau`` of the component shared across per-fold
differences:

======  ==================  ======================
``tau``  naive error rate    corrected error rate
======  ==================  ======================
0.00    0.050               0.008
0.01    0.242               0.103
0.02    0.468               0.306
0.04    0.712               0.598
======  ==================  ======================

Two things follow. When folds are genuinely independent the correction is
conservative, costing power. When they overlap -- the actual situation in
cross-validation -- it removes more than half the excess false positives but
does not restore the nominal five percent.

So a single p-value from this test, however corrected, is weak evidence. A
claim in this thesis should rest on an effect size, an equivalence test where
the claim is sameness, and replication across datasets, with the p-value as
one input among several rather than the verdict.

Three claims, three tools
-------------------------
* **"A differs from B."** :func:`corrected_resampled_ttest`. Rejecting the null
  supports a difference.
* **"A and B are practically the same."** :func:`tost_equivalence`. Failing to
  reject a difference does *not* support this; only an equivalence test does,
  and it requires stating in advance how large a difference would matter.
* **"We could have detected a difference of at least X."**
  :func:`minimum_detectable_effect`. The honest companion to any null result:
  without it, "no significant difference" is indistinguishable from "we had no
  power to find one".

References
----------
Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
    *Machine Learning*, 52(3), 239-281.
Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of
    significance tests for comparing learning algorithms. *PAKDD*.
Lakens, D. (2017). Equivalence tests: a practical primer for t tests,
    correlations, and meta-analyses. *SPPS*, 8(4), 355-362.
Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
    size. *Journal of Educational Statistics*, 6(2), 107-128.
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*, 6(2), 65-70.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "ComparisonResult",
    "compare",
    "corrected_resampled_ttest",
    "correction_factor",
    "hedges_g",
    "holm_bonferroni",
    "minimum_detectable_effect",
    "tost_equivalence",
]

logger = logging.getLogger(__name__)


def correction_factor(n_train: int, n_test: int, n_folds: int) -> float:
    r"""Nadeau-Bengio variance inflation factor.

    .. math::
        \rho = \frac{1}{k} + \frac{n_\text{test}}{n_\text{train}}

    Replaces the ``1/k`` of an ordinary paired t-test. The extra term accounts
    for the overlap between training sets: the more training data two folds
    share, the more correlated their scores, and the larger the true variance
    of the mean difference.

    The practical size of this matters. Under leave-one-subject-out with nine
    subjects, ``k = 9`` and ``n_test/n_train = 1/8``, giving ``rho = 0.236``
    against ``1/k = 0.111`` -- the corrected standard error is about 1.46 times
    larger, which moves many borderline p-values across 0.05.

    Args:
        n_train: Training trials per fold.
        n_test: Test trials per fold.
        n_folds: Number of folds.

    Returns:
        The inflation factor.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"n_train and n_test must be positive, got {n_train} and {n_test}."
        )
    return 1.0 / n_folds + n_test / n_train


def _paired_differences(
    scores_a: ArrayLike, scores_b: ArrayLike
) -> NDArray[np.float64]:
    """Validate two score vectors and return their paired differences.

    Args:
        scores_a: Per-fold scores of the first model.
        scores_b: Per-fold scores of the second model.

    Returns:
        ``scores_a - scores_b``.

    Raises:
        ValueError: If the vectors differ in length, are shorter than two, or
            contain non-finite values.
    """
    first = np.asarray(scores_a, dtype=np.float64)
    second = np.asarray(scores_b, dtype=np.float64)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError(
            f"Scores must be one-dimensional per-fold vectors, got shapes "
            f"{first.shape} and {second.shape}."
        )
    if first.shape != second.shape:
        raise ValueError(
            f"Paired comparison needs one score per fold from each model, got "
            f"{first.shape[0]} and {second.shape[0]}. The two models must be "
            f"evaluated on identical folds; scores from different splits are "
            f"not paired and cannot be compared this way."
        )
    if first.shape[0] < 2:
        raise ValueError("At least two folds are required.")
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("Scores contain non-finite values.")

    return first - second


def _has_negligible_spread(differences: NDArray[np.float64]) -> bool:
    """Return whether the fold differences are constant to within round-off.

    Tested relative to the magnitude of the differences rather than against
    exact zero. Subtracting two score vectors that differ by a constant leaves
    a spread near ``1e-17`` rather than exactly zero, so an equality test
    would miss the degenerate case and produce a t statistic of ``4e14`` --
    reported as overwhelming significance when the real situation is that the
    folds carry no independent information at all.

    Args:
        differences: Per-fold differences.

    Returns:
        True when the spread is negligible.
    """
    scale = max(float(np.max(np.abs(differences))), 1.0)
    return bool(np.std(differences, ddof=1) <= 1e-12 * scale)


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing two models across matched folds.

    Attributes:
        mean_difference: Mean of ``scores_a - scores_b``.
        statistic: The corrected t statistic.
        p_value: Two-sided p-value from the corrected test.
        degrees_of_freedom: ``n_folds - 1``.
        corrected_standard_error: Standard error after the Nadeau-Bengio
            inflation.
        naive_standard_error: Standard error without the correction, reported
            so the size of the correction is visible rather than implicit.
        confidence_interval: Two-sided interval for the mean difference at the
            requested level, using the corrected standard error.
        effect_size: Hedges' g on the paired differences.
        n_folds: Number of folds.
        alpha: Significance level used.
        equivalence_bound: The bound passed to the equivalence test, if any.
        equivalence_p_value: TOST p-value, if an equivalence bound was given.
        minimum_detectable_effect: Smallest true difference this design could
            have detected at the requested level and power.

    """

    mean_difference: float
    statistic: float
    p_value: float
    degrees_of_freedom: int
    corrected_standard_error: float
    naive_standard_error: float
    confidence_interval: tuple[float, float]
    effect_size: float
    n_folds: int
    alpha: float = 0.05
    equivalence_bound: float | None = None
    equivalence_p_value: float | None = None
    minimum_detectable_effect: float = float("nan")

    @property
    def significant(self) -> bool:
        """Whether a difference was detected at ``alpha``."""
        return self.p_value < self.alpha

    @property
    def equivalent(self) -> bool:
        """Whether practical equivalence was demonstrated at ``alpha``.

        False when no equivalence bound was supplied: equivalence is a claim
        that must be tested, never inferred from a non-significant difference.
        """
        return (
            self.equivalence_p_value is not None
            and self.equivalence_p_value < self.alpha
        )

    def verdict(self) -> str:
        """Return the conclusion the statistics actually support.

        The four possible outcomes are distinct, and conflating the last two
        is the most common error in a negative result. A non-significant
        difference with no equivalence test supports nothing at all; it is
        consistent both with the models being identical and with the study
        being too small to tell.
        """
        if self.significant and self.equivalent:
            return (
                "Contradictory: a statistically detectable difference that is "
                "also within the equivalence bound. The bound is wider than "
                "the effect, so it was probably set too loosely to be "
                "meaningful."
            )
        if self.significant:
            return (
                f"Difference detected (p={self.p_value:.4f}, "
                f"mean={self.mean_difference:+.4f}, g={self.effect_size:+.2f})."
            )
        if self.equivalent:
            return (
                f"Practically equivalent within +-{self.equivalence_bound:.4f} "
                f"(TOST p={self.equivalence_p_value:.4f})."
            )
        return (
            f"Inconclusive: no difference detected (p={self.p_value:.4f}) and "
            f"equivalence not established. This design could only have "
            f"detected a difference of {self.minimum_detectable_effect:.4f} or "
            f"larger, so a smaller real difference would have been missed. Do "
            f"not report this as evidence that the models perform the same."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary for a results file."""
        return {
            "mean_difference": self.mean_difference,
            "t_statistic": self.statistic,
            "p_value": self.p_value,
            "df": self.degrees_of_freedom,
            "se_corrected": self.corrected_standard_error,
            "se_naive": self.naive_standard_error,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "hedges_g": self.effect_size,
            "n_folds": self.n_folds,
            "alpha": self.alpha,
            "significant": self.significant,
            "equivalence_bound": self.equivalence_bound,
            "equivalence_p_value": self.equivalence_p_value,
            "equivalent": self.equivalent,
            "minimum_detectable_effect": self.minimum_detectable_effect,
        }


def corrected_resampled_ttest(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    *,
    n_train: int,
    n_test: int,
) -> tuple[float, float, float]:
    """Nadeau-Bengio corrected resampled paired t-test.

    Args:
        scores_a: Per-fold scores of the first model.
        scores_b: Per-fold scores of the second model, on the same folds.
        n_train: Training trials per fold.
        n_test: Test trials per fold.

    Returns:
        Tuple ``(statistic, p_value, corrected_standard_error)``.

    Raises:
        ValueError: If the score vectors are incompatible.
    """
    differences = _paired_differences(scores_a, scores_b)
    n_folds = differences.shape[0]
    variance = float(np.var(differences, ddof=1))

    if _has_negligible_spread(differences):
        # Identical fold-by-fold scores. The t statistic is undefined rather
        # than infinite: there is no evidence of a difference, and reporting
        # p = 0 for a zero difference would be actively misleading.
        mean = float(np.mean(differences))
        logger.info(
            "corrected_resampled_ttest: the two models scored identically on "
            "every fold; returning p=1 rather than an undefined statistic."
        )
        if abs(mean) <= 1e-12:
            return 0.0, 1.0, 0.0
        return float(np.sign(mean)) * np.inf, 0.0, 0.0

    rho = correction_factor(n_train, n_test, n_folds)
    standard_error = float(np.sqrt(rho * variance))
    statistic = float(np.mean(differences)) / standard_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n_folds - 1))
    return statistic, p_value, standard_error


def hedges_g(scores_a: ArrayLike, scores_b: ArrayLike) -> float:
    """Hedges' g for paired differences, with the small-sample correction.

    Cohen's d is biased upward when the sample is small, and cross-validation
    typically gives fewer than ten folds. The correction factor
    ``J = 1 - 3 / (4 * df - 1)`` removes most of that bias; at nine folds it
    shrinks the estimate by about nine percent, which is not negligible when
    the number is reported as an effect size in a paper.

    Args:
        scores_a: Per-fold scores of the first model.
        scores_b: Per-fold scores of the second model.

    Returns:
        Hedges' g. Zero when the differences have no spread.
    """
    differences = _paired_differences(scores_a, scores_b)
    if _has_negligible_spread(differences):
        return 0.0
    spread = float(np.std(differences, ddof=1))
    degrees_of_freedom = differences.shape[0] - 1
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
    return correction * float(np.mean(differences)) / spread


def tost_equivalence(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    *,
    bound: float,
    n_train: int,
    n_test: int,
) -> float:
    """Two one-sided tests for practical equivalence.

    Tests the null that the true difference lies *outside* ``[-bound, +bound]``.
    Rejecting it supports equivalence, which is the claim a negative result
    needs and which a non-significant difference cannot provide.

    The bound must be chosen before seeing the data and justified on
    substantive grounds -- for a BCI, the smallest accuracy difference that
    would change a clinical or engineering decision. Choosing it afterwards to
    make a result come out equivalent is the equivalence-testing analogue of
    p-hacking.

    Uses the same corrected standard error as
    :func:`corrected_resampled_ttest`. An uncorrected TOST would understate the
    variance and declare equivalence too readily, which is the more dangerous
    direction of error here.

    Args:
        scores_a: Per-fold scores of the first model.
        scores_b: Per-fold scores of the second model.
        bound: Equivalence margin, in the metric's own units and strictly
            positive.
        n_train: Training trials per fold.
        n_test: Test trials per fold.

    Returns:
        The TOST p-value: the larger of the two one-sided p-values.

    Raises:
        ValueError: If ``bound`` is not positive.
    """
    if bound <= 0:
        raise ValueError(
            f"The equivalence bound must be positive and expressed in the "
            f"metric's units, got {bound!r}."
        )

    differences = _paired_differences(scores_a, scores_b)
    n_folds = differences.shape[0]
    variance = float(np.var(differences, ddof=1))
    mean = float(np.mean(differences))

    if _has_negligible_spread(differences):
        return 0.0 if abs(mean) < bound else 1.0

    rho = correction_factor(n_train, n_test, n_folds)
    standard_error = float(np.sqrt(rho * variance))
    degrees_of_freedom = n_folds - 1

    lower = float(stats.t.sf((mean + bound) / standard_error, df=degrees_of_freedom))
    upper = float(stats.t.cdf((mean - bound) / standard_error, df=degrees_of_freedom))
    return max(lower, upper)


def minimum_detectable_effect(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    *,
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true difference this design could have detected.

    The number that makes a null result interpretable. Without it, "no
    significant difference" is compatible with the models being identical and
    with the study having been far too small to tell them apart, and a reader
    cannot distinguish the two.

    Computed from the observed variance, so it is a retrospective statement
    about this design's sensitivity, not a prospective power calculation.

    Args:
        scores_a: Per-fold scores of the first model.
        scores_b: Per-fold scores of the second model.
        n_train: Training trials per fold.
        n_test: Test trials per fold.
        alpha: Two-sided significance level.
        power: Desired power.

    Returns:
        The smallest detectable mean difference, in the metric's units.
    """
    differences = _paired_differences(scores_a, scores_b)
    n_folds = differences.shape[0]
    variance = float(np.var(differences, ddof=1))
    if _has_negligible_spread(differences):
        return 0.0

    rho = correction_factor(n_train, n_test, n_folds)
    standard_error = float(np.sqrt(rho * variance))
    degrees_of_freedom = n_folds - 1
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=degrees_of_freedom))
    beta = float(stats.t.ppf(power, df=degrees_of_freedom))
    return (critical + beta) * standard_error


def compare(
    result_a: Any,
    result_b: Any,
    *,
    metric: str = "kappa",
    alpha: float = 0.05,
    equivalence_bound: float | None = None,
    power: float = 0.8,
    require_matching_protocol: bool = True,
) -> ComparisonResult:
    """Compare two evaluation results on matched folds.

    Args:
        result_a: First :class:`~geoq.evaluation.protocol.EvaluationResult`.
        result_b: Second result, from the same protocol and folds.
        metric: Metric to compare. Defaults to kappa, which is comparable
            across datasets with different class balance in a way accuracy is
            not.
        alpha: Significance level.
        equivalence_bound: Margin for the equivalence test, in the metric's
            units. Supply it whenever the interesting claim is that two models
            perform the same.
        power: Power used for the minimum detectable effect.
        require_matching_protocol: Whether to refuse comparison across
            different protocols.

    Returns:
        The comparison.

    Raises:
        ValueError: If the two results are not comparable.
    """
    if require_matching_protocol and result_a.protocol != result_b.protocol:
        raise ValueError(
            f"Refusing to compare results from different protocols "
            f"({result_a.protocol.name!r} and {result_b.protocol.name!r}). A "
            f"difference between them would confound the models with the "
            f"evaluation, and the paired test assumes identical folds. Pass "
            f"require_matching_protocol=False only if you have a specific "
            f"reason and intend to say so."
        )

    sizes_a = result_a.fold_sizes
    sizes_b = result_b.fold_sizes
    if sizes_a != sizes_b:
        raise ValueError(
            "The two results have different fold sizes, so their scores are "
            "not paired. Evaluate both models with the same splitter on the "
            "same data before comparing."
        )

    scores_a = result_a.scores(metric)
    scores_b = result_b.scores(metric)
    n_folds = scores_a.shape[0]

    # Fold sizes vary across folds under some protocols, so the correction
    # uses the mean. This matches the practice in the literature and is exact
    # whenever folds are equal-sized, which is the common case.
    n_train = round(float(np.mean([size[0] for size in sizes_a])))
    n_test = round(float(np.mean([size[1] for size in sizes_a])))

    statistic, p_value, standard_error = corrected_resampled_ttest(
        scores_a, scores_b, n_train=n_train, n_test=n_test
    )
    differences = scores_a - scores_b
    mean_difference = float(np.mean(differences))
    naive_error = float(np.sqrt(np.var(differences, ddof=1) / n_folds))

    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_folds - 1))
    interval = (
        mean_difference - critical * standard_error,
        mean_difference + critical * standard_error,
    )

    equivalence_p = (
        None
        if equivalence_bound is None
        else tost_equivalence(
            scores_a,
            scores_b,
            bound=equivalence_bound,
            n_train=n_train,
            n_test=n_test,
        )
    )

    return ComparisonResult(
        mean_difference=mean_difference,
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=n_folds - 1,
        corrected_standard_error=standard_error,
        naive_standard_error=naive_error,
        confidence_interval=interval,
        effect_size=hedges_g(scores_a, scores_b),
        n_folds=n_folds,
        alpha=alpha,
        equivalence_bound=equivalence_bound,
        equivalence_p_value=equivalence_p,
        minimum_detectable_effect=minimum_detectable_effect(
            scores_a,
            scores_b,
            n_train=n_train,
            n_test=n_test,
            alpha=alpha,
            power=power,
        ),
    )


def holm_bonferroni(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Uniformly more powerful than plain Bonferroni while controlling the same
    family-wise error rate, so there is no reason to prefer Bonferroni.

    This matters concretely for the thesis: comparing one quantum model
    against three classical baselines across six datasets is eighteen tests,
    and at ``alpha = 0.05`` roughly one spurious "significant" result is
    expected by chance alone. Reporting the largest of them as a finding is
    how a null result becomes a false positive.

    Args:
        p_values: Uncorrected p-values.
        alpha: Family-wise error rate.

    Returns:
        Tuple of adjusted p-values and rejection flags, both in the input
        order.

    Raises:
        ValueError: If any p-value lies outside ``[0, 1]`` or the sequence is
            empty.
    """
    values = np.asarray(p_values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one p-value is required.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p-values must lie in [0, 1].")

    n_tests = values.size
    order = np.argsort(values)
    sorted_values = values[order]

    # Step-down: multiply each by the number of remaining tests, then enforce
    # monotonicity so an adjusted p-value never decreases down the list.
    scaled = sorted_values * (n_tests - np.arange(n_tests))
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.minimum(np.maximum.accumulate(adjusted_sorted), 1.0)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted < alpha
