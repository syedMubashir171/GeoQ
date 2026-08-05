"""Significance against chance: permutation tests, bootstrap, chance levels.

Move 11 compares two models. This module answers the prior question: is a
single model doing anything at all?

Why permutation rather than a t-test against 0.5
------------------------------------------------
Comparing a mean accuracy to 0.5 with a t-test assumes the scores are normally
distributed, independent, and centred on 0.5 under the null. None of those
holds for cross-validated BCI results. Accuracy is bounded, folds share
training data, and with few trials the sampling distribution of accuracy under
the null is visibly discrete and skewed.

A permutation test assumes none of this. It builds the null empirically by
destroying the label-feature association and re-running the entire pipeline,
so whatever biases the pipeline has -- optimistic protocol, leaky preprocessing
inside the estimator, a classifier that exploits class imbalance -- appear in
the null distribution too and are subtracted out. That property is what makes
it the right tool for a framework whose central concern is inflated results.

The permutation scheme is the test
----------------------------------
For grouped data the labels must be permuted *within* each subject. Permuting
globally would also destroy the subject structure, producing a null in which
the model faces an easier problem than the real one: any subject-level signal
that inflates the observed score would be absent from the null, and the
resulting p-value would be too small. :func:`permutation_test` therefore
permutes within groups by default and says so in its result, because "we ran a
permutation test" means nothing without the scheme.

Never report p = 0
------------------
The p-value is estimated as ``(1 + #{null >= observed}) / (1 + n_permutations)``
following Phipson and Smyth (2010). Without the added one, a model that beats
every permutation is reported at ``p = 0``, which claims infinite evidence from
a finite sample. The floor is ``1 / (1 + n_permutations)``: a thousand
permutations cannot support a claim below about ``0.001``, and
:class:`PermutationResult` reports that floor so the limit is visible.

References
----------
Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero.
    *Statistical Applications in Genetics and Molecular Biology*, 9(1).
Combrisson, E., & Jerbi, K. (2015). Exceeding chance level by chance.
    *Journal of Neuroscience Methods*, 250, 126-136.
Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier
    performance. *JMLR*, 11, 1833-1863.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "BootstrapResult",
    "PermutationResult",
    "bootstrap_ci",
    "empirical_chance_level",
    "permutation_test",
    "permute_labels",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Chance level
# --------------------------------------------------------------------------- #


def empirical_chance_level(
    n_samples: int, n_classes: int = 2, *, alpha: float = 0.05
) -> float:
    """Accuracy that a random classifier exceeds with probability ``alpha``.

    The theoretical chance level of a balanced two-class problem is 0.5, but
    that is the *mean* of the null distribution, not its upper tail. With few
    trials the tail is wide, and a random classifier clears 0.5 by a
    comfortable margin surprisingly often.

    Concretely, at 40 test trials the 95th percentile of the null is 0.625:
    an accuracy of 0.60 on a single subject's fold is not evidence of
    anything, despite being ten points above "chance". Single-subject BCI
    folds are routinely this small, which is how a table of per-subject
    accuracies comes to contain several apparently-above-chance entries that
    are nothing of the kind.

    Args:
        n_samples: Number of test trials.
        n_classes: Number of balanced classes.
        alpha: Upper-tail probability.

    Returns:
        The accuracy threshold.

    Raises:
        ValueError: If the arguments are out of range.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")

    threshold = stats.binom.ppf(1.0 - alpha, n_samples, 1.0 / n_classes)
    return float(threshold) / n_samples


# --------------------------------------------------------------------------- #
# Permutation
# --------------------------------------------------------------------------- #


def permute_labels(
    y: ArrayLike,
    rng: np.random.Generator,
    *,
    groups: ArrayLike | None = None,
    within_groups: bool = True,
) -> NDArray[Any]:
    """Shuffle labels, optionally within each group.

    Args:
        y: Labels of shape ``(n_samples,)``.
        rng: Generator. Required, so a null distribution can be reproduced.
        groups: Per-sample group identifiers.
        within_groups: Whether to permute within each group separately.

    Returns:
        The permuted labels.

    Raises:
        ValueError: If ``within_groups`` is requested without ``groups``, or
            shapes disagree.
    """
    labels = np.asarray(y)
    if groups is None:
        if within_groups:
            raise ValueError(
                "within_groups=True requires `groups`. Permuting globally "
                "when the data has subject structure destroys that structure "
                "as well as the labels, giving a null in which the model "
                "faces an easier problem than the real one and a p-value that "
                "is too small."
            )
        return rng.permutation(labels)

    group_array = np.asarray(groups)
    if group_array.shape != labels.shape:
        raise ValueError(
            f"groups must have shape {labels.shape}, got {group_array.shape}."
        )
    if not within_groups:
        return rng.permutation(labels)

    permuted = labels.copy()
    for group in np.unique(group_array):
        mask = group_array == group
        permuted[mask] = rng.permutation(labels[mask])
    return permuted


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of a permutation test against chance.

    Attributes:
        observed: The score on the real labels.
        null_scores: Scores under permuted labels.
        p_value: Phipson-Smyth estimate,
            ``(1 + #{null >= observed}) / (1 + n_permutations)``.
        n_permutations: Number of permutations run.
        metric: Metric name.
        within_groups: Whether labels were permuted within groups. Recorded
            because the scheme *is* the test; the same numbers under a
            different scheme answer a different question.
        null_mean: Mean of the null distribution. Compare it with the metric's
            theoretical chance value: a large gap means the pipeline itself is
            optimistic, and the permutation test is measuring that.
        null_std: Standard deviation of the null distribution.
    """

    observed: float
    null_scores: NDArray[np.float64]
    p_value: float
    n_permutations: int
    metric: str
    within_groups: bool
    null_mean: float = 0.0
    null_std: float = 0.0

    @property
    def minimum_achievable_p(self) -> float:
        """Smallest p-value this many permutations can produce."""
        return 1.0 / (1.0 + self.n_permutations)

    @property
    def at_resolution_limit(self) -> bool:
        """Whether the p-value is pinned at the permutation floor.

        When True the true p-value may be far smaller, and the only way to
        find out is to run more permutations. Reporting the floor as though it
        were an estimate overstates precision.
        """
        return self.p_value <= self.minimum_achievable_p + 1e-12

    def summary(self) -> str:
        """Return a one-line conclusion, flagging the resolution limit."""
        suffix = (
            f" (at the {self.n_permutations}-permutation resolution limit; "
            f"the true p-value may be smaller)"
            if self.at_resolution_limit
            else ""
        )
        return (
            f"{self.metric}={self.observed:.4f} against a null of "
            f"{self.null_mean:.4f}+-{self.null_std:.4f}, "
            f"p={self.p_value:.4f}{suffix}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat record, excluding the full null distribution."""
        return {
            "metric": self.metric,
            "observed": self.observed,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "within_groups": self.within_groups,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "minimum_achievable_p": self.minimum_achievable_p,
            "at_resolution_limit": self.at_resolution_limit,
        }


def permutation_test(
    estimator: Any,
    X: ArrayLike,  # noqa: N803
    y: ArrayLike,
    *,
    splitter: Any,
    groups: ArrayLike | None = None,
    metric: str = "balanced_accuracy",
    n_permutations: int = 1000,
    random_state: int = 0,
    within_groups: bool = True,
) -> PermutationResult:
    """Test whether a model performs above chance, by permuting labels.

    The entire evaluation is re-run for each permutation, so any optimism in
    the protocol or the pipeline appears in the null distribution as well as
    in the observed score and is cancelled. That is the property that makes
    this test worth its cost.

    Cost is real: ``n_permutations`` times the number of folds, model fits.
    A thousand permutations of a nine-fold leave-one-subject-out evaluation is
    nine thousand fits, so start at a hundred while developing and raise it
    only for a result that will be reported.

    Args:
        estimator: An unfitted estimator or pipeline, cloned per fold.
        X: Trials.
        y: Labels.
        splitter: The evaluation protocol.
        groups: Per-trial subject identifiers.
        metric: Metric to test. Defaults to balanced accuracy, which is
            centred at chance regardless of class imbalance; plain accuracy
            under imbalance would give a null centred well above 0.5 and
            invite misreading.
        n_permutations: Number of permutations.
        random_state: Seed for the permutations.
        within_groups: Whether to permute within each subject.

    Returns:
        The permutation result.

    Raises:
        ValueError: If arguments are inconsistent.
    """
    from geoq.evaluation.protocol import evaluate

    labels = np.asarray(y)
    group_array = None if groups is None else np.asarray(groups)

    if n_permutations < 1:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}.")
    if within_groups and group_array is None:
        raise ValueError(
            "within_groups=True requires `groups`. Pass within_groups=False "
            "explicitly if the data genuinely has no group structure."
        )

    observed = evaluate(
        estimator,
        X,
        labels,
        splitter=splitter,
        groups=group_array,
        metrics=(metric,),
    ).mean(metric)

    if n_permutations >= 200:
        logger.info(
            "permutation_test: running %d permutations x %d folds model fits. "
            "This is the expensive part of the analysis; reduce "
            "n_permutations while developing.",
            n_permutations,
            splitter.get_n_splits(X, labels, group_array),
        )

    rng = np.random.default_rng(random_state)
    null: list[float] = []
    skipped = 0

    for _ in range(n_permutations):
        permuted = permute_labels(
            labels, rng, groups=group_array, within_groups=within_groups
        )
        try:
            null.append(
                evaluate(
                    estimator,
                    X,
                    permuted,
                    splitter=splitter,
                    groups=group_array,
                    metrics=(metric,),
                ).mean(metric)
            )
        except ValueError:
            # A permutation can leave a fold single-class, which the evaluator
            # refuses. Dropping it is the only correct option -- substituting
            # a score would fabricate a null sample -- but the count is
            # reported, because a high rate means the design is too small for
            # a permutation test to be meaningful.
            skipped += 1

    if not null:
        raise ValueError(
            "Every permutation produced an unusable fold, so no null "
            "distribution could be built. The dataset is too small or too "
            "imbalanced for a permutation test under this protocol."
        )
    if skipped:
        logger.warning(
            "permutation_test: %d of %d permutations produced a fold with a "
            "single class and were discarded. The null distribution rests on "
            "%d samples; a high discard rate means this design cannot support "
            "a permutation test.",
            skipped,
            n_permutations,
            len(null),
        )

    null_scores = np.array(null, dtype=np.float64)
    at_least_as_extreme = int(np.sum(null_scores >= observed))
    p_value = (1.0 + at_least_as_extreme) / (1.0 + null_scores.size)

    return PermutationResult(
        observed=float(observed),
        null_scores=null_scores,
        p_value=float(p_value),
        n_permutations=int(null_scores.size),
        metric=metric,
        within_groups=within_groups,
        null_mean=float(np.mean(null_scores)),
        null_std=float(np.std(null_scores, ddof=1)) if null_scores.size > 1 else 0.0,
    )


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap confidence interval.

    Attributes:
        estimate: The statistic on the original sample.
        lower: Lower confidence limit.
        upper: Upper confidence limit.
        confidence_level: Nominal coverage.
        method: ``"percentile"`` or ``"bca"``.
        n_resamples: Number of bootstrap resamples.
        standard_error: Bootstrap standard error.
        bias: Difference between the bootstrap mean and the estimate. A large
            value relative to the standard error means the percentile interval
            is misplaced and the bias-corrected one should be preferred.
    """

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    n_resamples: int
    standard_error: float = 0.0
    bias: float = 0.0
    resamples: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64), repr=False
    )

    @property
    def width(self) -> float:
        """Return the interval width."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies inside the interval."""
        return self.lower <= value <= self.upper

    def to_dict(self) -> dict[str, Any]:
        """Return a flat record, excluding the resamples."""
        return {
            "estimate": self.estimate,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "n_resamples": self.n_resamples,
            "standard_error": self.standard_error,
            "bias": self.bias,
        }


def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    method: str = "bca",
    random_state: int = 0,
) -> BootstrapResult:
    """Bootstrap confidence interval for a statistic of per-fold scores.

    Makes no normality assumption, which matters because accuracy is bounded
    and its sampling distribution is skewed near the ceiling -- exactly where
    a good BCI result sits.

    One caveat that must not be forgotten. Resampling folds treats them as
    independent, and cross-validation folds are not: they share training data.
    The interval is therefore too narrow, in the same direction and for the
    same reason as an uncorrected t-test. Use it to describe the spread of
    fold scores, and use the corrected test in
    :mod:`geoq.statistics.comparison` for inference.

    Args:
        values: Per-fold scores.
        statistic: Function applied to each resample.
        confidence_level: Nominal coverage.
        n_resamples: Number of bootstrap resamples.
        method: ``"percentile"`` or ``"bca"``. BCa corrects for bias and
            skewness and is preferred; it needs at least three observations.
        random_state: Seed.

    Returns:
        The interval.

    Raises:
        ValueError: If the arguments are out of range or the sample is too
            small for the requested method.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError(
            f"values must be a one-dimensional sample of at least two "
            f"observations, got shape {sample.shape}."
        )
    if not np.isfinite(sample).all():
        raise ValueError("values contain non-finite entries.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must lie strictly in (0, 1), got {confidence_level}."
        )
    if method not in {"percentile", "bca"}:
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}.")
    if method == "bca" and sample.size < 3:
        raise ValueError(
            "BCa needs at least three observations for its jackknife "
            "acceleration estimate. Use method='percentile' for smaller "
            "samples, and treat the interval with corresponding caution."
        )

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, sample.size, size=(n_resamples, sample.size))
    resamples = np.array([statistic(sample[row]) for row in indices])

    estimate = float(statistic(sample))
    alpha = 1.0 - confidence_level

    if method == "percentile":
        lower, upper = np.quantile(resamples, [alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        lower, upper = _bca_limits(sample, resamples, estimate, statistic, alpha)

    return BootstrapResult(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        standard_error=float(np.std(resamples, ddof=1)),
        bias=float(np.mean(resamples) - estimate),
        resamples=resamples,
    )


def _bca_limits(
    sample: NDArray[np.float64],
    resamples: NDArray[np.float64],
    estimate: float,
    statistic: Callable[[NDArray[np.float64]], float],
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated percentile limits.

    Two adjustments to the plain percentile interval. The bias correction
    ``z0`` shifts it when the bootstrap distribution is not centred on the
    estimate. The acceleration ``a``, from a jackknife, adjusts for the
    statistic's variance changing with its value -- which it does for a
    bounded quantity like accuracy near the ceiling.

    Args:
        sample: Original observations.
        resamples: Bootstrap statistic values.
        estimate: Statistic on the original sample.
        statistic: The statistic function, for the jackknife.
        alpha: One minus the confidence level.

    Returns:
        Lower and upper limits.
    """
    proportion_below = float(np.mean(resamples < estimate))
    # Guard the degenerate cases: with every resample on one side of the
    # estimate the normal quantile is infinite, and BCa is undefined. Falling
    # back to the percentile interval is the honest response.
    if proportion_below <= 0.0 or proportion_below >= 1.0:
        logger.debug(
            "bootstrap_ci: BCa bias correction is undefined (all resamples on "
            "one side of the estimate); falling back to percentile limits."
        )
        return tuple(np.quantile(resamples, [alpha / 2.0, 1.0 - alpha / 2.0]))

    bias_correction = float(stats.norm.ppf(proportion_below))

    jackknife = np.array(
        [statistic(np.delete(sample, index)) for index in range(sample.size)]
    )
    deviations = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    acceleration = (
        0.0 if denominator == 0.0 else float(np.sum(deviations**3)) / denominator
    )

    def adjusted(probability: float) -> float:
        z = stats.norm.ppf(probability)
        shifted = bias_correction + (bias_correction + z) / (
            1.0 - acceleration * (bias_correction + z)
        )
        return float(stats.norm.cdf(shifted))

    lower_quantile = adjusted(alpha / 2.0)
    upper_quantile = adjusted(1.0 - alpha / 2.0)
    return tuple(np.quantile(resamples, [lower_quantile, upper_quantile]))
