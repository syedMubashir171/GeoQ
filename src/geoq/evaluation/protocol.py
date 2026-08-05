"""Evaluation runner: pipeline plus protocol yields a traceable result table.

This is the object every experiment in the thesis produces and every
statistical test consumes. It ties together the components built so far --
covariance estimation, tangent projection, classifiers, and splitters -- into
one call whose output can be traced back to the protocol that produced it.

Three design commitments
------------------------
**Per-fold results are the record; the mean is a derived view.** A single
accuracy figure cannot be tested for significance, cannot show which subject
failed, and cannot be re-analysed later under a different question. Every
fold's scores, sizes, and timings are kept, and the mean is computed from them
on demand.

**Fold sizes are recorded because the statistics need them.** Cross-validation
folds share training data, so their scores are correlated and an ordinary
paired t-test over folds is anticonservative -- it will declare differences
significant that are not. The Nadeau-Bengio corrected resampled test fixes
this, and its correction term is a function of the train/test size ratio. That
ratio must therefore be captured at evaluation time; it cannot be recovered
from a table of scores afterwards.

**The protocol travels with the numbers.** Every result carries the
:class:`~geoq.evaluation.splitters.SplitterInfo` of the splitter that produced
it, so a reader can tell from the results file alone whether a figure is a
generalisation estimate or a within-subject one.

Nested selection
----------------
When a ``param_grid`` is supplied, hyperparameters are chosen by an inner
cross-validation *inside each outer training fold*. Selecting them once on the
whole dataset and then reporting outer-fold scores is a leak: the choice has
already seen the test data, and the resulting estimate is optimistic by an
amount that grows with the size of the grid. Nested selection is the only way
to report a tuned model's performance honestly.

References
----------
Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
    *Machine Learning*, 52(3), 239-281.
Varoquaux, G. et al. (2017). Assessing and tuning brain decoders:
    cross-validation, caveats, and guidelines. *NeuroImage*, 145, 166-179.
Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection
    and subsequent selection bias in performance evaluation. *JMLR*, 11,
    2079-2107.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from geoq.evaluation.splitters import BaseSplitter, SplitterInfo

__all__ = [
    "METRIC_FUNCTIONS",
    "PROBABILITY_METRICS",
    "EvaluationResult",
    "FoldResult",
    "evaluate",
]

logger = logging.getLogger(__name__)


def _kappa(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """Cohen's kappa, the standard BCI metric.

    Reported because accuracy is uninterpretable without knowing the class
    balance: 70 percent means something very different on a balanced two-class
    problem than on one where 70 percent of trials share a label. Kappa is
    zero at chance regardless of balance, which makes results comparable
    across datasets and paradigms.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Cohen's kappa.
    """
    return float(cohen_kappa_score(y_true, y_pred))


def _roc_auc(y_true: NDArray[Any], y_score: NDArray[Any]) -> float:
    """Area under the ROC curve, handling both binary and multiclass cases.

    Args:
        y_true: True labels.
        y_score: Predicted class probabilities of shape
            ``(n_samples, n_classes)``.

    Returns:
        The AUC.
    """
    n_classes = y_score.shape[1]
    if n_classes == 2:
        return float(roc_auc_score(y_true, y_score[:, 1]))
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


METRIC_FUNCTIONS: dict[str, Any] = {
    "accuracy": lambda t, p: float(accuracy_score(t, p)),
    "balanced_accuracy": lambda t, p: float(balanced_accuracy_score(t, p)),
    "kappa": _kappa,
    "f1_macro": lambda t, p: float(f1_score(t, p, average="macro")),
    "roc_auc": _roc_auc,
}
"""Registry mapping metric names to callables."""

PROBABILITY_METRICS: frozenset[str] = frozenset({"roc_auc"})
"""Metrics scored against predicted probabilities rather than labels."""

DEFAULT_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "kappa")
"""Metrics reported unless the caller asks for others.

Kappa is included by default rather than offered as an option: an accuracy
figure without it is not interpretable across datasets with different class
balance, and reporting accuracy alone is how a 70-percent result on an
imbalanced dataset comes to look like a finding.
"""


@dataclass(frozen=True)
class FoldResult:
    """Everything recorded for a single evaluation fold.

    Attributes:
        fold: Zero-based fold index.
        scores: Metric name to value on the test fold.
        n_train: Number of training trials. Recorded because the corrected
            resampled t-test needs the train/test ratio, which cannot be
            recovered from scores alone.
        n_test: Number of test trials.
        test_groups: Subjects present in the test fold, as a sorted tuple.
            Under leave-one-subject-out this identifies which participant the
            fold's score belongs to, which is what makes a per-subject
            breakdown possible.
        train_scores: Metrics on the training fold, when requested. A large
            train-test gap is a diagnosis, not a failure, and it is worth
            having when a model underperforms.
        fit_seconds: Wall-clock time to fit.
        score_seconds: Wall-clock time to predict and score.
        best_params: Hyperparameters chosen by the inner search, when nested
            selection was used. Recorded per fold because instability across
            folds is itself a finding: a grid whose winner changes every fold
            is not selecting a better model, it is selecting noise.
    """

    fold: int
    scores: dict[str, float]
    n_train: int
    n_test: int
    test_groups: tuple[Any, ...] = ()
    train_scores: dict[str, float] | None = None
    fit_seconds: float = 0.0
    score_seconds: float = 0.0
    best_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """The complete outcome of one evaluation run.

    Attributes:
        folds: Per-fold results, in evaluation order.
        protocol: Guarantees of the splitter that produced the folds.
        estimator_repr: Representation of the evaluated pipeline.
        metrics: Metric names computed.
        n_samples: Total trials in the dataset.
        n_classes: Number of distinct labels.
        class_balance: Label to proportion, needed to interpret accuracy.
        chance_accuracy: Accuracy of always predicting the majority class.
            The honest baseline for any accuracy figure, and frequently higher
            than readers expect on imbalanced BCI datasets.
        param_grid: The search grid, when nested selection was used.
    """

    folds: tuple[FoldResult, ...]
    protocol: SplitterInfo
    estimator_repr: str
    metrics: tuple[str, ...]
    n_samples: int
    n_classes: int
    class_balance: dict[Any, float] = field(default_factory=dict)
    chance_accuracy: float = 0.0
    param_grid: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    def scores(self, metric: str) -> NDArray[np.float64]:
        """Return the per-fold values of one metric.

        Args:
            metric: Metric name.

        Returns:
            Array of shape ``(n_folds,)``.

        Raises:
            KeyError: If the metric was not computed.
        """
        if metric not in self.metrics:
            raise KeyError(
                f"Metric {metric!r} was not computed. Available: {list(self.metrics)}."
            )
        return np.array([fold.scores[metric] for fold in self.folds])

    def mean(self, metric: str) -> float:
        """Return the mean of one metric across folds."""
        return float(np.mean(self.scores(metric)))

    def std(self, metric: str) -> float:
        """Return the standard deviation of one metric across folds.

        Note that this is a description of fold-to-fold spread, not a standard
        error. Cross-validation folds share training data, so dividing it by
        the square root of the fold count would understate the true
        uncertainty. Use the corrected resampled test in the statistics layer
        for inference.
        """
        return float(np.std(self.scores(metric), ddof=1))

    @property
    def fold_sizes(self) -> tuple[tuple[int, int], ...]:
        """Return ``(n_train, n_test)`` per fold, for the corrected t-test."""
        return tuple((fold.n_train, fold.n_test) for fold in self.folds)

    def to_frame(self):
        """Return a tidy per-fold table.

        One row per fold, with the protocol's guarantees repeated on every row
        so that a table concatenated across experiments remains
        self-describing.

        Returns:
            A :class:`pandas.DataFrame`.
        """
        import pandas as pd

        rows = []
        for result in self.folds:
            row: dict[str, Any] = {
                "fold": result.fold,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "test_groups": ",".join(str(g) for g in result.test_groups),
                "fit_seconds": result.fit_seconds,
                "score_seconds": result.score_seconds,
                "protocol": self.protocol.name,
                "subject_independent": self.protocol.subject_independent,
                "temporally_disjoint": self.protocol.temporally_disjoint,
            }
            row.update(result.scores)
            if result.train_scores is not None:
                row.update({f"train_{k}": v for k, v in result.train_scores.items()})
            if result.best_params is not None:
                row.update({f"param_{k}": v for k, v in result.best_params.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """Return a compact dictionary suitable for a results file.

        Includes the chance level alongside every accuracy-like metric,
        because an accuracy reported without it invites the reader to assume a
        balanced problem.
        """
        summary: dict[str, Any] = {
            "protocol": self.protocol.name,
            "subject_independent": self.protocol.subject_independent,
            "temporally_disjoint": self.protocol.temporally_disjoint,
            "is_optimistic": self.protocol.is_optimistic,
            "estimator": self.estimator_repr,
            "n_folds": len(self.folds),
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "chance_accuracy": self.chance_accuracy,
        }
        for metric in self.metrics:
            summary[f"{metric}_mean"] = self.mean(metric)
            summary[f"{metric}_std"] = self.std(metric)
        return summary

    def __repr__(self) -> str:
        """Return a one-line summary naming the protocol and headline metric."""
        headline = self.metrics[0] if self.metrics else "accuracy"
        flag = " OPTIMISTIC" if self.protocol.is_optimistic else ""
        return (
            f"EvaluationResult({self.protocol.name}{flag}, "
            f"{len(self.folds)} folds, "
            f"{headline}={self.mean(headline):.3f}+-{self.std(headline):.3f}, "
            f"chance={self.chance_accuracy:.3f})"
        )


def _validate_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    """Validate requested metric names.

    Args:
        metrics: Requested names.

    Returns:
        The validated names as a tuple.

    Raises:
        ValueError: If any name is unknown or the sequence is empty.
    """
    if not metrics:
        raise ValueError("At least one metric must be requested.")
    unknown = [name for name in metrics if name not in METRIC_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}. Available: {sorted(METRIC_FUNCTIONS)}."
        )
    return tuple(metrics)


def _score(
    estimator: BaseEstimator,
    x: Any,
    y_true: NDArray[Any],
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """Compute every requested metric for one dataset.

    Args:
        estimator: A fitted estimator.
        x: Feature data.
        y_true: True labels.
        metrics: Metric names.

    Returns:
        Metric name to value.

    Raises:
        AttributeError: If a probability metric was requested but the
            estimator provides no ``predict_proba``.
    """
    scores: dict[str, float] = {}
    predictions = None
    probabilities = None

    for name in metrics:
        if name in PROBABILITY_METRICS:
            if probabilities is None:
                if not hasattr(estimator, "predict_proba"):
                    raise AttributeError(
                        f"Metric {name!r} needs predicted probabilities, but "
                        f"{type(estimator).__name__} has no predict_proba. "
                        f"Request a label-based metric instead, or use an "
                        f"estimator that exposes probabilities."
                    )
                probabilities = np.asarray(estimator.predict_proba(x))
            scores[name] = METRIC_FUNCTIONS[name](y_true, probabilities)
        else:
            if predictions is None:
                predictions = np.asarray(estimator.predict(x))
            scores[name] = METRIC_FUNCTIONS[name](y_true, predictions)

    return scores


def evaluate(
    estimator: BaseEstimator,
    X: ArrayLike,  # noqa: N803
    y: ArrayLike,
    *,
    splitter: BaseSplitter,
    groups: ArrayLike | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    inner_splitter: BaseSplitter | int | None = None,
    selection_metric: str = "balanced_accuracy",
    return_train_scores: bool = False,
) -> EvaluationResult:
    """Evaluate an estimator under a named protocol.

    Args:
        estimator: An unfitted scikit-learn estimator or pipeline. It is cloned
            before every fold, so no state can carry from one fold to the next.
        X: Trials, indexed along the first axis.
        y: Labels of shape ``(n_trials,)``.
        splitter: The evaluation protocol.
        groups: Per-trial subject identifiers, required by most protocols.
        metrics: Metric names to compute. Defaults to accuracy, balanced
            accuracy and kappa.
        param_grid: Optional hyperparameter grid. When given, selection runs
            as an inner cross-validation inside each outer training fold.
        inner_splitter: Splitter or fold count for the inner search. Required
            when ``param_grid`` is given.
        selection_metric: Metric optimised by the inner search.
        return_train_scores: Whether to also score the training fold.

    Returns:
        The evaluation result.

    Raises:
        ValueError: If arguments are inconsistent or a metric is unknown.
    """
    metric_names = _validate_metrics(metrics)
    x_array = np.asarray(X)
    y_array = np.asarray(y)

    if y_array.shape[0] != x_array.shape[0]:
        raise ValueError(
            f"X has {x_array.shape[0]} trials but y has {y_array.shape[0]} labels."
        )
    if param_grid is not None and inner_splitter is None:
        raise ValueError(
            "param_grid was given without inner_splitter. Hyperparameters must "
            "be selected inside each outer training fold; choosing them once "
            "on the whole dataset would let the selection see the test data "
            "and inflate every score that follows."
        )
    if param_grid is not None and selection_metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown selection_metric {selection_metric!r}. Available: "
            f"{sorted(METRIC_FUNCTIONS)}."
        )

    groups_array = None if groups is None else np.asarray(groups)
    labels, counts = np.unique(y_array, return_counts=True)
    proportions = counts / counts.sum()

    protocol = splitter.info
    if protocol.is_optimistic:
        logger.warning(
            "Evaluating under protocol %r, which does not provide both "
            "evaluation guarantees. %s",
            protocol.name,
            protocol.caveat or "Interpret the scores accordingly.",
        )

    fold_results: list[FoldResult] = []

    for index, (train_index, test_index) in enumerate(
        splitter.split(x_array, y_array, groups_array)
    ):
        if np.unique(y_array[train_index]).shape[0] < 2:
            raise ValueError(
                f"Fold {index} has a single class in its training set. This "
                f"usually means the protocol was applied without stratifying, "
                f"or a subject performed only one task condition."
            )
        if np.unique(y_array[test_index]).shape[0] < 2:
            #  Not merely awkward: with one class present, kappa is undefined,
            #  balanced accuracy reduces to that class's recall, and accuracy
            #  is either 0 or 1. Averaging such a fold with the others
            #  produces a number that looks like a score and is not one, so
            #  this fails rather than warning.
            present = np.unique(y_array[test_index]).tolist()
            raise ValueError(
                f"Fold {index} has a single class ({present}) in its test set, "
                f"so kappa is undefined and balanced accuracy degenerates to "
                f"one class's recall. Under leave-one-subject-out this means "
                f"a participant performed only one task condition; exclude "
                f"that subject explicitly, or use a protocol whose test folds "
                f"contain every class."
            )

        model = _build_fold_model(
            estimator,
            param_grid,
            inner_splitter,
            selection_metric,
        )

        started = time.perf_counter()
        _fit_fold(model, x_array, y_array, groups_array, train_index)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        scores = _score(model, x_array[test_index], y_array[test_index], metric_names)
        score_seconds = time.perf_counter() - started

        train_scores = (
            _score(model, x_array[train_index], y_array[train_index], metric_names)
            if return_train_scores
            else None
        )

        fold_results.append(
            FoldResult(
                fold=index,
                scores=scores,
                n_train=int(train_index.size),
                n_test=int(test_index.size),
                test_groups=(
                    ()
                    if groups_array is None
                    else tuple(np.unique(groups_array[test_index]).tolist())
                ),
                train_scores=train_scores,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
                best_params=getattr(model, "best_params_", None),
            )
        )

    if not fold_results:
        raise ValueError(
            "The splitter produced no folds. Check that `groups` is supplied "
            "and that the dataset contains enough subjects or trials."
        )

    return EvaluationResult(
        folds=tuple(fold_results),
        protocol=protocol,
        estimator_repr=repr(estimator),
        metrics=metric_names,
        n_samples=int(x_array.shape[0]),
        n_classes=int(labels.shape[0]),
        class_balance={
            label: float(proportion)
            for label, proportion in zip(labels.tolist(), proportions, strict=True)
        },
        chance_accuracy=float(proportions.max()),
        param_grid=None if param_grid is None else dict(param_grid),
    )


def _build_fold_model(
    estimator: BaseEstimator,
    param_grid: Mapping[str, Sequence[Any]] | None,
    inner_splitter: BaseSplitter | int | None,
    selection_metric: str,
) -> BaseEstimator:
    """Return a freshly cloned model for one outer fold.

    Cloning is what guarantees fold independence. A reused estimator would
    carry fitted attributes from the previous fold, and for a transformer that
    caches anything the previous fold's data would influence this one's -- a
    leak that no splitter can prevent.

    Args:
        estimator: The template estimator.
        param_grid: Optional search grid.
        inner_splitter: Inner cross-validator.
        selection_metric: Metric optimised by the inner search.

    Returns:
        An unfitted model for this fold.
    """
    model = clone(estimator)
    if param_grid is None:
        return model

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
        "kappa": "matthews_corrcoef",
    }.get(selection_metric, "balanced_accuracy")

    return GridSearchCV(
        model,
        param_grid=dict(param_grid),
        cv=inner_splitter,
        scoring=scoring,
        refit=True,
    )


def _fit_fold(
    model: BaseEstimator,
    x: NDArray[Any],
    y: NDArray[Any],
    groups: NDArray[Any] | None,
    train_index: NDArray[np.intp],
) -> None:
    """Fit ``model`` on the training portion of one fold.

    When the inner search is group-aware, the training fold's subject labels
    must be forwarded so the inner split is subject-independent too. Omitting
    them would make hyperparameter selection a within-subject procedure inside
    an ostensibly subject-independent evaluation -- a subtle leak that leaves
    the outer protocol looking correct.

    Args:
        model: The model for this fold.
        x: Full feature array.
        y: Full label array.
        groups: Full subject identifiers, or None.
        train_index: Indices of the training portion.
    """
    x_train = x[train_index]
    y_train = y[train_index]

    if isinstance(model, GridSearchCV) and groups is not None:
        inner_cv = model.cv
        if isinstance(inner_cv, BaseSplitter):
            model.fit(x_train, y_train, groups=groups[train_index])
            return

    model.fit(x_train, y_train)
