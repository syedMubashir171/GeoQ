"""Per-subject scores under any protocol, computed the same way for each.

Why this script exists
----------------------
Comparing the standard deviation of *fold* scores across protocols is not a
like-for-like comparison. Under leave-one-subject-out a fold is one subject, so
the spread of fold scores is the spread across people. Under a shuffled
protocol every fold contains every subject, so a fold score is an average over
the whole cohort and its spread is the sampling noise of that average -- small
by construction, and unrelated to whether the protocol leaks.

Any claim about between-subject variability therefore has to be measured the
same way under both protocols. This script collects out-of-fold predictions,
which every protocol produces, and scores each subject separately from the
predictions made for that subject's own trials. The result is one number per
subject per protocol, comparable directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score


def out_of_fold_predictions(estimator, X, y, groups, splitter):  # noqa: N803
    """Return a prediction for every trial, made by a model that never saw it.

    Args:
        estimator: Unfitted estimator, cloned per fold.
        X: Features.
        y: Labels.
        groups: Subject identifier per trial.
        splitter: The evaluation protocol.

    Returns:
        Array of predictions aligned with ``y``.

    Raises:
        ValueError: If any trial is never in a test fold, which would leave a
            prediction undefined and silently bias the per-subject scores.
    """
    predictions = np.empty(len(y), dtype=object)
    covered = np.zeros(len(y), dtype=bool)

    for train, test in splitter.split(X, y, groups):
        model = clone(estimator).fit(X[train], y[train])
        predictions[test] = model.predict(X[test])
        covered[test] = True

    if not covered.all():
        raise ValueError(
            f"{int((~covered).sum())} trial(s) were never in a test fold, so "
            f"their predictions are undefined. Per-subject scores computed "
            f"from a partial set would be biased toward whichever subjects "
            f"happened to be covered."
        )
    return predictions


def per_subject_scores(estimator, X, y, groups, splitter):  # noqa: N803
    """Score each subject from out-of-fold predictions for that subject.

    Args:
        estimator: Unfitted estimator.
        X: Features.
        y: Labels.
        groups: Subject identifier per trial.
        splitter: The evaluation protocol.

    Returns:
        Dict mapping subject to its kappa and balanced accuracy.
    """
    predicted = out_of_fold_predictions(estimator, X, y, groups, splitter)
    scores = {}
    for subject in np.unique(groups):
        mask = groups == subject
        scores[subject] = {
            "kappa": float(cohen_kappa_score(y[mask], list(predicted[mask]))),
            "balanced_accuracy": float(
                balanced_accuracy_score(y[mask], list(predicted[mask]))
            ),
        }
    return scores
