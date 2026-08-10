"""Sweep the conditions under which Riemannian re-centring helps.

The question
------------
Alignment raises cross-subject MDM by 0.231 on BCI IV 2a (22 channels) and by
0.023 on 2b (3 channels). The obvious reading is that the gain scales with the
dimension of the SPD manifold. That reading is not supported by those two
numbers: the datasets differ in subjects, montage, paradigm and session
structure, so dimension is confounded with everything else.

This script isolates the variable by subsampling channels within a single
dataset. Subjects, paradigm, recording and preprocessing are held fixed; only
the channel count changes. At each count several random subsets are drawn and
averaged, so channel *count* is not confounded with channel *identity* -- a
sweep using nested subsets would let the specific electrodes chosen explain the
trend.

What the mechanism actually predicts
------------------------------------
Re-centring removes a per-subject congruence. Dimension is only a proxy for how
much displacement there is to remove, so the sharper prediction is that the
gain tracks the *inter-subject Frechet distance*: the mean geodesic distance
between subjects' reference means. If gain correlates with displacement more
tightly than with dimension, the mechanism is supported directly rather than by
association.

Three predictors are therefore recorded for every configuration: the SPD
dimension ``n(n+1)/2``, the median covariance condition number, and the
inter-subject Frechet distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.alignment import RiemannianAlignment, domain_references
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import distance_airm
from geoq.geometry.spd import condition_number
from geoq.models.classical.mdm import MDM


def inter_subject_distance(covariances, subjects, metric: str = "airm") -> float:
    """Mean geodesic distance between subjects' reference means.

    The mechanism's own quantity: how far apart the subjects' coordinate
    frames sit before anything is done about it.

    Args:
        covariances: SPD stack of shape ``(n_trials, n, n)``.
        subjects: Subject identifier per trial.
        metric: Geometry used for the reference means.

    Returns:
        The mean pairwise distance between subject means.
    """
    references = domain_references(covariances, subjects, metric=metric)
    keys = sorted(references)
    pairs = [
        float(distance_airm(references[a], references[b]))
        for index, a in enumerate(keys)
        for b in keys[index + 1 :]
    ]
    return float(np.mean(pairs))


def evaluate_condition(
    epochs, labels, subjects, *, channels, estimator, seed
) -> list[dict]:
    """Measure raw and aligned performance for one channel subset.

    Args:
        epochs: Raw epochs of shape ``(n_trials, n_channels, n_times)``.
        labels: Class labels.
        subjects: Subject identifier per trial.
        channels: Indices of the channels to keep.
        estimator: Covariance estimator name.
        seed: Subset identifier, recorded for traceability.

    Returns:
        One record per model, with the predictors and the alignment gain.
    """
    subset = epochs[:, channels, :]
    covariances = Covariances(
        estimator=estimator, audit_conditioning=False
    ).fit_transform(subset)
    aligned = RiemannianAlignment(assume_calibration_data=True).fit_transform(
        covariances, domains=subjects
    )

    n_channels = len(channels)
    predictors = {
        "n_channels": n_channels,
        "spd_dimension": n_channels * (n_channels + 1) // 2,
        "median_condition_number": float(np.median(condition_number(covariances))),
        "inter_subject_distance": inter_subject_distance(covariances, subjects),
        "estimator": estimator,
        "subset_seed": seed,
    }

    splitter = LeaveOneSubjectOut()
    records = []
    for model_name, model in (
        ("mdm", MDM()),
        ("ts_lda", make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())),
    ):
        scores = {}
        for label, features in (("raw", covariances), ("aligned", aligned)):
            scores[label] = evaluate(
                model,
                features,
                labels,
                groups=subjects,
                splitter=splitter,
                metrics=("kappa",),
            ).mean("kappa")
        records.append(
            {
                **predictors,
                "model": model_name,
                "kappa_raw": scores["raw"],
                "kappa_aligned": scores["aligned"],
                "gain": scores["aligned"] - scores["raw"],
            }
        )
    return records


def channel_sweep(
    dataset,
    *,
    counts=(3, 5, 8, 12, 16, 22),
    n_subsets: int = 5,
    estimator: str = "oas",
    seed: int = 0,
):
    """Vary the channel count within one dataset, holding everything else fixed.

    Args:
        dataset: An :class:`geoq.datasets.base.EEGDataset`.
        counts: Channel counts to test. The largest should be the full montage.
        n_subsets: Random subsets drawn per count, averaged over. One subset
            would confound the count with the particular electrodes chosen.
        estimator: Covariance estimator.
        seed: Master seed for subset selection.

    Returns:
        A :class:`pandas.DataFrame`, one row per model per subset.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    available = dataset.n_channels
    rows = []
    for count in counts:
        if count > available:
            continue
        # The full montage has only one subset, so drawing several would
        # repeat identical work and understate the variance at every other
        # count by comparison.
        repeats = 1 if count == available else n_subsets
        for repeat in range(repeats):
            channels = (
                np.arange(available)
                if count == available
                else rng.choice(available, size=count, replace=False)
            )
            rows.extend(
                evaluate_condition(
                    dataset.epochs,
                    dataset.labels,
                    dataset.subjects,
                    channels=np.sort(channels),
                    estimator=estimator,
                    seed=repeat,
                )
            )
            print(f"  {count:3d} channels, subset {repeat} done", flush=True)
    return pd.DataFrame(rows)


def correlate_gain(frame, model: str = "mdm"):
    """Report how well each predictor explains the alignment gain.

    Spearman rather than Pearson, because the relationships need not be linear
    and the predictors span orders of magnitude. Partial correlations are also
    given for inter-subject distance controlling for dimension, since the two
    are themselves correlated and the question is which one carries the effect.

    Args:
        frame: Output of :func:`channel_sweep`.
        model: Which model's gain to explain.

    Returns:
        A :class:`pandas.DataFrame` of correlations.
    """
    import pandas as pd
    from scipy import stats

    subset = frame[frame["model"] == model]
    gain = subset["gain"].to_numpy()

    rows = []
    for name in (
        "spd_dimension",
        "median_condition_number",
        "inter_subject_distance",
    ):
        values = subset[name].to_numpy()
        rho, p = stats.spearmanr(values, gain)
        rows.append({"predictor": name, "spearman_rho": rho, "p_value": p})

    # Partial correlation of displacement with gain, dimension held fixed.
    # If it survives, displacement is not merely standing in for dimension.
    dimension = subset["spd_dimension"].to_numpy()
    displacement = subset["inter_subject_distance"].to_numpy()

    def residual(target):
        design = np.vstack([np.ones_like(dimension), np.log(dimension)]).T
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ coefficients

    rho, p = stats.spearmanr(residual(displacement), residual(gain))
    rows.append(
        {
            "predictor": "inter_subject_distance | spd_dimension",
            "spearman_rho": rho,
            "p_value": p,
        }
    )
    return pd.DataFrame(rows)
