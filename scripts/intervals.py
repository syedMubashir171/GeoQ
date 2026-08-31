"""Confidence intervals for every quantity the manuscript reports.

Why this is a separate script
-----------------------------
The manuscript reports Spearman correlations, regression slopes, alignment
benefits and cross-dataset comparisons, and until now gave a point estimate
and a p-value for each. A p-value states whether an effect is
distinguishable from zero; it says nothing about how precisely the effect is
known, which is the quantity a reader needs in order to judge whether a
correlation of 0.55 over thirty conditions means much.

Every interval here is computed from the archived CSV files, so an interval
cannot drift from the estimate it accompanies.

Choice of method
----------------
Correlations and benefits use a percentile bootstrap over the unit of
observation, which for a correlation is the pair and for a mean is the
condition. Resampling pairs rather than the two variables independently
preserves the association under resampling, which is what an interval on a
correlation requires.

Where observations are not independent, an interval computed as though they
were is too narrow. The subject-level quantities are therefore taken from
the mixed models rather than bootstrapped, since those account for the
repeated measurement of each subject directly; this script reports them for
completeness but does not recompute them.

Correlations are also reported with the Fisher-transform interval as a check
on the bootstrap. The two use different assumptions, and agreement between
them is worth more than either alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper1"
)
N_BOOT = 10000
SEED = 0


def load(pattern: str) -> pd.DataFrame:
    """Concatenate every CSV matching a glob, or exit with a clear message."""
    files = sorted(RESULTS.glob(pattern))
    if not files:
        sys.exit(f"No files matching {pattern} under {RESULTS}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def spearman_ci(x, y, n_boot: int = N_BOOT) -> str:
    """Spearman correlation with bootstrap and Fisher intervals.

    Args:
        x: First variable.
        y: Second variable.
        n_boot: Bootstrap resamples.

    Returns:
        A formatted line giving the estimate and both intervals.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    rho, p = stats.spearmanr(x, y)

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        #  A resample can be degenerate: with a predictor taking only two
        #  values, as manifold dimension does, a resample may draw a single
        #  level and leave the correlation undefined. Those draws are
        #  discarded rather than recorded as zero, which would drag the
        #  interval toward the null. The threshold is two distinct values,
        #  the minimum a rank correlation needs, rather than three.
        if np.unique(x[index]).size >= 2 and np.unique(y[index]).size >= 2:
            value = stats.spearmanr(x[index], y[index])[0]
            if np.isfinite(value):
                draws.append(value)
    draws = np.asarray(draws)
    if draws.size < n_boot // 10:
        #  Too few usable resamples for a percentile interval to mean
        #  anything. Reporting this is better than reporting an interval
        #  computed from a handful of draws.
        return (
            f"rho={rho:+.3f}  bootstrap unavailable "
            f"({draws.size} usable resamples)  p={p:.4g}  n={n}"
        )
    boot_lo, boot_hi = np.percentile(draws, [2.5, 97.5])

    #  Fisher's transform assumes bivariate normality, which rank data do not
    #  satisfy exactly; it is reported as an independent check rather than as
    #  the primary interval.
    if n > 3 and abs(rho) < 1:
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 3)
        fisher = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))
        fisher_text = f"Fisher [{fisher[0]:+.3f}, {fisher[1]:+.3f}]"
    else:
        fisher_text = "Fisher n/a"

    return (
        f"rho={rho:+.3f}  bootstrap [{boot_lo:+.3f}, {boot_hi:+.3f}]  "
        f"{fisher_text}  p={p:.4g}  n={n}"
    )


def mean_ci(values, n_boot: int = N_BOOT) -> str:
    """Mean with a percentile bootstrap interval."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return f"{values.mean():+.3f}  [{low:+.3f}, {high:+.3f}]  n={values.size}"


def slope_ci(x, y, n_boot: int = N_BOOT) -> str:
    """Least-squares slope and intercept with bootstrap intervals.

    Pairs are resampled together, so the interval reflects uncertainty in the
    relationship rather than in either variable alone.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    slope, intercept = np.polyfit(x, y, 1)

    rng = np.random.default_rng(SEED)
    slopes, intercepts = [], []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        if np.unique(x[index]).size > 1:
            s, i = np.polyfit(x[index], y[index], 1)
            slopes.append(s)
            intercepts.append(i)
    s_lo, s_hi = np.percentile(slopes, [2.5, 97.5])
    i_lo, i_hi = np.percentile(intercepts, [2.5, 97.5])
    return (
        f"slope {slope:+.4f} [{s_lo:+.4f}, {s_hi:+.4f}]   "
        f"intercept {intercept:+.4f} [{i_lo:+.4f}, {i_hi:+.4f}]   n={n}"
    )


def banner(text: str) -> None:
    """Print a heading matching the manuscript's section numbering."""
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


# --------------------------------------------------------------------- #
banner("Section 5.4  band sweep, dataset 2a")

gain = load("A_gain_*.csv")
for model in ("mdm", "ts_lda"):
    sub = gain[gain.model == model]
    print(f"\n{model}, pooled over {len(sub)} configurations:")
    for name, label in (
        ("inter_subject_distance", "displacement"),
        ("spd_dimension", "dimension"),
        ("median_condition_number", "conditioning"),
    ):
        print(f"  gain ~ {label:13s} {spearman_ci(sub[name], sub.gain)}")

    for n_channels in sorted(sub.n_channels.unique()):
        cell = (
            sub[sub.n_channels == n_channels]
            .groupby("band")[
                ["inter_subject_distance", "median_condition_number", "gain"]
            ]
            .mean()
        )
        print(f"  {n_channels:2d} ch, across bands:")
        print(
            f"    displacement  {spearman_ci(cell.inter_subject_distance, cell.gain)}"
        )
        print(
            f"    conditioning  {spearman_ci(cell.median_condition_number, cell.gain)}"
        )

# --------------------------------------------------------------------- #
banner("Section 5.5  rotation-reducible residual")

rot = load("B2_rotation_*.csv")
keep = rot[rot.band != "theta"]
print(f"\nrotation ~ displacement, all {len(rot)} conditions:")
print(f"  {spearman_ci(rot.displacement, rot.after_rotational)}")
for model in ("mdm", "ts_lda"):
    column = f"{model}_gain"
    print(f"\n{model} gain ~ rotation-reducible component:")
    print(f"  all conditions  {spearman_ci(rot.after_rotational, rot[column])}")
    print(f"  excluding theta {spearman_ci(keep.after_rotational, keep[column])}")

# --------------------------------------------------------------------- #
banner("Section 5.7  cross-dataset comparison")

cross = load("F_cross_*.csv")
usable = cross[cross.kappa_raw > 0.02]
for model in ("mdm", "ts_lda"):
    sub = usable[usable.model == model]
    print(f"\n{model}:")
    for dataset in sorted(sub.dataset.unique()):
        cell = sub[sub.dataset == dataset]
        if len(cell) >= 4:
            print(f"  {dataset:14s} {slope_ci(cell.displacement, cell.gain)}")
    #  The intercepts are what the cross-dataset claim rests on, so whether
    #  their intervals overlap is the question a reader will ask.
    print(f"  pooled         {spearman_ci(sub.displacement, sub.gain)}")

# --------------------------------------------------------------------- #
banner("Section 5.1  alignment benefit, dataset 2a full montage")

subject = load("G_subject_*.csv")
full = subject[subject.n_channels == 18]
for model in ("mdm", "ts_lda"):
    cell = full[full.classifier == model]
    per_subject = cell.groupby("subject").gain.mean()
    print(f"  {model:7s} mean benefit {mean_ci(per_subject)}")

# --------------------------------------------------------------------- #
banner("Subject-level slopes: intervals from the mixed models")

print("""
These are reproduced from subject_level.py and split_sample.py rather than
recomputed. A bootstrap over the 180 subject-condition observations would
resample nine people as though they were 180, and would give an interval
narrower than the data supports; the mixed model accounts for the repeated
measurement directly.

  full data    MDM     +0.0406  [+0.0173, +0.0639]  p = 0.00063
               TS+LDA  -0.0208  [-0.0351, -0.0065]  p = 0.0044
               interaction -0.0887  [-0.1197, -0.0578]  p = 1.9e-08

  split sample MDM     +0.0386  [+0.0146, +0.0626]  p = 0.0016
               TS+LDA  -0.0171  [-0.0313, -0.0030]  p = 0.018
               interaction -0.0690  [-0.0974, -0.0407]  p = 1.8e-06
""")

print("=" * 74)
print("Check every interval above against the manuscript before submitting.")
print("=" * 74)
