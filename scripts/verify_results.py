"""Regenerate every number in the Results section directly from the saved CSVs.

Why this exists
---------------
The manuscript's Results section was first assembled from figures accumulated
across a long sequence of analyses. Two of them were taken from the wrong run:
a conditioning contrast and a benefit contrast quoted from a 22-channel
experiment while the accompanying table reported an 18-channel one. Both
numbers were real measurements of real conditions, but they did not belong to
the table beside them.

That failure mode is systematic rather than accidental. The defence is to
regenerate every reported quantity from the archived data in one pass, print
it in the order the manuscript uses, and check the manuscript against the
output line by line. Nothing here recomputes an experiment; it only reads what
was written to disk and reports it.
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


def rho(x, y) -> str:
    """Return a formatted Spearman correlation with its sample size."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = stats.spearmanr(x, y)
    return f"rho={r:+.3f}  p={p:.4f}  n={len(x)}"


def partial_rho(y, x, z) -> str:
    """Spearman correlation of residuals after regressing both on log(z).

    The hybrid procedure described in Methods: least-squares removal of
    log(z) from each variable, then a rank correlation of what remains.
    """
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    design = np.vstack([np.ones_like(z), np.log(z)]).T

    def residual(v):
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    r, p = stats.spearmanr(residual(x), residual(y))
    return f"rho={r:+.3f}  p={p:.4f}  n={len(x)}"


def load(pattern: str) -> pd.DataFrame:
    """Concatenate every CSV matching a glob, or exit with a clear message."""
    files = sorted(RESULTS.glob(pattern))
    if not files:
        sys.exit(f"No files matching {pattern} under {RESULTS}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def banner(text: str) -> None:
    """Print a section heading matching the manuscript's numbering."""
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# --------------------------------------------------------------------- #
banner("4.2  CHANNEL COUNT  (from E_channel_sweep.csv, 8-30 Hz, OAS)")

#  The channel sweep has its own file. Reading it from the band sweep instead
#  silently restricts the analysis to the two channel counts that sweep used,
#  which is how an earlier draft came to report trend statistics over two
#  points while presenting a six-point table.
gain = load("A_gain_*.csv")
sweep = load("E_channel_sweep.csv")

print("\nTable 2 -- means over subsets at each channel count")
table2 = sweep.pivot_table(
    index="n_channels",
    columns="model",
    values=[
        "median_condition_number",
        "inter_subject_distance",
        "kappa_raw",
        "kappa_aligned",
        "gain",
    ],
).round(3)
print(table2.to_string())

print("\nSubsets per channel count:")
print(sweep[sweep.model == "mdm"].groupby("n_channels").size().to_string())

means = sweep.pivot_table(index="n_channels", columns="model", values="gain")
raw = sweep.pivot_table(index="n_channels", columns="model", values="kappa_raw")
ali = sweep.pivot_table(index="n_channels", columns="model", values="kappa_aligned")
counts = np.asarray(raw.index, float)

print("\nTrend with channel count (n = number of channel counts, subset means):")
print(f"  MDM raw       {rho(counts, raw['mdm'])}")
print(f"  MDM aligned   {rho(counts, ali['mdm'])}")
print(f"  TS+LDA raw    {rho(counts, raw['ts_lda'])}")
print(f"  TS+LDA aligned{rho(counts, ali['ts_lda'])}")
print(f"  raw gap       {rho(counts, raw['ts_lda'] - raw['mdm'])}")
print(f"  aligned gap   {rho(counts, ali['ts_lda'] - ali['mdm'])}")
print(f"  raw gap:     {(raw['ts_lda'] - raw['mdm']).round(3).to_dict()}")
print(f"  aligned gap: {(ali['ts_lda'] - ali['mdm']).round(3).to_dict()}")

mdm = sweep[sweep.model == "mdm"]
print("\nPredictors of benefit, over all individual runs in this sweep:")
for name in ("spd_dimension", "median_condition_number", "inter_subject_distance"):
    print(f"  {name:26s} {rho(mdm[name], mdm.gain)}")
print(
    f"  displacement | dimension   "
    f"{partial_rho(mdm.gain, mdm.inter_subject_distance, mdm.spd_dimension)}"
)
print("\nCollinearity among predictors within this sweep:")
print(
    f"  dimension vs conditioning  {rho(mdm.spd_dimension, mdm.median_condition_number)}"
)
print(
    f"  dimension vs displacement  {rho(mdm.spd_dimension, mdm.inter_subject_distance)}"
)

# --------------------------------------------------------------------- #
banner("4.3  COVARIANCE ESTIMATOR  (from C_estimator.csv)")

est = load("C_estimator.csv")
est = est[est.model == "mdm"]
print(
    est.pivot_table(
        index="n_channels",
        columns="estimator",
        values=[
            "median_condition_number",
            "inter_subject_distance",
            "kappa_raw",
            "kappa_aligned",
            "gain",
        ],
    )
    .round(3)
    .to_string()
)
for n in sorted(est.n_channels.unique()):
    cell = est[est.n_channels == n]
    try:
        s = cell[cell.estimator == "scm"].mean(numeric_only=True)
        o = cell[cell.estimator == "oas"].mean(numeric_only=True)
    except KeyError:
        continue
    print(f"\n  {n} channels (dimension identical):")
    print(
        f"    conditioning ratio scm/oas = "
        f"{s.median_condition_number / o.median_condition_number:.1f}x"
    )
    print(
        f"    displacement differ by "
        f"{(s.inter_subject_distance - o.inter_subject_distance) / o.inter_subject_distance * 100:+.1f}%"
    )
    print(
        f"    benefit      differ by "
        f"{(s.gain - o.gain) / o.gain * 100:+.1f}%   "
        f"(scm {s.gain:.3f} vs oas {o.gain:.3f})"
    )

# --------------------------------------------------------------------- #
banner("4.4  FREQUENCY BAND  (from A_gain_*.csv, all bands)")

print("\nTable 4 -- means over subsets")
t4 = (
    gain[gain.model == "mdm"]
    .pivot_table(
        index=["band", "n_channels"],
        values=[
            "median_condition_number",
            "inter_subject_distance",
            "kappa_raw",
            "gain",
        ],
    )
    .round(3)
)
t4["gain_ts"] = (
    gain[gain.model == "ts_lda"]
    .pivot_table(index=["band", "n_channels"], values="gain")
    .round(3)
)
print(t4.to_string())

for model in ("mdm", "ts_lda"):
    sub = gain[gain.model == model]
    print(f"\nPooled over all {len(sub)} configurations -- {model}:")
    for name in ("spd_dimension", "median_condition_number", "inter_subject_distance"):
        print(f"  {name:26s} {rho(sub[name], sub.gain)}")
    print(
        f"  displacement | dimension   "
        f"{partial_rho(sub.gain, sub.inter_subject_distance, sub.spd_dimension)}"
    )

    for n in sorted(sub.n_channels.unique()):
        cell = (
            sub[sub.n_channels == n]
            .groupby("band")[
                ["inter_subject_distance", "median_condition_number", "gain"]
            ]
            .mean()
        )
        print(
            f"  {n:2d} ch, across bands: "
            f"displacement {rho(cell.inter_subject_distance, cell.gain)} | "
            f"conditioning {rho(cell.median_condition_number, cell.gain)}"
        )

print("\nDecisive contrast, beta vs mu+beta, at each channel count:")
m = gain[gain.model == "mdm"]
for n in sorted(m.n_channels.unique()):
    cell = m[m.n_channels == n].groupby("band").mean(numeric_only=True)
    if not {"beta", "mu_beta"} <= set(cell.index):
        continue
    b, mb = cell.loc["beta"], cell.loc["mu_beta"]
    print(
        f"  {n:2d} ch:  conditioning "
        f"{(mb.median_condition_number - b.median_condition_number) / b.median_condition_number * 100:+5.1f}%   "
        f"displacement {(mb.inter_subject_distance - b.inter_subject_distance) / b.inter_subject_distance * 100:+5.1f}%   "
        f"benefit {(mb.gain - b.gain) / b.gain * 100:+5.1f}%"
    )

print("\nTheta check (benefit undefined where raw kappa <= 0):")
theta = (
    m[m.band == "theta"]
    .groupby("n_channels")[["median_condition_number", "kappa_raw", "gain"]]
    .mean()
    .round(3)
)
print(theta.to_string())

# --------------------------------------------------------------------- #
banner("4.5  RESIDUAL ROTATION  (from B_rotation.csv)")

#  B2 supersedes B_rotation.csv: three subsets per condition instead of one,
#  and the benefit computed on the same subsets as the rotation.
rot = load("B2_rotation_*.csv")
cols = [
    c
    for c in (
        "band",
        "n_channels",
        "displacement",
        "after_total",
        "after_rotational",
        "after_spectral",
        "after_rotational_fraction",
        "before_total",
    )
    if c in rot
]
print(rot[cols].round(3).to_string(index=False))

print(
    f"\n  rotational ~ displacement, pooled      {rho(rot.displacement, rot.after_rotational)}"
)
print(
    f"  rotational ~ displacement | n_channels {partial_rho(rot.after_rotational, rot.displacement, rot.n_channels)}"
)

#  B2 carries the benefit alongside the rotation, so no merge is needed and
#  the two quantities are guaranteed to come from the same electrode subset.
keep = rot[rot.band != "theta"]
print(f"\n  conditions: {len(rot)} (all), {len(keep)} (excluding theta)")
for model in ("mdm", "ts_lda"):
    column = f"{model}_gain"
    print(
        f"  {model:7s} gain ~ residual rotation: "
        f"all {rho(rot.after_rotational, rot[column])} | "
        f"no-theta {rho(keep.after_rotational, keep[column])}"
    )


# --------------------------------------------------------------------- #
banner("4.6  CLASSIFIER ASYMMETRY  (band conditions)")

no_theta = rot[rot.band != "theta"]
for model in ("mdm", "ts_lda"):
    column = f"{model}_gain"
    print(
        f"  {model:7s} gain ~ displacement: "
        f"all {rho(rot.displacement, rot[column])} | "
        f"no-theta {rho(no_theta.displacement, no_theta[column])}"
    )

print("\n  Ordering by displacement at the higher channel count:")
high = (
    rot[rot.n_channels == rot.n_channels.max()]
    .groupby("band")[["displacement", "mdm_gain", "ts_lda_gain"]]
    .mean()
    .sort_values("displacement")
)
print(high.round(3).to_string())


print("\n" + "=" * 72)
print("Check every figure in Section 4 against the output above.")
print("=" * 72)
