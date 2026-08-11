"""Publication figures for the alignment-benefit paper.

Design choices
--------------
Every figure is drawn from the archived CSVs rather than from values typed
into the script, so a figure cannot drift from the table it accompanies. This
matters here specifically: an earlier draft of the Results section quoted two
numbers from the wrong experiment, and a figure built from retyped values
would have reproduced that error in a form nobody re-checks.

Vector PDF output, no rasterisation, so the figures survive journal
typesetting at any scale. Colours are distinguishable in greyscale and to the
common forms of colour vision deficiency: the two classifiers differ in marker
shape and line style as well as hue, so the distinction does not depend on
colour being reproduced.

Axis limits are not padded to make trends look steeper, and no regression line
is drawn where the reported association is a rank correlation, since a fitted
line would imply a linear model the statistics do not assume.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper1"
)
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

#  Elsevier single-column is 90 mm, double-column 190 mm.
SINGLE, DOUBLE = 3.54, 7.48

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed as TrueType, not Type 3
        "ps.fonttype": 42,
    }
)

MDM_STYLE = {"color": "#0072B2", "marker": "o", "linestyle": "-"}
TS_STYLE = {"color": "#D55E00", "marker": "s", "linestyle": "--"}


def load(pattern: str) -> pd.DataFrame:
    """Concatenate every CSV matching a glob, or exit with a clear message."""
    files = sorted(RESULTS.glob(pattern))
    if not files:
        sys.exit(f"No files matching {pattern} under {RESULTS}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def annotate_rho(ax, x, y, position, style) -> None:
    """Print a Spearman coefficient in the corner, in the series' colour."""
    r, p = stats.spearmanr(np.asarray(x, float), np.asarray(y, float))
    text = f"$\\rho$ = {r:+.2f}, " + ("$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}")
    ax.text(
        *position,
        text,
        transform=ax.transAxes,
        fontsize=7,
        color=style["color"],
        va="top",
        ha="left",
    )


# --------------------------------------------------------------------- #
def figure_1() -> None:
    """Benefit against displacement, both classifiers, band conditions.

    The paper's central result. The two classifiers are plotted on shared
    axes because the claim is about the *difference in sign* between them,
    which separate panels would obscure.
    """
    rot = load("B2_rotation_*.csv")
    keep = rot[rot.band != "theta"]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))

    for column, label, style in (
        ("mdm_gain", "MDM", MDM_STYLE),
        ("ts_lda_gain", "TS+LDA", TS_STYLE),
    ):
        ax.scatter(
            keep.displacement,
            keep[column],
            s=14,
            alpha=0.75,
            edgecolors="none",
            label=label,
            color=style["color"],
            marker=style["marker"],
        )
        #  Theta is shown hollow: reported for completeness, excluded from
        #  the statistics for the reason given in Section 3.7.
        theta = rot[rot.band == "theta"]
        ax.scatter(
            theta.displacement,
            theta[column],
            s=14,
            facecolors="none",
            linewidths=0.6,
            edgecolors=style["color"],
            marker=style["marker"],
        )

    #  Headroom reserved above the data so a coefficient never sits on a
    #  marker. Set before annotating, since the text is placed in axes
    #  coordinates.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.30 * (high - low))

    annotate_rho(ax, keep.displacement, keep.mdm_gain, (0.03, 0.98), MDM_STYLE)
    annotate_rho(ax, keep.displacement, keep.ts_lda_gain, (0.03, 0.89), TS_STYLE)

    ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_xlabel("Inter-subject Fréchet distance")
    ax.set_ylabel(r"Alignment benefit, $\Delta\kappa$")
    ax.legend(loc="upper right", handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(FIGURES / "fig1_benefit_vs_displacement.pdf")
    plt.close(fig)
    print("fig1 written")


def figure_2() -> None:
    """Performance against channel count, aligned and unaligned.

    Two panels because the comparison is within-classifier (aligned against
    unaligned), not between classifiers.
    """
    sweep = load("E_channel_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.72, 2.7), sharey=True)

    for ax, model, name, style in (
        (axes[0], "mdm", "MDM", MDM_STYLE),
        (axes[1], "ts_lda", "TS+LDA", TS_STYLE),
    ):
        sub = sweep[sweep.model == model]
        annotations: list[str] = []
        grouped = sub.groupby("n_channels")
        counts = np.asarray(sorted(sub.n_channels.unique()), float)

        for column, marker, fill, label in (
            ("kappa_raw", style["marker"], "none", "unaligned"),
            ("kappa_aligned", style["marker"], style["color"], "aligned"),
        ):
            mean = grouped[column].mean()
            sd = grouped[column].std()
            ax.errorbar(
                counts,
                mean,
                yerr=sd,
                capsize=1.8,
                elinewidth=0.6,
                color=style["color"],
                marker=marker,
                markersize=4,
                markerfacecolor=fill,
                markeredgewidth=0.8,
                linestyle="-" if fill != "none" else ":",
                label=label,
            )
            r, p = stats.spearmanr(counts, mean)
            annotations.append(
                f"{label}: $\\rho$ = {r:+.2f}, "
                + ("$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}")
            )

        #  Headroom first, then the text, so a long label cannot run off
        #  the top-right corner as it did with the axis left autoscaled.
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.34 * (high - low))
        for offset, text in enumerate(annotations):
            ax.text(
                0.04,
                0.97 - 0.09 * offset,
                text,
                transform=ax.transAxes,
                fontsize=6.5,
                color=style["color"],
                va="top",
            )

        ax.set_title(name)
        ax.set_xlabel("Number of electrodes")
        ax.set_xticks([3, 5, 8, 12, 16, 22])
        ax.legend(loc="lower right", handletextpad=0.3)

    axes[0].set_ylabel(r"Cross-subject $\kappa$")
    fig.savefig(FIGURES / "fig2_channel_count.pdf")
    plt.close(fig)
    print("fig2 written")


def figure_3() -> None:
    """Decomposition of the residual discrepancy after re-centring.

    Left: the rotational component against displacement, which is the
    mechanism's own prediction. Right: the two components stacked by band,
    showing that the rotational share is largest where no class structure
    exists.
    """
    rot = load("B2_rotation_*.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.78, 2.7))

    ax = axes[0]
    for count, marker, fill in ((8, "o", "none"), (18, "o", "#0072B2")):
        sub = rot[rot.n_channels == count]
        ax.scatter(
            sub.displacement,
            sub.after_rotational,
            s=16,
            marker=marker,
            facecolors=fill,
            edgecolors="#0072B2",
            linewidths=0.7,
            label=f"{count} electrodes",
        )
    r, p = stats.spearmanr(rot.displacement, rot.after_rotational)
    ax.text(
        0.04,
        0.96,
        f"$\\rho$ = {r:+.2f}, " + ("$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
    )
    ax.set_xlabel("Inter-subject Fréchet distance")
    ax.set_ylabel("Residual rotational component")
    ax.legend(loc="lower right", handletextpad=0.3)

    ax = axes[1]
    order = (
        rot[rot.n_channels == 18]
        .groupby("band")
        .displacement.mean()
        .sort_values()
        .index
    )
    means = (
        rot[rot.n_channels == 18]
        .groupby("band")[["after_rotational", "after_spectral"]]
        .mean()
        .loc[order]
    )
    positions = np.arange(len(order))
    ax.bar(
        positions,
        means.after_rotational,
        0.62,
        label="rotational",
        color="#0072B2",
        edgecolor="none",
    )
    ax.bar(
        positions,
        means.after_spectral,
        0.62,
        bottom=means.after_rotational,
        label="spectral",
        color="#56B4E9",
        edgecolor="none",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([b.replace("_", "+") for b in order], rotation=30, ha="right")
    ax.set_ylabel("Residual discrepancy")
    ax.set_xlabel("Band, ordered by displacement")
    ax.legend(loc="upper left", handletextpad=0.4)

    fig.savefig(FIGURES / "fig3_rotation_decomposition.pdf")
    plt.close(fig)
    print("fig3 written")


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    print(f"\nfigures in {FIGURES}")
