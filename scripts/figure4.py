"""Figure 4: the cross-dataset comparison.

Displacement ranks conditions within a dataset but does not calibrate
across them, and this figure is where that is visible.

Why this figure is necessary
Section 5.7 is the manuscript's most consequential negative result, and it
is inherently visual: two datasets whose fitted lines share a slope
direction but differ in intercept, over displacement ranges that barely
overlap. Prose can state the intercepts; only a plot shows that the datasets
occupy different regions and that the pooled correlation is therefore
dominated by the offset between them rather than by any trend within either.

The figure is drawn from F_cross_*.csv, the same files the section's numbers
come from, so it cannot drift from the text it accompanies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper1"
)
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

DOUBLE = 7.48  # Elsevier/IOP double-column width in inches

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
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

#  Okabe-Ito, distinguishable in greyscale and to the common forms of
#  colour vision deficiency. Marker shape carries the distinction as well,
#  so the figure does not depend on colour being reproduced.
STYLE = {
    "bci_iv_2a_lr": ("#0072B2", "o", "BCI IV 2a"),
    "physionet_mi": ("#D55E00", "s", "PhysioNet MI"),
    "bci_iv_2b": ("#009E73", "^", "BCI IV 2b"),
}


def main() -> None:
    """Draw both classifiers as panels of one figure."""
    files = sorted(RESULTS.glob("F_cross_*.csv"))
    if not files:
        sys.exit(f"No F_cross_*.csv under {RESULTS}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    #  Conditions where the unaligned classifier is at or below chance are
    #  excluded, as everywhere else in the paper: a benefit is undefined
    #  when there is no performance to improve.
    usable = frame[frame.kappa_raw > 0.02]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.82, 2.9), sharex=True)

    for ax, model, title in ((axes[0], "mdm", "MDM"), (axes[1], "ts_lda", "TS+LDA")):
        sub = usable[usable.model == model]
        for name, (colour, marker, label) in STYLE.items():
            cell = sub[sub.dataset == name]
            if cell.empty:
                continue
            ax.scatter(
                cell.displacement,
                cell.gain,
                s=16,
                alpha=0.8,
                edgecolors="none",
                color=colour,
                marker=marker,
                label=label,
            )
            #  A fitted line is drawn here, unlike the other figures, because
            #  the claim concerns the intercepts: the reader needs to see two
            #  lines that do not coincide, not a rank correlation.
            if len(cell) >= 4:
                slope, intercept = np.polyfit(cell.displacement, cell.gain, 1)
                span = np.linspace(cell.displacement.min(), cell.displacement.max(), 10)
                ax.plot(
                    span,
                    slope * span + intercept,
                    color=colour,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.9,
                )

        ax.axhline(0, color="0.75", linewidth=0.5, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Inter-subject Fréchet distance")

    axes[0].set_ylabel(r"Alignment benefit, $\Delta\kappa$")
    axes[0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)

    out = FIGURES / "figure4_cross_dataset.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"written: {out}")

    #  Print the fitted parameters so the caption and section 5.7 can be
    #  checked against the figure rather than against a remembered value.
    print("\nfitted lines (for checking against section 5.7):")
    for model in ("mdm", "ts_lda"):
        sub = usable[usable.model == model]
        for name in STYLE:
            cell = sub[sub.dataset == name]
            if len(cell) >= 4:
                s, i = np.polyfit(cell.displacement, cell.gain, 1)
                print(
                    f"  {model:7s} {name:14s} slope {s:+.4f}  "
                    f"intercept {i:+.4f}  range "
                    f"{cell.displacement.min():.2f}-{cell.displacement.max():.2f}"
                )


if __name__ == "__main__":
    main()
