"""Kernel spectra, bandwidth, and the drift toward the identity.

Block 5.5 deliverable
---------------------
Two things the syllabus asks for. First, compute a quantum kernel matrix,
compare its spectrum to that of an RBF kernel, and show how rescaling the
bandwidth changes the condition number. Second, reproduce kernel drift: a
fixed-bandwidth quantum kernel on high-dimensional data approaches the
identity, and separability is destroyed.

Why this block exists
---------------------
The quantum kernel experiment on EEG found the entangled fidelity kernel at
chance across every qubit count tested, and the scale that an unsupervised
search selected fell as the circuit widened. Both observations are the same
phenomenon, and the kernel literature names it: bandwidth governs whether a
quantum kernel generalises at all, and an untuned kernel drifts toward the
identity as dimension grows (Shaydulin and Wild, 2022; Canatar et al.,
2022).

What the term means here
------------------------
Bandwidth is the multiplier applied to features before encoding, which is
the ``scale`` parameter elsewhere in this project. The two words denote the
same quantity. Using the literature's term here makes the connection to the
published analysis explicit rather than leaving a reader to infer it.

The diagnostic quantities
-------------------------
A kernel matrix is read through its spectrum. The condition number, the
ratio of largest to smallest eigenvalue, states how close the matrix is to
singular. Effective rank, the exponential of the entropy of the normalised
eigenvalues, states how many directions the kernel actually uses: a kernel
that has drifted to the identity has full rank and maximal effective rank
while carrying no information, so rank alone is not enough and the two are
reported together with the off-diagonal spread.

References
----------
Schuld, M. (2021). Supervised quantum machine learning models are kernel
    methods. arXiv:2101.11020.
Shaydulin, R., & Wild, S. M. (2022). Importance of kernel bandwidth in
    quantum machine learning. *Physical Review A*, 106(4), 042407.
Canatar, A. et al. (2022). Bandwidth enables generalization in quantum
    kernel models. *TMLR*.
Huang, H.-Y. et al. (2021). Power of data in quantum machine learning.
    *Nature Communications*, 12(1), 2631.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from geoq.models.quantum.kernel import fidelity_kernel

RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/block5"
)
RESULTS.mkdir(parents=True, exist_ok=True)

BANDWIDTHS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
N_SAMPLES = 48


def effective_rank(gram: np.ndarray) -> float:
    """Number of directions the kernel meaningfully uses.

    Defined as the exponential of the Shannon entropy of the eigenvalues
    normalised to sum to one. A kernel using every direction equally has
    effective rank equal to its size; one dominated by a single direction
    has effective rank near one.

    Args:
        gram: A symmetric positive semi-definite matrix.

    Returns:
        The effective rank.
    """
    values = np.linalg.eigvalsh(gram)
    values = values[values > 1e-12]
    if values.size == 0:
        return 0.0
    weights = values / values.sum()
    return float(np.exp(-np.sum(weights * np.log(weights))))


def describe(gram: np.ndarray) -> dict:
    """Summarise a kernel matrix by spectrum and off-diagonal spread.

    Args:
        gram: The kernel matrix.

    Returns:
        Condition number, effective rank, and the mean and standard
        deviation of the off-diagonal entries.
    """
    values = np.linalg.eigvalsh(gram)
    off = gram[~np.eye(gram.shape[0], dtype=bool)]
    smallest = max(float(values.min()), 1e-15)
    return {
        "condition_number": float(values.max()) / smallest,
        "effective_rank": effective_rank(gram),
        "off_diagonal_mean": float(off.mean()),
        "off_diagonal_std": float(off.std()),
    }


def make_data(n_features: int, n: int = N_SAMPLES, seed: int = 0):
    """A separable two-class problem in a chosen dimension.

    The signal lies in the first two coordinates only, so adding features
    adds noise dimensions. This is what makes the drift experiment
    meaningful: a classifier that fails as dimension grows is failing
    because of the kernel, not because the problem became harder.

    Args:
        n_features: Dimension of the feature space.
        n: Number of points.
        seed: Random seed.

    Returns:
        Features scaled to ``[-1, 1]`` and binary labels.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    features = rng.normal(0, 1, (n, n_features))
    features[:, 0] += 2.5 * labels
    features[:, 1] += 2.5 * labels
    return features / np.abs(features).max(), labels


def bandwidth_sweep() -> pd.DataFrame:
    """How bandwidth changes the spectrum, for quantum and RBF kernels."""
    features, labels = make_data(n_features=6)
    rows = []
    for bandwidth in BANDWIDTHS:
        quantum = fidelity_kernel(features, feature_map="zz", scale=bandwidth)
        #  The RBF kernel is given the same sweep. Its bandwidth enters as
        #  gamma rather than as a multiplier on the features, but the role
        #  is the same: it sets the length scale over which two points count
        #  as similar, and the comparison is meaningless unless both are
        #  varied.
        classical = rbf_kernel(features, gamma=bandwidth**2)

        for name, gram in (("quantum_zz", quantum), ("rbf", classical)):
            accuracy = cross_val_score(
                SVC(kernel="precomputed"), gram, labels, cv=4
            ).mean()
            rows.append(
                {
                    "kernel": name,
                    "bandwidth": bandwidth,
                    "accuracy": float(accuracy),
                    **describe(gram),
                }
            )
    return pd.DataFrame(rows)


def drift_sweep() -> pd.DataFrame:
    """Kernel drift: fixed bandwidth, growing dimension."""
    rows = []
    for n_features in (2, 4, 6, 8, 10, 12):
        features, labels = make_data(n_features=n_features)
        for bandwidth, label in ((1.5, "fixed"), (None, "tuned")):
            if bandwidth is None:
                #  Tuning selects the bandwidth maximising off-diagonal
                #  spread, which uses no labels and is the same unsupervised
                #  rule used in the EEG experiment.
                best, best_spread = BANDWIDTHS[0], -np.inf
                for candidate in BANDWIDTHS:
                    gram = fidelity_kernel(features, feature_map="zz", scale=candidate)
                    spread = describe(gram)["off_diagonal_std"]
                    if spread > best_spread:
                        best, best_spread = candidate, spread
                bandwidth = best
            gram = fidelity_kernel(features, feature_map="zz", scale=bandwidth)
            accuracy = cross_val_score(
                SVC(kernel="precomputed"), gram, labels, cv=4
            ).mean()
            rows.append(
                {
                    "regime": label,
                    "n_features": n_features,
                    "bandwidth": bandwidth,
                    "accuracy": float(accuracy),
                    **describe(gram),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run both experiments and report them."""
    print("=" * 72)
    print("Bandwidth and the kernel spectrum")
    print("=" * 72)
    sweep = bandwidth_sweep()
    sweep.to_csv(RESULTS / "kernel_bandwidth.csv", index=False)
    print(
        sweep.pivot(
            index="bandwidth",
            columns="kernel",
            values=["condition_number", "off_diagonal_std", "accuracy"],
        )
        .round(3)
        .to_string()
    )
    print("\n  At small bandwidth every pair looks identical and the matrix")
    print("  is near-singular. At large bandwidth it approaches the identity")
    print("  and the condition number returns to one, but the kernel now")
    print("  relates nothing to anything. Both extremes classify at chance.")

    print("\n" + "=" * 72)
    print("Kernel drift: fixed bandwidth against growing dimension")
    print("=" * 72)
    drift = drift_sweep()
    drift.to_csv(RESULTS / "kernel_drift.csv", index=False)
    print(
        drift.pivot(
            index="n_features",
            columns="regime",
            values=["off_diagonal_mean", "effective_rank", "accuracy"],
        )
        .round(3)
        .to_string()
    )

    fixed = drift[drift.regime == "fixed"]
    tuned = drift[drift.regime == "tuned"]
    print(
        f"\n  fixed bandwidth: off-diagonal mean falls from "
        f"{fixed.off_diagonal_mean.iloc[0]:.4f} to "
        f"{fixed.off_diagonal_mean.iloc[-1]:.4f}"
    )
    print(
        f"  its effective rank rises from "
        f"{fixed.effective_rank.iloc[0]:.1f} to "
        f"{fixed.effective_rank.iloc[-1]:.1f} of {N_SAMPLES}, which is the "
        f"signature of\n  drift toward the identity rather than of a "
        f"richer representation"
    )
    print(f"  accuracy: {fixed.accuracy.iloc[0]:.3f} -> {fixed.accuracy.iloc[-1]:.3f}")
    print(
        f"\n  tuned bandwidth: {tuned.accuracy.iloc[0]:.3f} -> "
        f"{tuned.accuracy.iloc[-1]:.3f}, with the selected value falling "
        f"from\n  {tuned.bandwidth.iloc[0]} to {tuned.bandwidth.iloc[-1]} "
        f"as dimension grows"
    )
    print("\n  This is the mechanism behind the entangled kernel sitting at")
    print("  chance on EEG. It is also why a bandwidth-tuned quantum kernel")
    print("  converges on classical behaviour: tuning removes the very")
    print("  structure that made the kernel different.")


if __name__ == "__main__":
    main()
