"""Decompose the discrepancy that survives re-centring into rotational and spectral parts.

The mechanism under test
------------------------
Re-centring whitens each subject by its own geometric mean, sending every
subject's mean covariance to the identity. It does not make two subjects'
data identical: if one subject's covariances are another's under a congruence,
whitening each by its own mean leaves them related by an *orthogonal*
transform, because any two whitenings of the same matrix differ by a rotation.

That residual rotation is invisible to geodesic distances, which are
congruence-invariant, and visible to tangent-space coordinates, which are not.
It is the proposed explanation for why nearest-centroid classifiers gain far
more from alignment than discriminative ones. This module measures it instead
of inferring it.

The decomposition
-----------------
For two SPD matrices, the smallest achievable affine-invariant distance over
all rotations of one of them is the distance between their sorted
log-eigenvalue spectra:

    min_Q d(Q A Q^T, B) = || log lambda(A) - log lambda(B) ||_2

Verified numerically against direct optimisation over SO(n), agreeing to
around 1e-3 with the numerical minimum always slightly above the bound, as it
must be. The spectral term therefore captures everything a rotation cannot
remove, and

    rotational^2 = d_AIRM^2 - spectral^2

is the part of the discrepancy that a rotation would remove. Note this is a
property of the pair of matrices, not an estimate: no optimisation is run and
no labels are used to obtain it.

Why class-conditional means
---------------------------
After alignment every subject's overall mean is exactly the identity, so
comparing subject means measures nothing. The residual structure lives in the
class-conditional means, which is also where a classifier's decision
boundaries come from.
"""

from __future__ import annotations

import numpy as np

from geoq.geometry.riemannian import distance_airm, frechet_mean

__all__ = ["decompose_pair", "residual_structure"]


def decompose_pair(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Split the distance between two SPD matrices into its two components.

    Args:
        a: SPD matrix of shape ``(n, n)``.
        b: SPD matrix of shape ``(n, n)``.

    Returns:
        Mapping with the total distance, the spectral (rotation-irreducible)
        part, the rotational part, and the fraction of squared distance that
        is rotational.
    """
    total = float(distance_airm(a, b))
    spectral = float(
        np.sqrt(
            np.sum((np.log(np.linalg.eigvalsh(a)) - np.log(np.linalg.eigvalsh(b))) ** 2)
        )
    )
    #  Clipped at zero: the spectral term is a lower bound on the total, so a
    #  negative radicand is round-off near coincident matrices rather than a
    #  meaningful quantity.
    rotational = float(np.sqrt(max(total**2 - spectral**2, 0.0)))
    return {
        "total": total,
        "spectral": spectral,
        "rotational": rotational,
        "rotational_fraction": (0.0 if total == 0 else float(rotational**2 / total**2)),
    }


def residual_structure(
    covariances: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    metric: str = "airm",
) -> dict[str, float]:
    """Measure the between-subject discrepancy remaining in class means.

    Applied to aligned covariances, this quantifies what re-centring left
    behind. Applied to unaligned ones, it gives the baseline for comparison.

    Args:
        covariances: SPD stack of shape ``(n_trials, n, n)``.
        labels: Class label per trial.
        subjects: Subject identifier per trial.
        metric: Geometry used for the class means.

    Returns:
        Mean total, spectral and rotational discrepancy across all
        subject pairs and classes, plus the rotational fraction.
    """
    class_means: dict[tuple, np.ndarray] = {}
    for subject in np.unique(subjects):
        for label in np.unique(labels):
            mask = (subjects == subject) & (labels == label)
            if mask.sum() >= 2:
                class_means[(subject, label)] = frechet_mean(
                    covariances[mask], metric=metric
                )

    records = []
    for label in np.unique(labels):
        present = sorted(s for s in np.unique(subjects) if (s, label) in class_means)
        for index, first in enumerate(present):
            for second in present[index + 1 :]:
                records.append(
                    decompose_pair(
                        class_means[(first, label)], class_means[(second, label)]
                    )
                )

    if not records:
        raise ValueError(
            "No subject pair had at least two trials in a shared class, so no "
            "class-conditional means could be compared."
        )

    return {
        "n_pairs": float(len(records)),
        "total": float(np.mean([r["total"] for r in records])),
        "spectral": float(np.mean([r["spectral"] for r in records])),
        "rotational": float(np.mean([r["rotational"] for r in records])),
        "rotational_fraction": float(
            np.mean([r["rotational_fraction"] for r in records])
        ),
    }
