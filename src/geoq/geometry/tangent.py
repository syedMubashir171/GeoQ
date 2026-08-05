"""Tangent-space representation of SPD matrices.

The tangent space at a reference point is a genuine Euclidean vector space.
Mapping covariance matrices into it is what allows ordinary linear
classifiers -- LDA, SVM, logistic regression -- to be applied to data that
lives on a curved manifold. It is the basis of the Tangent Space + LDA
baseline that this thesis must beat, or honestly fail to beat.

The whitening step is not optional
----------------------------------
This module maps ``X`` to ``vec(log(P^-1/2 X P^-1/2))``, not to
``vec(Log_P(X))``. The two differ by a congruence, and the difference is the
whole point.

The Riemannian inner product at ``P`` is
``<U, V>_P = tr(P^-1 U P^-1 V)``, which is *not* the Euclidean inner product
of the raw tangent matrices. Whitening by ``P^-1/2`` transports the metric to
the identity, where the Riemannian inner product coincides exactly with the
Frobenius one. Only after that does Euclidean distance in the feature vector
mean Riemannian distance on the manifold, and only then is a linear
classifier operating on the geometry rather than on an arbitrary coordinate
chart.

Concretely: ``||tangent_space(X, P)||_2 == distance_airm(P, X)``. That
identity is the contract of this module, and it fails if whitening is skipped.

Vectorisation convention
------------------------
A symmetric ``n x n`` matrix has ``n (n + 1) / 2`` free entries. They are
taken in row-major order over the upper triangle including the diagonal, with
off-diagonal entries scaled by ``sqrt(2)`` so that the Euclidean norm of the
vector equals the Frobenius norm of the matrix. Without that factor each
off-diagonal entry would be counted once instead of twice and the isometry
would be lost.

This matches pyRiemann, which matters: it is the difference between
reproducing a published tangent-space result and quietly reporting different
numbers. The convention is pinned down by
``tests/regression/test_pyriemann_parity.py``.

References
----------
Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). Multiclass
    brain-computer interface classification by Riemannian geometry.
    *IEEE TBME*, 59(4), 920-928.
Tuzel, O., Porikli, F., & Meer, P. (2008). Pedestrian detection via
    classification on Riemannian manifolds. *IEEE TPAMI*, 30(10), 1713-1727.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from geoq.geometry.riemannian import Metric, whiten
from geoq.geometry.spd import (
    FloatArray,
    check_spd,
    check_symmetric,
    expm_sym,
    logm_spd,
    sqrtm_spd,
    symmetrize,
)

__all__ = [
    "OFF_DIAGONAL_SCALE",
    "matrix_dimension",
    "tangent_space",
    "untangent_space",
    "unvectorize",
    "vector_dimension",
    "vectorize",
]

logger = logging.getLogger(__name__)

OFF_DIAGONAL_SCALE: Final[float] = float(np.sqrt(2.0))
"""Scaling applied to off-diagonal entries so vectorisation is an isometry."""


# --------------------------------------------------------------------------- #
# Dimension arithmetic
# --------------------------------------------------------------------------- #


def vector_dimension(n: int) -> int:
    """Length of the vectorised form of an ``n x n`` symmetric matrix.

    Args:
        n: Matrix dimension.

    Returns:
        ``n * (n + 1) // 2``.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    return n * (n + 1) // 2


def matrix_dimension(d: int) -> int:
    """Inverse of :func:`vector_dimension`.

    Args:
        d: Vector length.

    Returns:
        The matrix dimension ``n`` with ``n (n + 1) / 2 == d``.

    Raises:
        ValueError: If no integer ``n`` satisfies the relation. This catches a
            feature matrix that has been sliced, padded, or concatenated with
            something else before reaching an inverse transform -- a failure
            that is otherwise silent until a reshape produces nonsense.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}.")
    n = round((np.sqrt(8 * d + 1) - 1) / 2)
    if vector_dimension(n) != d:
        raise ValueError(
            f"Vector length {d} does not correspond to any symmetric matrix; "
            f"valid lengths are n(n+1)/2 for integer n "
            f"(1, 3, 6, 10, 15, ...). The nearest valid length is "
            f"{vector_dimension(n)}."
        )
    return n


def _triangular_indices(n: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Row and column indices of the upper triangle, including the diagonal."""
    return np.triu_indices(n)


def _scaling(n: int) -> FloatArray:
    """Per-entry scaling vector: 1 on the diagonal, ``sqrt(2)`` off it."""
    rows, cols = _triangular_indices(n)
    return np.where(rows == cols, 1.0, OFF_DIAGONAL_SCALE)


# --------------------------------------------------------------------------- #
# Vectorisation
# --------------------------------------------------------------------------- #


def vectorize(s: ArrayLike, *, validate: bool = True) -> FloatArray:
    """Flatten symmetric matrices to vectors, preserving the Frobenius norm.

    Args:
        s: Symmetric stack of shape ``(..., n, n)``. Positive definiteness is
            not required: tangent vectors are displacements and are routinely
            indefinite.
        validate: Whether to enforce symmetry.

    Returns:
        Array of shape ``(..., n (n + 1) / 2)``.
    """
    arr = check_symmetric(s, name="s") if validate else symmetrize(s)
    n = arr.shape[-1]
    rows, cols = _triangular_indices(n)
    return arr[..., rows, cols] * _scaling(n)


def unvectorize(v: ArrayLike) -> FloatArray:
    """Rebuild symmetric matrices from their vectorised form.

    Exact inverse of :func:`vectorize`, including the off-diagonal scaling.

    Args:
        v: Array of shape ``(..., n (n + 1) / 2)``.

    Returns:
        Symmetric stack of shape ``(..., n, n)``.

    Raises:
        ValueError: If the trailing dimension is not a valid vector length.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError(f"v must have at least 1 dimension, got shape {arr.shape}.")
    n = matrix_dimension(arr.shape[-1])
    rows, cols = _triangular_indices(n)

    result = np.zeros((*arr.shape[:-1], n, n), dtype=np.float64)
    entries = arr / _scaling(n)
    result[..., rows, cols] = entries
    result[..., cols, rows] = entries
    return result


# --------------------------------------------------------------------------- #
# The tangent map
# --------------------------------------------------------------------------- #


def tangent_space(
    x: ArrayLike,
    reference: ArrayLike,
    *,
    metric: Metric = "airm",
    validate: bool = True,
) -> FloatArray:
    """Project SPD matrices into the tangent space at ``reference``.

    Computes ``vec(log(P^-1/2 X P^-1/2))`` for the affine-invariant metric.
    The resulting vectors are isometric to the manifold: the Euclidean norm of
    a feature vector equals the geodesic distance from ``reference`` to the
    matrix it came from.

    The reference point must be estimated on training data only. This function
    takes it as an explicit argument rather than computing it internally,
    which is deliberate: a transformer that silently derived its own reference
    from whatever array it was handed would leak test-fold information into
    training every time it was called on a full dataset. Making the reference
    a required input forces the caller to decide where it came from, and
    :mod:`geoq.features` is where that decision is enforced structurally.

    Args:
        x: SPD stack of shape ``(..., n, n)``.
        reference: SPD matrix of shape ``(n, n)``, the base point.
        metric: ``"airm"`` for the whitened logarithm, or ``"logeuclid"`` for
            ``log(X) - log(P)``. The latter is cheaper and needs no matrix
            square root, but is not an isometry of the affine-invariant
            geometry.
        validate: Whether to enforce the SPD contract.

    Returns:
        Array of shape ``(..., n (n + 1) / 2)``.

    Raises:
        ValueError: If ``metric`` is unsupported or ``reference`` is not a
            single matrix.
    """
    if validate:
        x = check_spd(x, name="x")
        reference = check_spd(reference, name="reference")
    reference_arr = np.asarray(reference, dtype=np.float64)
    if reference_arr.ndim != 2:
        raise ValueError(
            f"reference must be a single matrix of shape (n, n), got "
            f"{reference_arr.shape}. A per-sample reference would make the "
            f"feature space sample-dependent and its coordinates meaningless."
        )

    if metric == "airm":
        inner = logm_spd(whiten(reference_arr, x, validate=False), validate=False)
    elif metric == "logeuclid":
        inner = logm_spd(x, validate=False) - logm_spd(reference_arr, validate=False)
    else:
        raise ValueError(
            f"Unsupported metric {metric!r} for the tangent map. Use 'airm' "
            f"or 'logeuclid'; 'stein' and 'euclid' have no tangent-space "
            f"formulation here."
        )
    return vectorize(inner, validate=False)


def untangent_space(
    v: ArrayLike,
    reference: ArrayLike,
    *,
    metric: Metric = "airm",
    validate: bool = True,
) -> FloatArray:
    """Map tangent vectors back onto the manifold.

    Inverse of :func:`tangent_space`. Needed wherever a result computed in the
    tangent space must be interpreted as a covariance again: visualising a
    class centroid, inspecting a classifier's decision direction as a spatial
    filter, or generating synthetic trials along a geodesic.

    Args:
        v: Array of shape ``(..., n (n + 1) / 2)``.
        reference: SPD matrix of shape ``(n, n)`` used for the forward map.
        metric: Must match the metric used in the forward map.
        validate: Whether to enforce the SPD contract on ``reference``.

    Returns:
        SPD stack of shape ``(..., n, n)``.

    Raises:
        ValueError: If ``metric`` is unsupported.
    """
    reference_arr = (
        check_spd(reference, name="reference")
        if validate
        else np.asarray(reference, dtype=np.float64)
    )
    inner = unvectorize(v)

    if metric == "airm":
        root = sqrtm_spd(reference_arr, validate=False)
        return symmetrize(root @ expm_sym(inner, validate=False) @ root)
    if metric == "logeuclid":
        return expm_sym(inner + logm_spd(reference_arr, validate=False), validate=False)
    raise ValueError(
        f"Unsupported metric {metric!r} for the inverse tangent map. Use "
        f"'airm' or 'logeuclid'."
    )
