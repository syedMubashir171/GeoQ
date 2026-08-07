"""Shared fixtures and numerical helpers for the GeoQ test suite.

Everything here is deterministic. No test anywhere in this repository may draw
from an unseeded generator: a suite that fails once every fifty runs is worse
than no suite, because it trains you to re-run until green.

Tolerance policy
----------------
Absolute tolerances are never compared against a matrix's raw magnitude. EEG
covariance entries span roughly ``1e-14`` to ``1e-8`` depending only on whether
the recording is stored in volts or microvolts, so a fixed epsilon would make
a test's verdict depend on the amplifier. Errors are judged relative to the
norm of the reference, and where a bound must be stated it is stated as a
multiple of ``eps * kappa``, the textbook error bound for a spectral matrix
function.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from geoq.testing import (
    DIMENSIONS,
    EPS,
    assert_exactly_symmetric,
    assert_spd,
    relative_error,
    spectral_error_bound,
)

__all__ = [
    "DIMENSIONS",
    "EPS",
    "assert_exactly_symmetric",
    "assert_spd",
    "relative_error",
    "spectral_error_bound",
]

MASTER_SEED: int = 20260804


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator for the test requesting it.

    Returns:
        A fresh :class:`numpy.random.Generator`. Fresh per test, so execution
        order and ``-k`` filtering cannot change any test's outcome.
    """
    return np.random.default_rng(MASTER_SEED)


@pytest.fixture
def rng_pair() -> tuple[np.random.Generator, np.random.Generator]:
    """Two independent generators derived from the master seed.

    Needed wherever a test draws two sample sets that must not be correlated,
    such as comparing distances between two independent groups of matrices.

    Returns:
        A pair of independent generators.
    """
    first, second = np.random.SeedSequence(MASTER_SEED).spawn(2)
    return np.random.default_rng(first), np.random.default_rng(second)


#  Optional dependencies are skipped, never failed. The geometry layer must
#  stay testable on a bare `pip install -e ".[dev]"` with no quantum or EEG
#  stack present -- that is the property the layered extras in pyproject.toml
#  exist to guarantee, and these fixtures are what enforce it.


@pytest.fixture(scope="session")
def pyriemann():
    """The pyRiemann module, or skip the test if it is not installed."""
    return pytest.importorskip(
        "pyriemann", reason="requires the 'eeg' extra: pip install -e '.[eeg]'"
    )


@pytest.fixture(scope="session")
def pennylane():
    """The PennyLane module, or skip the test if it is not installed."""
    return pytest.importorskip(
        "pennylane", reason="requires the 'quantum' extra: pip install -e '.[quantum]'"
    )


#  Hypothesis profiles.
#
#  The seed is left random on purpose: a fixed seed would make the property
#  tests deterministic and therefore blind to exactly the rare inputs they
#  exist to find. Three of this framework's numerical bugs were discovered by
#  a fresh Hypothesis draw finding a case an earlier run had missed.
#
#  What the ci profile actually changes: print_blob, which makes a failing
#  example reproducible from the CI log with a single paste. That is the point
#  of it -- a property-test failure on a runner you cannot attach to is
#  otherwise painful to reproduce.
#
#  max_examples applies only to property tests that do not carry their own
#  @settings decorator. Most in this suite do, so raising it here is a floor
#  rather than a global increase; the timings barely move. Stated because a
#  comment claiming CI runs more examples than it does would be worse than no
#  comment.
#
#  An unknown profile name raises InvalidArgument at collection rather than
#  falling back to a default, so a typo in the workflow fails visibly.
settings.register_profile(
    "dev",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=300,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
