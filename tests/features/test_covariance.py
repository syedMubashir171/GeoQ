"""Tests for :mod:`geoq.features.covariance`.

What is being defended
----------------------
Covariance estimation is where a pipeline's numerical fate is decided. Every
downstream operation -- geodesic distance, tangent projection, quantum
encoding -- inherits the conditioning of the matrices produced here, and the
geometry layer's measurements put the relative error of an AIRM computation at
roughly ``eps * kappa ** 2``.

So these tests check three things beyond "does it return the right numbers":

* **Rank is a hard boundary.** ``T <= N`` produces a singular matrix, and no
  amount of downstream care recovers from a point that is not on the manifold.
  ``TestRankRequirements`` asserts the refusal.
* **Shrinkage does what it is for.** ``TestConditioning`` measures the
  condition number under realistic spatially correlated data and asserts the
  improvement is orders of magnitude, not decorative.
* **Estimation is leakage-free by construction.** Each covariance depends on
  its own epoch alone. ``TestNoLeakage`` asserts that trial-wise independence
  directly, so a future change that pools statistics across trials fails here
  rather than inflating an accuracy figure.

Simulating EEG
--------------
White Gaussian noise is a bad model for this domain. Volume conduction makes
real EEG strongly spatially correlated, so its covariance spectrum decays
sharply and its condition number is orders of magnitude worse than white
noise's at the same epoch length. The fixture below mixes independent sources
through a decaying spatial spectrum, which reproduces the regime where
conditioning actually bites: at ``T/N = 2`` the sample covariance reaches
``kappa ~ 7e3`` under this model against ``~30`` for white noise.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the 'ml' extra: pip install -e '.[ml]'")

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import NotFittedError

from geoq.features.covariance import (
    SUPPORTED_ESTIMATORS,
    Covariances,
)
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.spd import condition_number, is_spd, random_spd

N_TRIALS = 30
N_CHANNELS = 8
N_TIMES = 512

ESTIMATORS = list(SUPPORTED_ESTIMATORS)


@pytest.fixture
def white_epochs(rng: np.random.Generator) -> np.ndarray:
    """Uncorrelated epochs: the easy case, useful for exact-value checks."""
    return rng.standard_normal((N_TRIALS, N_CHANNELS, N_TIMES))


@pytest.fixture
def eeg_like(rng: np.random.Generator):
    """A factory for spatially correlated epochs resembling real EEG.

    Independent sources are mixed through an orthogonal basis with an
    exponentially decaying spectrum, which is what volume conduction does to
    scalp potentials. The resulting covariances are far worse conditioned than
    white noise at the same epoch length, which is the regime the conditioning
    tests need.
    """

    def _make(n_trials: int, n_channels: int, n_times: int, decay: float = 3.0):
        basis, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
        spectrum = np.exp(-np.arange(n_channels) / decay)
        mixing = basis @ np.diag(np.sqrt(spectrum))
        sources = rng.standard_normal((n_trials, n_channels, n_times))
        return np.einsum("ij,bjt->bit", mixing, sources)

    return _make


# --------------------------------------------------------------------------- #
# 1. Correctness
# --------------------------------------------------------------------------- #


class TestEstimation:
    """Each estimator returns valid SPD matrices of the right shape."""

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_output_shape_and_definiteness(
        self, estimator: str, white_epochs: np.ndarray
    ) -> None:
        result = Covariances(estimator=estimator).fit_transform(white_epochs)
        assert result.shape == (N_TRIALS, N_CHANNELS, N_CHANNELS)
        assert bool(np.all(is_spd(result)))

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_output_is_exactly_symmetric(
        self, estimator: str, white_epochs: np.ndarray
    ) -> None:
        result = Covariances(estimator=estimator).fit_transform(white_epochs)
        assert np.array_equal(result, np.swapaxes(result, -1, -2))

    def test_scm_matches_the_textbook_definition(
        self, white_epochs: np.ndarray
    ) -> None:
        """Checked against an independent hand-written computation.

        Normalisation is by ``n_times``, matching scikit-learn's
        ``empirical_covariance`` and therefore pyRiemann. Recomputing it here
        from first principles rather than calling the same helper is what
        makes this a test rather than a tautology.
        """
        result = Covariances(estimator="scm").fit_transform(white_epochs)
        epoch = white_epochs[0]
        centred = epoch - epoch.mean(axis=1, keepdims=True)
        expected = centred @ centred.T / epoch.shape[1]
        assert np.allclose(result[0], expected, atol=1e-12)

    def test_assume_centered_skips_mean_removal(self, rng: np.random.Generator) -> None:
        """An unremoved mean adds a rank-one term that is not neural."""
        epochs = rng.standard_normal((5, 4, 200)) + 10.0
        centred = Covariances(estimator="scm", assume_centered=False)
        uncentred = Covariances(estimator="scm", assume_centered=True)
        # The offset is large, so the uncentred estimate is dominated by it.
        assert np.trace(uncentred.fit_transform(epochs)[0]) > 10.0 * np.trace(
            centred.fit_transform(epochs)[0]
        )

    def test_scale_equivariance(self, white_epochs: np.ndarray) -> None:
        """Scaling the signal scales the covariance quadratically."""
        base = Covariances().fit_transform(white_epochs)
        scaled = Covariances().fit_transform(white_epochs * 3.0)
        assert np.allclose(scaled, 9.0 * base, rtol=1e-10)

    def test_channel_permutation_equivariance(
        self, white_epochs: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Relabelling electrodes must permute the matrix, not change it."""
        order = rng.permutation(N_CHANNELS)
        base = Covariances().fit_transform(white_epochs)
        permuted = Covariances().fit_transform(white_epochs[:, order, :])
        assert np.allclose(permuted, base[:, order][:, :, order], atol=1e-12)

    def test_explicit_shrinkage_preserves_trace(self, white_epochs: np.ndarray) -> None:
        """Shrinkage redistributes power; it does not add any."""
        plain = Covariances(estimator="scm").fit_transform(white_epochs)
        shrunk = Covariances(estimator="scm", shrinkage=0.3).fit_transform(white_epochs)
        assert np.allclose(
            np.trace(shrunk, axis1=-2, axis2=-1),
            np.trace(plain, axis1=-2, axis2=-1),
            rtol=1e-12,
        )

    def test_explicit_shrinkage_reduces_conditioning(self, eeg_like) -> None:
        epochs = eeg_like(20, 16, 40)
        plain = Covariances(estimator="scm", audit_conditioning=False).fit_transform(
            epochs
        )
        shrunk = Covariances(
            estimator="scm", shrinkage=0.1, audit_conditioning=False
        ).fit_transform(epochs)
        assert np.median(condition_number(shrunk)) < np.median(condition_number(plain))


# --------------------------------------------------------------------------- #
# 2. Rank
# --------------------------------------------------------------------------- #


class TestRankRequirements:
    """``T > N`` is a hard boundary, not a guideline."""

    @pytest.mark.parametrize("n_times", [1, 4, 7, 8])
    def test_too_few_samples_rejected(
        self, n_times: int, rng: np.random.Generator
    ) -> None:
        """With ``T <= N`` the covariance is singular and the manifold undefined.

        Failing here, with a message naming the cause, is the only useful
        behaviour. Returning a singular matrix would push the failure into the
        matrix logarithm several layers away.
        """
        epochs = rng.standard_normal((5, 8, n_times))
        with pytest.raises(ValueError, match="singular"):
            Covariances().fit(epochs)

    def test_error_message_names_the_remedies(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError) as excinfo:
            Covariances().fit(rng.standard_normal((5, 8, 6)))
        message = str(excinfo.value)
        assert "n_times=6" in message and "n_channels=8" in message
        # Shrinkage must be explicitly ruled out: it would fabricate the
        # missing rank rather than estimate it, producing a matrix that passes
        # every downstream check and means nothing.
        assert "Shrinkage cannot repair this" in message

    def test_minimum_viable_length_accepted(self, rng: np.random.Generator) -> None:
        """``T = N + 1`` is the smallest full-rank case and must work."""
        epochs = rng.standard_normal((5, 8, 9))
        result = Covariances(audit_conditioning=False).fit_transform(epochs)
        assert bool(np.all(is_spd(result)))

    def test_flat_channel_rejected(self, white_epochs: np.ndarray) -> None:
        """A dead electrode produces a singular covariance."""
        epochs = white_epochs.copy()
        epochs[2, 5, :] = 0.0
        with pytest.raises(ValueError, match="flat channel"):
            Covariances().fit(epochs)

    def test_rank_deficient_channels_rejected_with_diagnosis(
        self, white_epochs: np.ndarray
    ) -> None:
        """Linearly dependent channels are the average-reference signature.

        Average referencing removes exactly one degree of freedom, so the
        covariance is singular despite ``T >> N`` and despite no channel being
        flat. The error must say so, because the cause is a preprocessing
        choice and not a data defect.
        """
        epochs = white_epochs.copy()
        epochs -= epochs.mean(axis=1, keepdims=True)  # average reference
        with pytest.raises(ValueError, match="average reference"):
            Covariances(estimator="scm").fit_transform(epochs)

    @pytest.mark.parametrize(
        ("kappa", "expected_phrase"),
        [
            (1e20, "exactly linearly dependent"),
            (5e12, "sampling problem, not a rank defect"),
        ],
    )
    def test_singular_error_distinguishes_its_two_causes(
        self,
        kappa: float,
        expected_phrase: str,
        white_epochs: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Exact rank deficiency and ill-conditioning need different remedies.

        A covariance with a zero eigenvalue is a preprocessing artefact -- an
        average reference or an ICA rank reduction. One with a small but
        genuine smallest eigenvalue is a sampling problem, fixed by longer
        epochs or shrinkage. Reporting the first for the second sends the
        reader hunting for a reference that was never applied.

        The second case is exercised directly rather than through ``fit``:
        it needs a condition number between the SPD limit of 1e12 and the
        rank-deficiency cutoff of 1e14, a window too narrow to reach reliably
        by choosing an epoch length.
        """
        transformer = Covariances().fit(white_epochs)
        offending = random_spd(8, rng=rng, condition_number=kappa)
        error = transformer._singular_error(0, 1, (10, 8, 20), offending)
        assert expected_phrase in str(error)

    def test_shrinkage_estimators_survive_average_reference(
        self, white_epochs: np.ndarray
    ) -> None:
        """The remedy the error message recommends must actually work."""
        epochs = white_epochs - white_epochs.mean(axis=1, keepdims=True)
        result = Covariances(
            estimator="ledoit_wolf", audit_conditioning=False
        ).fit_transform(epochs)
        assert bool(np.all(is_spd(result)))


# --------------------------------------------------------------------------- #
# 3. Conditioning
# --------------------------------------------------------------------------- #


class TestConditioning:
    """Shrinkage buys orders of magnitude, and the audit reports the truth."""

    def test_shrinkage_dramatically_improves_conditioning(self, eeg_like) -> None:
        """Measured on spatially correlated data, not white noise.

        At ``T/N = 2`` under a realistic decaying spatial spectrum, the sample
        covariance reaches ``kappa ~ 1e3-1e4`` while Ledoit-Wolf stays near
        ``1e1``. Since AIRM error scales as ``eps * kappa ** 2``, that is the
        difference between roughly nine significant digits and fifteen.
        """
        epochs = eeg_like(40, 22, 44)
        options = {"audit_conditioning": False}
        plain = condition_number(
            Covariances(estimator="scm", **options).fit_transform(epochs)
        )
        shrunk = condition_number(
            Covariances(estimator="ledoit_wolf", **options).fit_transform(epochs)
        )
        #  The observed ratio over 30 seeds runs 86x to 113x with a median of
        #  98x. Asserting 100x would therefore fail about half the time; the
        #  threshold is set at 30x, roughly a third of the worst case, so the
        #  test measures the effect rather than the seed.
        assert np.median(plain) > 30.0 * np.median(shrunk)

    def test_conditioning_improves_with_longer_epochs(self, eeg_like) -> None:
        """More samples per channel is the other remedy, and it must show."""
        options = {"estimator": "scm", "audit_conditioning": False}
        short = condition_number(
            Covariances(**options).fit_transform(eeg_like(20, 16, 32))
        )
        long = condition_number(
            Covariances(**options).fit_transform(eeg_like(20, 16, 2000))
        )
        assert np.median(long) < np.median(short)

    def test_condition_numbers_are_recorded(self, white_epochs: np.ndarray) -> None:
        """Available for the run's provenance record and the methods section."""
        transformer = Covariances().fit(white_epochs)
        result = transformer.transform(white_epochs)
        assert transformer.condition_numbers_.shape == (N_TRIALS,)
        assert np.allclose(transformer.condition_numbers_, condition_number(result))

    def test_audit_warns_on_poor_conditioning(
        self, eeg_like, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A badly conditioned dataset is a finding, and must be logged."""
        #  Tuned to land between the two boundaries that matter: badly
        #  conditioned enough to trip the audit (kappa ~ 4e6, past the 1e6
        #  threshold) but still inside the framework's SPD limit of 1e12, so
        #  the matrices are valid and the warning is the only outcome.
        epochs = eeg_like(20, 22, 40, decay=1.5)
        with caplog.at_level(logging.WARNING, logger="geoq.features.covariance"):
            Covariances(estimator="scm").fit_transform(epochs)
        assert "condition number" in caplog.text

    def test_audit_can_be_disabled(
        self, eeg_like, caplog: pytest.LogCaptureFixture
    ) -> None:
        epochs = eeg_like(20, 22, 40, decay=1.5)
        with caplog.at_level(logging.WARNING, logger="geoq.features.covariance"):
            Covariances(estimator="scm", audit_conditioning=False).fit_transform(epochs)
        assert "condition number" not in caplog.text

    def test_short_epoch_warning(
        self, rng: np.random.Generator, caplog: pytest.LogCaptureFixture
    ) -> None:
        epochs = rng.standard_normal((10, 20, 30))  # T/N = 1.5
        with caplog.at_level(logging.WARNING, logger="geoq.features.covariance"):
            Covariances().fit(epochs)
        assert "below" in caplog.text

    def test_no_short_epoch_warning_when_ample(
        self, white_epochs: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="geoq.features.covariance"):
            Covariances().fit(white_epochs)
        assert caplog.text == ""


# --------------------------------------------------------------------------- #
# 4. Leakage
# --------------------------------------------------------------------------- #


class TestNoLeakage:
    """Each covariance depends on its own epoch and nothing else."""

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_trials_are_independent(
        self, estimator: str, white_epochs: np.ndarray
    ) -> None:
        """Transforming one trial must equal transforming all of them.

        This is the property that makes covariance estimation leakage-free,
        and it must hold for the shrinkage estimators too: their shrinkage
        intensity is estimated per epoch, not pooled. If a future change
        pooled it across trials, a test-fold epoch would influence a training
        epoch's features and this test would catch it.
        """
        transformer = Covariances(estimator=estimator, audit_conditioning=False)
        all_at_once = transformer.fit_transform(white_epochs)
        one_at_a_time = np.vstack(
            [transformer.fit_transform(epoch[None]) for epoch in white_epochs]
        )
        assert np.allclose(all_at_once, one_at_a_time, atol=1e-14)

    def test_result_is_order_invariant(
        self, white_epochs: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Shuffling trials must permute the output, not change it."""
        order = rng.permutation(N_TRIALS)
        base = Covariances().fit_transform(white_epochs)
        shuffled = Covariances().fit_transform(white_epochs[order])
        assert np.allclose(shuffled, base[order], atol=1e-14)

    def test_subset_matches_full(self, white_epochs: np.ndarray) -> None:
        """A fold's covariances equal the corresponding rows of the whole set."""
        full = Covariances().fit_transform(white_epochs)
        subset = Covariances().fit_transform(white_epochs[5:15])
        assert np.allclose(subset, full[5:15], atol=1e-14)


# --------------------------------------------------------------------------- #
# 5. Estimator contract and pipeline use
# --------------------------------------------------------------------------- #


class TestEstimatorContract:
    """Compliance with the scikit-learn interface."""

    def test_get_params_round_trips(self) -> None:
        transformer = Covariances(estimator="oas", shrinkage=0.1)
        params = transformer.get_params()
        assert params["estimator"] == "oas"
        assert params["shrinkage"] == 0.1
        assert Covariances(**params).get_params() == params

    def test_clone_drops_fitted_state(self, white_epochs: np.ndarray) -> None:
        fitted = Covariances(estimator="oas").fit(white_epochs)
        cloned = clone(fitted)
        assert cloned.estimator == "oas"
        assert not hasattr(cloned, "n_channels_")

    def test_fit_returns_self(self, white_epochs: np.ndarray) -> None:
        transformer = Covariances()
        assert transformer.fit(white_epochs) is transformer

    def test_transform_before_fit_raises(self, white_epochs: np.ndarray) -> None:
        with pytest.raises(NotFittedError):
            Covariances().transform(white_epochs)

    def test_fitted_attributes(self, white_epochs: np.ndarray) -> None:
        fitted = Covariances().fit(white_epochs)
        assert fitted.n_channels_ == N_CHANNELS
        assert fitted.n_times_ == N_TIMES
        assert fitted.n_features_in_ == N_CHANNELS


class TestPipelineIntegration:
    """The full classical baseline, end to end."""

    def test_covariances_tangent_space_lda(
        self, white_epochs: np.ndarray, rng: np.random.Generator
    ) -> None:
        """The TS+LDA baseline from raw epochs.

        This is the pipeline the whole thesis is measured against, assembled
        from the framework's own components for the first time.
        """
        labels = rng.integers(0, 2, size=N_TRIALS)
        pipeline = make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(),
            LinearDiscriminantAnalysis(),
        )
        pipeline.fit(white_epochs, labels)
        assert pipeline.predict(white_epochs).shape == (N_TRIALS,)

    def test_cross_validation_runs(
        self, white_epochs: np.ndarray, rng: np.random.Generator
    ) -> None:
        labels = np.zeros(N_TRIALS, dtype=int)
        labels[N_TRIALS // 2 :] = 1
        rng.shuffle(labels)
        pipeline = make_pipeline(
            Covariances(estimator="oas"), TangentSpace(), LinearDiscriminantAnalysis()
        )
        scores = cross_val_score(pipeline, white_epochs, labels, cv=3)
        assert scores.shape == (3,)

    def test_estimator_is_settable_through_the_pipeline(self) -> None:
        pipeline = make_pipeline(Covariances(), TangentSpace())
        pipeline.set_params(covariances__estimator="ledoit_wolf")
        assert pipeline.named_steps["covariances"].estimator == "ledoit_wolf"


# --------------------------------------------------------------------------- #
# 6. Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    """Bad input and bad configuration fail at the point of use."""

    @pytest.mark.parametrize(
        ("kwargs", "pattern"),
        [
            ({"estimator": "sample"}, "estimator must be one of"),
            ({"estimator": "lwf"}, "estimator must be one of"),
            ({"shrinkage": -0.1}, r"\[0, 1\]"),
            ({"shrinkage": 1.5}, r"\[0, 1\]"),
        ],
    )
    def test_invalid_parameters_rejected_at_fit(
        self, kwargs: dict, pattern: str, white_epochs: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match=pattern):
            Covariances(**kwargs).fit(white_epochs)

    def test_constructor_does_not_validate(self) -> None:
        """Required by scikit-learn: ``clone`` must work before ``fit``."""
        assert Covariances(estimator="nonsense").estimator == "nonsense"  # type: ignore[arg-type]

    def test_two_dimensional_input_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="n_trials, n_channels, n_times"):
            Covariances().fit(rng.standard_normal((30, 512)))

    def test_transposed_input_hint(self, rng: np.random.Generator) -> None:
        """The channels-first convention must be stated in the error.

        A ``(n_trials, n_times, n_channels)`` array has the right rank and the
        wrong meaning; without the hint it produces a plausible matrix of the
        wrong size and no explanation.
        """
        #  The realistic mistake: an array that is genuinely (n_trials,
        #  n_times, n_channels). It has the right rank, so only the absurd
        #  axis sizes reveal it -- 512 "channels" recorded for 8 "samples".
        with pytest.raises(ValueError) as excinfo:
            Covariances().fit(rng.standard_normal((10, 512, 8)))
        assert "needs transposing to channels-first" in str(excinfo.value)

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="no trials"):
            Covariances().fit(np.empty((0, 4, 100)))

    def test_non_finite_input_rejected(self, white_epochs: np.ndarray) -> None:
        epochs = white_epochs.copy()
        epochs[4, 2, 100] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            Covariances().fit(epochs)

    def test_non_finite_message_names_the_trial(self, white_epochs: np.ndarray) -> None:
        epochs = white_epochs.copy()
        epochs[7, 0, 0] = np.inf
        with pytest.raises(ValueError, match="trial 7"):
            Covariances().fit(epochs)

    def test_channel_mismatch_rejected(
        self, white_epochs: np.ndarray, rng: np.random.Generator
    ) -> None:
        fitted = Covariances().fit(white_epochs)
        with pytest.raises(ValueError, match="Expected 8 channels"):
            fitted.transform(rng.standard_normal((5, 12, 512)))
