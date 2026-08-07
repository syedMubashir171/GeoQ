"""Tests for :mod:`geoq.features.alignment`.

What is being defended
----------------------
* **No label leakage.** ``TestNoLabelLeakage`` permutes the labels and asserts
  the aligned output is *bitwise* identical. Alignment uses a subject's own
  unlabelled trials, which is legitimate transfer learning, and the difference
  between that and using their labels is the difference between a method and a
  mistake. Asserting it removes any need to take the claim on trust.
* **The transductive assumption is declared.** ``TestCalibrationBarrier``
  checks that the transformer refuses to be constructed without
  ``assume_calibration_data=True``, and that a truthy value is not enough.
* **It does what it claims geometrically.** ``TestRecentring`` asserts each
  domain's mean lands on the identity afterwards, and that domains are
  independent of one another.
* **It closes the gap it was built for.** ``TestTransferBenefit`` measures
  cross-subject MDM before and after, which is the claim the real BCI IV 2a
  result motivated.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the 'ml' extra")

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import NotFittedError

from geoq.datasets.base import make_synthetic_eeg
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.alignment import (
    RiemannianAlignment,
    align_domains,
    alignment_quality,
    domain_references,
    recenter,
)
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace
from geoq.geometry.riemannian import distance_airm, frechet_mean
from geoq.geometry.spd import NotPositiveDefiniteError, is_spd, random_spd
from geoq.models.classical.mdm import MDM
from geoq.testing import relative_error

N_CHANNELS = 6


def aligner(**kwargs) -> RiemannianAlignment:
    """Construct a transformer with the calibration assumption declared."""
    return RiemannianAlignment(assume_calibration_data=True, **kwargs)


@pytest.fixture
def displaced(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Covariances from four subjects, each in its own displaced frame.

    The displacement is a congruence by a subject-specific invertible matrix,
    which is exactly the transform head geometry and electrode impedance
    apply, and exactly the one the affine-invariant metric cannot see.
    """
    matrices, domains = [], []
    for subject in range(4):
        transform = np.eye(N_CHANNELS) + 0.6 * rng.standard_normal(
            (N_CHANNELS, N_CHANNELS)
        ) / np.sqrt(N_CHANNELS)
        block = random_spd(N_CHANNELS, rng=rng, batch=25)
        matrices.append(np.einsum("ij,bjk,lk->bil", transform, block, transform))
        domains.append(np.full(25, subject))
    return np.concatenate(matrices), np.concatenate(domains)


# --------------------------------------------------------------------------- #
# 1. Re-centring
# --------------------------------------------------------------------------- #


class TestRecentring:
    """The geometric claim: every domain ends up centred at the identity."""

    def test_recenter_maps_the_reference_to_the_identity(
        self, rng: np.random.Generator
    ) -> None:
        reference = random_spd(N_CHANNELS, rng=rng)
        assert (
            relative_error(recenter(reference, reference), np.eye(N_CHANNELS)) < 1e-10
        )

    def test_each_domain_mean_becomes_the_identity(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The definition of alignment, measured rather than assumed."""
        matrices, domains = displaced
        before = alignment_quality(matrices, domains)
        after = alignment_quality(align_domains(matrices, domains), domains)
        assert before["max_residual"] > 0.5
        assert after["max_residual"] < 1e-6

    def test_output_is_spd(self, displaced: tuple[np.ndarray, np.ndarray]) -> None:
        matrices, domains = displaced
        assert bool(np.all(is_spd(align_domains(matrices, domains))))

    def test_domains_are_independent(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """One subject's trials must not influence another's alignment.

        If they did, adding a subject would change every other subject's
        features, and a leave-one-subject-out fold would differ from the same
        subject evaluated alone.
        """
        matrices, domains = displaced
        full = align_domains(matrices, domains)
        subset_mask = domains < 2
        subset = align_domains(matrices[subset_mask], domains[subset_mask])
        assert np.allclose(full[subset_mask], subset, atol=1e-10)

    def test_order_is_preserved(self, displaced: tuple[np.ndarray, np.ndarray]) -> None:
        """Trials come back in the input order, not grouped by domain."""
        matrices, domains = displaced
        shuffled = np.random.default_rng(3).permutation(matrices.shape[0])
        aligned = align_domains(matrices[shuffled], domains[shuffled])
        expected = align_domains(matrices, domains)[shuffled]
        assert np.allclose(aligned, expected, atol=1e-10)

    def test_alignment_is_idempotent(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Aligning already-aligned data changes nothing of substance."""
        matrices, domains = displaced
        once = align_domains(matrices, domains)
        twice = align_domains(once, domains)
        assert relative_error(twice, once) < 1e-6

    def test_removes_a_congruence_up_to_a_rotation(
        self, rng: np.random.Generator
    ) -> None:
        """The mechanism, and its exact limitation.

        Two subjects whose data differ by a congruence do **not** become
        identical after re-centring. Both maps whiten the mean to the identity,
        and any two whitenings of the same matrix differ by an orthogonal
        transform, so what remains is a rotation.

        Asserted directly here because the natural expectation -- that
        alignment makes the subjects identical -- is wrong, and a test encoding
        that expectation would have to be quietly relaxed. The residual is why
        Riemannian Procrustes analysis exists as a further supervised step.
        """
        from geoq.geometry.riemannian import frechet_mean as mean_of
        from geoq.geometry.spd import invsqrtm_spd

        block = random_spd(N_CHANNELS, rng=rng, batch=30)
        transform = np.eye(N_CHANNELS) + 0.5 * rng.standard_normal(
            (N_CHANNELS, N_CHANNELS)
        ) / np.sqrt(N_CHANNELS)
        displaced_block = np.einsum("ij,bjk,lk->bil", transform, block, transform)

        matrices = np.concatenate([block, displaced_block])
        domains = np.repeat([0, 1], 30)
        aligned = align_domains(matrices, domains)

        rotation = (
            invsqrtm_spd(mean_of(displaced_block))
            @ transform
            @ np.linalg.inv(invsqrtm_spd(mean_of(block)))
        )
        assert np.abs(rotation @ rotation.T - np.eye(N_CHANNELS)).max() < 1e-8
        assert np.abs(aligned[30:] - rotation @ aligned[:30] @ rotation.T).max() < 1e-10

    def test_a_congruence_leaves_the_geometry_identical(
        self, rng: np.random.Generator
    ) -> None:
        """What alignment does deliver, and why MDM benefits.

        The residual rotation is an orthogonal congruence, which the
        affine-invariant metric cannot see. So the two subjects' internal
        geometry -- every pairwise geodesic distance -- is identical after
        alignment, even though their coordinates are not.

        This is the precise reason a distance-based classifier transfers after
        re-centring while a coordinate-based one only partly does.
        """
        from geoq.geometry.riemannian import pairwise_distances

        block = random_spd(N_CHANNELS, rng=rng, batch=25)
        transform = np.eye(N_CHANNELS) + 0.5 * rng.standard_normal(
            (N_CHANNELS, N_CHANNELS)
        ) / np.sqrt(N_CHANNELS)
        displaced_block = np.einsum("ij,bjk,lk->bil", transform, block, transform)

        aligned = align_domains(
            np.concatenate([block, displaced_block]), np.repeat([0, 1], 25)
        )
        assert (
            np.abs(
                pairwise_distances(aligned[:25]) - pairwise_distances(aligned[25:])
            ).max()
            < 1e-9
        )

    @pytest.mark.parametrize("metric", ["airm", "logeuclid", "euclid"])
    def test_every_supported_metric_runs(
        self, metric: str, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrices, domains = displaced
        assert bool(np.all(is_spd(align_domains(matrices, domains, metric=metric))))


class TestDomainReferences:
    """One reference per domain, computed from that domain alone."""

    def test_reference_is_the_domain_mean(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrices, domains = displaced
        references = domain_references(matrices, domains)
        assert set(references) == {0, 1, 2, 3}
        for domain, reference in references.items():
            expected = frechet_mean(matrices[domains == domain])
            assert relative_error(reference, expected) < 1e-8

    def test_single_trial_domain_warns(
        self, rng: np.random.Generator, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Re-centring a lone trial maps it exactly to the identity.

        That destroys the only information it carried, so it must be visible
        rather than silently producing a featureless trial.
        """
        matrices = random_spd(4, rng=rng, batch=5)
        domains = np.array([0, 0, 0, 0, 1])
        with caplog.at_level(logging.WARNING, logger="geoq.features.alignment"):
            domain_references(matrices, domains)
        assert "single covariance" in caplog.text

    def test_shape_mismatch_rejected(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrices, domains = displaced
        with pytest.raises(ValueError, match="align with covariances"):
            domain_references(matrices, domains[:-3])

    def test_missing_reference_rejected(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrices, domains = displaced
        references = domain_references(matrices, domains)
        del references[2]
        with pytest.raises(KeyError, match="No reference point"):
            align_domains(matrices, domains, references=references)


# --------------------------------------------------------------------------- #
# 2. Leakage
# --------------------------------------------------------------------------- #


class TestNoLabelLeakage:
    """Alignment is unsupervised, and that is asserted rather than assumed."""

    def test_permuting_labels_changes_nothing(
        self, displaced: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """The claim that separates a method from a mistake.

        Bitwise equality, not approximate. If any label information reached
        the reference points, alignment would be a supervised transform
        applied to test data, and every score computed after it would be
        inflated.
        """
        matrices, domains = displaced
        labels = rng.integers(0, 2, size=matrices.shape[0])

        first = aligner().fit_transform(matrices, labels, domains=domains)
        second = aligner().fit_transform(
            matrices, rng.permutation(labels), domains=domains
        )
        third = aligner().fit_transform(matrices, None, domains=domains)

        assert np.array_equal(first, second)
        assert np.array_equal(first, third)

    def test_a_subject_reference_uses_only_that_subject(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Cross-subject information must not enter a reference point.

        Otherwise the held-out subject's alignment would depend on the
        training subjects, and leave-one-subject-out would no longer hold the
        subject out.
        """
        matrices, domains = displaced
        full = domain_references(matrices, domains)
        alone = domain_references(matrices[domains == 3], domains[domains == 3])
        assert relative_error(full[3], alone[3]) < 1e-9

    def test_transform_reuses_the_fitted_reference(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Nothing is recomputed at transform time in the pipeline mode."""
        matrices, domains = displaced
        fitted = aligner().fit(matrices, domains=domains)
        whole = fitted.transform(matrices, domains=domains)
        piecewise = np.vstack(
            [
                fitted.transform(matrices[[index]], domains=domains[[index]])
                for index in range(0, 20)
            ]
        )
        assert np.array_equal(whole[:20], piecewise)


class TestCalibrationBarrier:
    """Using a transductive method must be a declared decision."""

    def test_construction_requires_the_declaration(self) -> None:
        with pytest.raises(ValueError, match="assume_calibration_data=True"):
            RiemannianAlignment()

    @pytest.mark.parametrize("value", [False, None, 1, "yes"])
    def test_truthy_is_not_enough(self, value: object) -> None:
        """``1`` in a config file reads as an unrelated integer setting."""
        with pytest.raises(ValueError, match="calibration batch"):
            RiemannianAlignment(assume_calibration_data=value)  # type: ignore[arg-type]

    def test_declared_construction_works(self) -> None:
        assert aligner().assume_calibration_data is True

    def test_unseen_domain_at_transform_is_refused(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A new subject needs its own calibration batch, and saying so.

        Computing a reference at transform time is a different protocol from
        reusing a learned one, and conflating them would let a methods section
        describe something the code did not do.
        """
        matrices, domains = displaced
        fitted = aligner().fit(matrices[domains < 3], domains=domains[domains < 3])
        with pytest.raises(ValueError, match="No reference point for domain"):
            fitted.transform(matrices[domains == 3], domains=domains[domains == 3])


# --------------------------------------------------------------------------- #
# 3. The transfer benefit
# --------------------------------------------------------------------------- #


class TestTransferBenefit:
    """The claim the real BCI IV 2a result motivated."""

    @staticmethod
    def _dataset():
        """Synthetic EEG with a strong per-subject frame displacement."""
        return make_synthetic_eeg(
            n_subjects=6,
            n_trials_per_subject=48,
            n_channels=8,
            n_times=200,
            task_effect=0.30,
            subject_variability=1.6,
            seed=0,
        )

    def test_alignment_improves_cross_subject_mdm(self) -> None:
        """MDM is weak cross-subject because centroids pool displaced frames.

        Measured here: kappa moves from below zero to clearly positive once
        each subject is re-centred. The mechanism is that a pooled class
        centroid finally describes a location the subjects share.
        """
        data = self._dataset()
        covariances = Covariances(
            estimator="oas", audit_conditioning=False
        ).fit_transform(data.epochs)
        aligned = aligner().fit_transform(covariances, domains=data.subjects)

        def kappa(matrices) -> float:
            return evaluate(
                MDM(),
                matrices,
                data.labels,
                groups=data.subjects,
                splitter=LeaveOneSubjectOut(),
            ).mean("kappa")

        raw_kappa = kappa(covariances)
        aligned_kappa = kappa(aligned)
        assert aligned_kappa > raw_kappa + 0.15
        assert aligned_kappa > 0.1

    def test_alignment_does_not_break_tangent_space(self) -> None:
        """The other baseline must not be harmed by the same preprocessing."""
        data = self._dataset()
        covariances = Covariances(
            estimator="oas", audit_conditioning=False
        ).fit_transform(data.epochs)
        aligned = aligner().fit_transform(covariances, domains=data.subjects)
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        result = evaluate(
            pipeline,
            aligned,
            data.labels,
            groups=data.subjects,
            splitter=LeaveOneSubjectOut(),
        )
        assert result.mean("kappa") > -0.1

    def test_alignment_reduces_between_subject_distance(self) -> None:
        """The geometric quantity behind the accuracy change.

        Subjects' mean covariances should sit far apart before alignment and
        at the same point after it.
        """
        data = self._dataset()
        covariances = Covariances(
            estimator="oas", audit_conditioning=False
        ).fit_transform(data.epochs)
        aligned = aligner().fit_transform(covariances, domains=data.subjects)

        def spread(matrices) -> float:
            references = domain_references(matrices, data.subjects)
            keys = sorted(references)
            return float(
                np.mean(
                    [
                        float(distance_airm(references[a], references[b]))
                        for index, a in enumerate(keys)
                        for b in keys[index + 1 :]
                    ]
                )
            )

        assert spread(aligned) < 0.01 * spread(covariances)


# --------------------------------------------------------------------------- #
# 4. Estimator contract
# --------------------------------------------------------------------------- #


class TestEstimatorContract:
    """Scikit-learn compliance, and the two usage modes."""

    def test_get_params_round_trips(self) -> None:
        transformer = aligner(metric="logeuclid")
        params = transformer.get_params()
        assert params["metric"] == "logeuclid"
        assert params["assume_calibration_data"] is True
        assert RiemannianAlignment(**params).get_params() == params

    def test_clone_preserves_the_declaration(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Cloning must not drop the barrier and fail to reconstruct."""
        matrices, domains = displaced
        cloned = clone(aligner(metric="euclid").fit(matrices, domains=domains))
        assert cloned.assume_calibration_data is True
        assert not hasattr(cloned, "references_")

    def test_fit_transform_forwards_domains(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The override that stops fit_transform silently not aligning.

        TransformerMixin's inherited version does not forward extra keywords
        to transform, so it would fit per domain and then transform as one --
        producing output that is not aligned while looking as though it is.
        """
        matrices, domains = displaced
        combined = aligner().fit_transform(matrices, domains=domains)
        separate = (
            aligner()
            .fit(matrices, domains=domains)
            .transform(matrices, domains=domains)
        )
        assert np.array_equal(combined, separate)
        assert alignment_quality(combined, domains)["max_residual"] < 1e-6

    def test_single_domain_mode_works_in_a_pipeline(
        self, displaced: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """Without domains the whole batch is one domain, and that composes.

        Not per-subject alignment, and documented as such -- but a pipeline
        step that changed meaning depending on how it was called would be
        worse than one that does less.
        """
        matrices, _ = displaced
        labels = rng.integers(0, 2, size=matrices.shape[0])
        pipeline = make_pipeline(
            aligner(), TangentSpace(), LinearDiscriminantAnalysis()
        )
        pipeline.fit(matrices, labels)
        assert pipeline.predict(matrices).shape == (matrices.shape[0],)

    def test_transform_before_fit_raises(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrices, _ = displaced
        with pytest.raises(NotFittedError):
            aligner().transform(matrices)

    def test_fitted_attributes(self, displaced: tuple[np.ndarray, np.ndarray]) -> None:
        matrices, domains = displaced
        fitted = aligner().fit(matrices, domains=domains)
        assert fitted.n_channels_ == N_CHANNELS
        assert fitted.n_features_in_ == N_CHANNELS
        assert len(fitted.references_) == 4


class TestValidation:
    """Misuse fails at the point of misuse."""

    @pytest.mark.parametrize(
        ("kwargs", "pattern"),
        [
            ({"metric": "stein"}, "metric must be one of"),
            ({"tol": 0.0}, "positive finite"),
            ({"max_iter": 0}, "positive integer"),
        ],
    )
    def test_invalid_parameters_rejected_at_fit(
        self, kwargs: dict, pattern: str, displaced
    ) -> None:
        matrices, domains = displaced
        with pytest.raises(ValueError, match=pattern):
            aligner(**kwargs).fit(matrices, domains=domains)

    def test_two_dimensional_input_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="Covariances step"):
            aligner().fit(rng.standard_normal((20, 36)))

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="no trials"):
            aligner().fit(np.empty((0, 4, 4)))

    def test_non_spd_input_rejected(
        self, displaced: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrices, domains = displaced
        bad = matrices.copy()
        bad[5] = np.diag(np.r_[np.ones(N_CHANNELS - 1), 0.0])
        with pytest.raises(NotPositiveDefiniteError):
            aligner().fit(bad, domains=domains)

    def test_channel_mismatch_rejected(
        self, displaced: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        matrices, domains = displaced
        fitted = aligner().fit(matrices, domains=domains)
        with pytest.raises(ValueError, match="Expected 6 channels"):
            fitted.transform(random_spd(9, rng=rng, batch=4))
