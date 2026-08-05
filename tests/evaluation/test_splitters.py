"""Tests for :mod:`geoq.evaluation.splitters`.

What is being defended
----------------------
A splitter's :class:`~geoq.evaluation.splitters.SplitterInfo` is a claim about
what a protocol guarantees, and that claim ends up in a paper's methods
section. So the tests here do not check that the flags are *set* correctly --
they check that the flags are *true*, by inspecting the folds a splitter
actually produces.

``TestGuaranteesAreReal`` verifies subject independence by intersecting the
subject sets of each train and test fold, and temporal disjointness by
comparing recording positions. If someone later changes a splitter's logic and
forgets its metadata, or changes the metadata and forgets the logic, these
tests fail.

``TestLeakageIsMeasurable`` closes the loop: it simulates EEG with subject
identity, temporal drift, and a weak task effect, then shows that the leaky
protocol reports roughly forty accuracy points more than leave-one-subject-out
on identical data and an identical pipeline. That is Paper 1's central
experiment in miniature, and having it as a test means the framework's ability
to demonstrate the effect cannot silently regress.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from geoq.evaluation.splitters import (
    SPLITTERS,
    LeakyShuffleSplit,
    LeaveOneSubjectOut,
    SplitterInfo,
    WithinSubjectChronological,
    WithinSubjectKFold,
    make_splitter,
)

N_SUBJECTS = 6
N_PER_SUBJECT = 20
N_TRIALS = N_SUBJECTS * N_PER_SUBJECT


@pytest.fixture
def trials() -> np.ndarray:
    """Placeholder trial array; splitters only use its first-axis length."""
    return np.zeros((N_TRIALS, 3, 3))


@pytest.fixture
def subjects() -> np.ndarray:
    """Subject identifiers in recording order."""
    return np.repeat(np.arange(N_SUBJECTS), N_PER_SUBJECT)


@pytest.fixture
def labels(rng: np.random.Generator) -> np.ndarray:
    """Balanced binary labels."""
    y = np.tile([0, 1], N_TRIALS // 2)
    rng.shuffle(y)
    return y


def all_splitters() -> list[tuple[str, dict]]:
    """Every splitter with a minimal valid configuration."""
    return [
        ("loso", {}),
        ("within_subject_chronological", {}),
        ("within_subject_kfold", {"n_splits": 4}),
        ("leaky_shuffle", {"random_state": 0, "acknowledge_leakage": True}),
    ]


# --------------------------------------------------------------------------- #
# 1. The guarantees, verified rather than declared
# --------------------------------------------------------------------------- #


class TestGuaranteesAreReal:
    """Each ``SplitterInfo`` flag is checked against the folds produced."""

    @pytest.mark.parametrize(("name", "kwargs"), all_splitters())
    def test_subject_independence_flag_matches_behaviour(
        self,
        name: str,
        kwargs: dict,
        trials: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        """The claim is verified by intersecting subject sets, not trusted.

        A splitter whose metadata says ``subject_independent=True`` while its
        folds share a subject would put a false guarantee into a methods
        section. Checking the folds is the only way to know.
        """
        splitter = make_splitter(name, **kwargs)
        overlaps = [
            bool(set(subjects[train]) & set(subjects[test]))
            for train, test in splitter.split(trials, labels, subjects)
        ]
        assert splitter.info.subject_independent == (not any(overlaps))

    def test_loso_shares_no_subject(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        for train, test in LeaveOneSubjectOut().split(trials, labels, subjects):
            assert len(set(subjects[test])) == 1
            assert not set(subjects[train]) & set(subjects[test])

    def test_chronological_split_is_temporally_ordered(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        """Every training trial precedes every test trial, per subject.

        The guarantee that makes a chronological split honest despite
        autocorrelation: a test epoch's neighbours are other test epochs, not
        training ones.
        """
        splitter = WithinSubjectChronological(train_fraction=0.6)
        for train, test in splitter.split(trials, labels, subjects):
            for subject in np.unique(subjects):
                train_positions = train[subjects[train] == subject]
                test_positions = test[subjects[test] == subject]
                assert train_positions.max() < test_positions.min()

    def test_leaky_shuffle_actually_mixes_subjects(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        """The negative control must genuinely leak.

        If it stopped leaking, the inflation measurement would silently return
        zero and Paper 1's central comparison would show no effect for the
        wrong reason.
        """
        splitter = LeakyShuffleSplit(
            n_splits=4, random_state=0, acknowledge_leakage=True
        )
        for train, test in splitter.split(trials, labels, subjects):
            assert set(subjects[train]) & set(subjects[test])

    def test_leaky_shuffle_breaks_temporal_order(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        """Adjacent trials land on opposite sides of the split.

        This is the autocorrelation leak, distinct from the subject leak, and
        it is what makes a shuffled split unsound even for a single subject.

        Adjacency is counted only between trials of the *same* subject.
        Counting global index neighbours would score the chronological
        splitter at 0.15 purely because one subject's block abuts the next
        one's in the array, which is not leakage at all.

        Stated comparatively rather than against a fixed cutoff. Under
        shuffling with ``k`` folds the expected rate is
        ``1 - (1 - 1/k) ** 2``, about 0.44 at ``k = 4``. A chronological split
        has exactly one boundary per subject, so its rate is roughly
        ``n_subjects / n_train`` -- an order of magnitude smaller, and that
        gap is the quantity worth asserting.
        """

        def adjacency_rate(splitter) -> float:
            train, test = next(iter(splitter.split(trials, labels, subjects)))
            test_set = set(test.tolist())
            neighbours = sum(
                1
                for index in train
                if any(
                    (index + offset) in test_set
                    and subjects[index + offset] == subjects[index]
                    for offset in (-1, 1)
                    if 0 <= index + offset < subjects.size
                )
            )
            return neighbours / train.size

        leaky = adjacency_rate(
            LeakyShuffleSplit(n_splits=4, random_state=0, acknowledge_leakage=True)
        )
        honest = adjacency_rate(WithinSubjectChronological(train_fraction=0.6))

        assert leaky > 0.3
        assert honest < 0.15
        assert leaky > 3.0 * honest

    def test_is_optimistic_matches_the_two_flags(self) -> None:
        clean = SplitterInfo("x", True, True, "")
        assert not clean.is_optimistic
        for pair in ((True, False), (False, True), (False, False)):
            assert SplitterInfo("x", *pair, "").is_optimistic


# --------------------------------------------------------------------------- #
# 2. Structural correctness
# --------------------------------------------------------------------------- #


class TestFoldStructure:
    """Folds partition the data as they claim to."""

    @pytest.mark.parametrize(("name", "kwargs"), all_splitters())
    def test_train_and_test_are_disjoint(
        self,
        name: str,
        kwargs: dict,
        trials: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        splitter = make_splitter(name, **kwargs)
        for train, test in splitter.split(trials, labels, subjects):
            assert not set(train.tolist()) & set(test.tolist())
            assert train.size > 0
            assert test.size > 0

    @pytest.mark.parametrize(("name", "kwargs"), all_splitters())
    def test_indices_are_valid(
        self,
        name: str,
        kwargs: dict,
        trials: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        splitter = make_splitter(name, **kwargs)
        for train, test in splitter.split(trials, labels, subjects):
            for indices in (train, test):
                assert indices.dtype.kind == "i"
                assert indices.min() >= 0
                assert indices.max() < N_TRIALS
                assert len(set(indices.tolist())) == indices.size

    @pytest.mark.parametrize(("name", "kwargs"), all_splitters())
    def test_get_n_splits_matches_the_folds_produced(
        self,
        name: str,
        kwargs: dict,
        trials: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        """Declared and produced fold counts must agree.

        Scikit-learn calls ``get_n_splits`` before iterating, so a mismatch
        silently truncates or pads a results array.
        """
        splitter = make_splitter(name, **kwargs)
        produced = sum(1 for _ in splitter.split(trials, labels, subjects))
        assert splitter.get_n_splits(trials, labels, subjects) == produced

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("loso", {}),
            ("within_subject_kfold", {"n_splits": 4}),
            ("leaky_shuffle", {"random_state": 0, "acknowledge_leakage": True}),
        ],
    )
    def test_every_trial_is_tested_exactly_once(
        self,
        name: str,
        kwargs: dict,
        trials: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        """Complete, non-overlapping test coverage.

        Excludes the chronological splitter, which deliberately never tests a
        subject's earliest trials -- those are the calibration data.
        """
        splitter = make_splitter(name, **kwargs)
        tested = np.concatenate(
            [test for _, test in splitter.split(trials, labels, subjects)]
        )
        assert np.array_equal(np.sort(tested), np.arange(N_TRIALS))

    def test_loso_matches_sklearn(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        """Fold-for-fold parity with scikit-learn's LeaveOneGroupOut.

        The one protocol here with an exact reference equivalent, so the
        comparison is worth making.
        """
        sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
        reference = sklearn_model_selection.LeaveOneGroupOut()
        mine = list(LeaveOneSubjectOut().split(trials, labels, subjects))
        theirs = list(reference.split(trials, labels, subjects))
        assert len(mine) == len(theirs)
        for (my_train, my_test), (their_train, their_test) in zip(
            mine, theirs, strict=True
        ):
            assert np.array_equal(my_train, their_train)
            assert np.array_equal(my_test, their_test)

    def test_chronological_expanding_window(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        """With several folds the training window grows monotonically."""
        splitter = WithinSubjectChronological(train_fraction=0.4, n_splits=3)
        sizes = [train.size for train, _ in splitter.split(trials, labels, subjects)]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_kfold_blocks_are_contiguous(
        self, trials: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> None:
        """Contiguity is what limits the leak to the fold boundaries."""
        splitter = WithinSubjectKFold(n_splits=4)
        for _, test in splitter.split(trials, labels, subjects):
            for subject in np.unique(subjects):
                block = test[subjects[test] == subject]
                assert np.array_equal(block, np.arange(block.min(), block.max() + 1))


# --------------------------------------------------------------------------- #
# 3. The leakage measurement
# --------------------------------------------------------------------------- #


class TestLeakageIsMeasurable:
    """Paper 1's central experiment, reduced to a test."""

    @staticmethod
    def _simulate(
        seed: int,
        *,
        n_subjects: int = 6,
        n_per_subject: int = 40,
        n_channels: int = 5,
        task_effect: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate EEG with the three structures that matter.

        Subject identity dominates: each participant's trials cluster around
        their own anchor on the manifold. Temporal drift makes consecutive
        trials similar, reproducing autocorrelation. The task effect is
        deliberately weak, because that is the realistic regime -- a strong
        effect would be decodable under any protocol and the leak would not
        show.

        Args:
            seed: Random seed.
            n_subjects: Number of simulated participants.
            n_per_subject: Trials per participant.
            n_channels: Channel count.
            task_effect: Interpolation weight toward the class direction.

        Returns:
            Trials, labels, and subject identifiers.
        """
        from geoq.geometry.riemannian import geodesic
        from geoq.geometry.spd import random_spd

        rng = np.random.default_rng(seed)
        directions = [random_spd(n_channels, rng=rng) for _ in range(2)]
        matrices, targets, groups = [], [], []
        for subject in range(n_subjects):
            state = random_spd(n_channels, rng=rng)
            for _ in range(n_per_subject):
                state = geodesic(state, random_spd(n_channels, rng=rng), 0.02)
                label = int(rng.integers(0, 2))
                matrices.append(geodesic(state, directions[label], task_effect))
                targets.append(label)
                groups.append(subject)
        return np.stack(matrices), np.array(targets), np.array(groups)

    def test_leaky_protocol_inflates_accuracy(self) -> None:
        """Identical data, identical pipeline, two protocols.

        Asserted on Cohen's kappa rather than accuracy, because accuracy is
        not interpretable here. On this simulation the honest protocol yields
        accuracy near 0.53 while balanced accuracy is exactly 0.500 and kappa
        exactly 0.000: the classifier predicts a single class throughout, and
        the accuracy figure is reporting each fold's class balance rather than
        any decoding. Kappa is zero at chance regardless of balance, so it
        states the situation unambiguously.

        Measured: honest kappa 0.000, leaky kappa 0.975. The leaky protocol
        does not merely improve on a weak result -- it manufactures a
        near-perfect one from data in which the honest protocol finds nothing
        generalisable at all.
        """
        pytest.importorskip("sklearn")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline

        from geoq.features.tangent_space import TangentSpace

        matrices, targets, groups = self._simulate(seed=0)
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())

        honest = cross_val_score(
            pipeline,
            matrices,
            targets,
            groups=groups,
            cv=LeaveOneSubjectOut(),
            scoring="balanced_accuracy",
        ).mean()
        leaky = cross_val_score(
            pipeline,
            matrices,
            targets,
            cv=LeakyShuffleSplit(n_splits=5, random_state=0, acknowledge_leakage=True),
            scoring="balanced_accuracy",
        ).mean()

        assert honest < 0.6, "honest protocol should find little to generalise"
        assert leaky > 0.9, "leaky protocol should look near-perfect"
        assert leaky - honest > 0.3

    def test_honest_protocol_is_not_trivially_broken(self) -> None:
        """LOSO must still detect the task effect, not merely score at chance.

        Without this, the previous test could pass because the honest
        protocol was broken rather than because the leaky one was inflated.
        """
        pytest.importorskip("sklearn")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline

        from geoq.features.tangent_space import TangentSpace

        matrices, targets, groups = self._simulate(seed=0, task_effect=0.5)
        pipeline = make_pipeline(TangentSpace(), LinearDiscriminantAnalysis())
        honest = cross_val_score(
            pipeline,
            matrices,
            targets,
            groups=groups,
            cv=LeaveOneSubjectOut(),
            scoring="balanced_accuracy",
        ).mean()
        assert honest > 0.75


# --------------------------------------------------------------------------- #
# 4. The leakage acknowledgement barrier
# --------------------------------------------------------------------------- #


class TestLeakyShuffleBarrier:
    """Using the unsound protocol must be a deliberate, visible act."""

    def test_construction_requires_acknowledgement(self) -> None:
        with pytest.raises(ValueError, match="acknowledge_leakage=True"):
            LeakyShuffleSplit(n_splits=5, random_state=0)

    @pytest.mark.parametrize("value", [False, None, 1, "yes"])
    def test_only_literal_true_is_accepted(self, value: object) -> None:
        """Truthy is not enough.

        ``acknowledge_leakage=1`` would pass an ordinary truthiness check while
        looking like an unrelated integer argument in a config file. Requiring
        the literal keeps the acknowledgement legible in a diff.
        """
        with pytest.raises(ValueError, match="inflated scores by design"):
            LeakyShuffleSplit(
                n_splits=5,
                random_state=0,
                acknowledge_leakage=value,  # type: ignore[arg-type]
            )

    def test_random_state_is_mandatory(self) -> None:
        """An unreproducible inflation estimate is not a measurement."""
        with pytest.raises(TypeError):
            LeakyShuffleSplit(n_splits=5, acknowledge_leakage=True)  # type: ignore[call-arg]

    def test_warns_on_every_use(
        self, trials: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning belongs in the log beside the numbers it explains."""
        splitter = LeakyShuffleSplit(
            n_splits=3, random_state=0, acknowledge_leakage=True
        )
        with caplog.at_level(logging.WARNING, logger="geoq.evaluation.splitters"):
            list(splitter.split(trials))
            list(splitter.split(trials))
        assert caplog.text.count("LeakyShuffleSplit in use") == 2

    def test_repr_names_the_hazard(self) -> None:
        """Anything printing the CV object should show the warning too."""
        splitter = LeakyShuffleSplit(
            n_splits=3, random_state=0, acknowledge_leakage=True
        )
        assert "DELIBERATELY LEAKY" in repr(splitter)

    def test_folds_are_reproducible(self, trials: np.ndarray) -> None:
        first = list(
            LeakyShuffleSplit(
                n_splits=4, random_state=7, acknowledge_leakage=True
            ).split(trials)
        )
        second = list(
            LeakyShuffleSplit(
                n_splits=4, random_state=7, acknowledge_leakage=True
            ).split(trials)
        )
        for (train_a, test_a), (train_b, test_b) in zip(first, second, strict=True):
            assert np.array_equal(train_a, train_b)
            assert np.array_equal(test_a, test_b)

    def test_different_seeds_give_different_folds(self, trials: np.ndarray) -> None:
        first = next(
            iter(
                LeakyShuffleSplit(
                    n_splits=4, random_state=1, acknowledge_leakage=True
                ).split(trials)
            )
        )
        second = next(
            iter(
                LeakyShuffleSplit(
                    n_splits=4, random_state=2, acknowledge_leakage=True
                ).split(trials)
            )
        )
        assert not np.array_equal(first[1], second[1])


# --------------------------------------------------------------------------- #
# 5. Registry and validation
# --------------------------------------------------------------------------- #


class TestRegistry:
    """Protocols are selected by name from configuration."""

    def test_every_registered_name_constructs(self) -> None:
        for name, kwargs in all_splitters():
            assert name in SPLITTERS
            assert isinstance(make_splitter(name, **kwargs).info, SplitterInfo)

    def test_info_name_matches_the_registry_key(self) -> None:
        """Otherwise the provenance record would name a different protocol."""
        for name, kwargs in all_splitters():
            assert make_splitter(name, **kwargs).info.name == name

    def test_unknown_name_raises(self) -> None:
        """A typo must not silently fall back to a default protocol."""
        with pytest.raises(ValueError, match="Unknown splitter"):
            make_splitter("loso_kfold")

    def test_error_lists_available_names(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            make_splitter("nonsense")
        assert all(name in str(excinfo.value) for name in SPLITTERS)

    def test_optimistic_protocols_warn_at_construction(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="geoq.evaluation.splitters"):
            make_splitter("within_subject_kfold", n_splits=4)
        assert "does not provide both evaluation guarantees" in caplog.text

    def test_loso_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="geoq.evaluation.splitters"):
            make_splitter("loso")
        assert caplog.text == ""


class TestValidation:
    """Misuse fails at the point of misuse."""

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("loso", {}),
            ("within_subject_chronological", {}),
            ("within_subject_kfold", {"n_splits": 3}),
        ],
    )
    def test_missing_groups_rejected(
        self, name: str, kwargs: dict, trials: np.ndarray
    ) -> None:
        """Falling back to a random split would misreport the protocol."""
        splitter = make_splitter(name, **kwargs)
        with pytest.raises(ValueError, match="requires `groups`"):
            list(splitter.split(trials))

    def test_groups_length_mismatch_rejected(
        self, trials: np.ndarray, subjects: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            list(LeaveOneSubjectOut().split(trials, groups=subjects[:-3]))

    def test_single_subject_rejected_for_loso(self, trials: np.ndarray) -> None:
        with pytest.raises(ValueError, match="at least two subjects"):
            list(LeaveOneSubjectOut().split(trials, groups=np.zeros(N_TRIALS)))

    def test_too_few_subjects_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Few folds means variance dominates any method difference."""
        trials = np.zeros((30, 2, 2))
        groups = np.repeat([0, 1, 2], 10)
        with caplog.at_level(logging.WARNING, logger="geoq.evaluation.splitters"):
            list(LeaveOneSubjectOut(min_subjects=4).split(trials, groups=groups))
        assert "below the recommended minimum" in caplog.text

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one trial"):
            list(LeaveOneSubjectOut().split(np.zeros((0, 2, 2)), groups=np.array([])))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_train_fraction_rejected(
        self, fraction: float, trials: np.ndarray, subjects: np.ndarray
    ) -> None:
        splitter = WithinSubjectChronological(train_fraction=fraction)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            list(splitter.split(trials, groups=subjects))

    @pytest.mark.parametrize("n_splits", [0, 1, -2])
    def test_invalid_kfold_count_rejected(
        self, n_splits: int, trials: np.ndarray, subjects: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            list(WithinSubjectKFold(n_splits=n_splits).split(trials, groups=subjects))

    def test_kfold_rejects_subject_with_too_few_trials(self) -> None:
        trials = np.zeros((12, 2, 2))
        groups = np.array([0] * 10 + [1] * 2)
        with pytest.raises(ValueError, match="fewer than n_splits"):
            list(WithinSubjectKFold(n_splits=5).split(trials, groups=groups))

    def test_chronological_rejects_single_trial_subject(self) -> None:
        trials = np.zeros((11, 2, 2))
        groups = np.array([0] * 10 + [1])
        with pytest.raises(ValueError, match="at least two"):
            list(WithinSubjectChronological().split(trials, groups=groups))
