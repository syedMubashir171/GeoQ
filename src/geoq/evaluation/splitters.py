"""Cross-validation splitters for EEG, with leakage made explicit.

Two mechanisms inflate BCI accuracy, and neither is visible in a results table.

**Subject leakage.** Training and testing on the same subject measures how well
a model fits one person's neurophysiology, not whether it generalises. Both
questions are legitimate, but they are different questions, and a paper that
reports the first while claiming the second is not reproducible by anyone
applying the model to a new participant.

**Temporal leakage.** EEG is strongly autocorrelated. Epochs recorded seconds
apart share drift, electrode impedance, muscle tone, and alertness. A shuffled
split therefore places near-duplicate trials on both sides of the partition,
and the classifier can score highly by recognising the recording conditions
rather than the task. This is the flaw in the QSVM-QNN paper that motivates
Paper 1.

Why leakage is a first-class object here
----------------------------------------
Paper 1 must *run* the leaky protocol deliberately, to measure the inflation.
So leakage cannot simply be prevented -- it has to be nameable, configurable,
and recorded.

Every splitter therefore carries a :class:`SplitterInfo` describing exactly
which guarantees it provides, and the experiment runner writes that record
alongside the results. A number in a table can then be traced to the protocol
that produced it without reading any code. The guarantees are reported as two
independent facts rather than one "leaky" flag, because they fail
independently: a within-subject chronological split is temporally disjoint but
not subject-independent, and calling it simply "leaky" or simply "clean" would
be wrong either way.

:class:`LeakyShuffleSplit` additionally refuses to be constructed without an
explicit acknowledgement argument. It exists to be used, once, on purpose.

References
----------
Varoquaux, G. et al. (2017). Assessing and tuning brain decoders:
    cross-validation, caveats, and guidelines. *NeuroImage*, 145, 166-179.
Saeb, S. et al. (2017). The need to approximate the use-case in clinical
    machine learning. *GigaScience*, 6(5).
Chaibub Neto, E. et al. (2019). Detecting the impact of subject
    characteristics on machine learning-based diagnostic applications.
    *npj Digital Medicine*, 2, 99.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SPLITTERS",
    "BaseSplitter",
    "LeakyShuffleSplit",
    "LeaveOneSubjectOut",
    "SplitterInfo",
    "WithinSubjectChronological",
    "WithinSubjectKFold",
    "make_splitter",
]

logger = logging.getLogger(__name__)

IndexArray = NDArray[np.intp]


@dataclass(frozen=True)
class SplitterInfo:
    """The evaluation guarantees a splitter provides.

    Written into every experiment's provenance record so that a reported
    number carries its protocol with it.

    Two independent booleans rather than one ``leaky`` flag, because the two
    failure modes are independent and conflating them loses information: a
    within-subject chronological split is temporally disjoint but not
    subject-independent, which is honest for a calibration study and dishonest
    if reported as generalisation.

    Attributes:
        name: Configuration name of the splitter.
        subject_independent: Whether no subject appears in both a training and
            its own test fold.
        temporally_disjoint: Whether training and test trials are separated in
            recording order, so autocorrelated neighbours cannot straddle the
            split.
        description: One-line summary for logs and tables.
        caveat: What this protocol cannot support, stated plainly. Empty when
            the protocol provides both guarantees.
    """

    name: str
    subject_independent: bool
    temporally_disjoint: bool
    description: str
    caveat: str = ""

    @property
    def is_optimistic(self) -> bool:
        """Whether results from this protocol are expected to be inflated."""
        return not (self.subject_independent and self.temporally_disjoint)

    def summary(self) -> str:
        """Return a one-line description suitable for a log or a table."""
        guarantees = (
            f"subject_independent={self.subject_independent}, "
            f"temporally_disjoint={self.temporally_disjoint}"
        )
        return f"{self.name} ({guarantees}): {self.description}"


class BaseSplitter(ABC):
    """Interface shared by every splitter, compatible with scikit-learn.

    Implements the ``split``/``get_n_splits`` protocol that
    :func:`sklearn.model_selection.cross_val_score` expects, so these objects
    can be passed as ``cv=`` without adaptation.
    """

    @property
    @abstractmethod
    def info(self) -> SplitterInfo:
        """The evaluation guarantees this splitter provides."""

    @abstractmethod
    def split(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield ``(train_indices, test_indices)`` for each fold."""

    @abstractmethod
    def get_n_splits(
        self,
        X: ArrayLike | None = None,  # noqa: N803
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of folds."""

    def __repr__(self) -> str:
        """Return a representation naming the class."""
        return f"{type(self).__name__}()"

    # ------------------------------------------------------------------ #
    # Shared validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _n_samples(x: ArrayLike) -> int:
        """Return the number of trials in ``x``.

        Args:
            x: Any array whose first axis indexes trials.

        Returns:
            The trial count.

        Raises:
            ValueError: If ``x`` is empty or not indexable by trial.
        """
        arr = np.asarray(x)
        if arr.ndim < 1 or arr.shape[0] == 0:
            raise ValueError(
                f"X must have at least one trial along its first axis, got "
                f"shape {arr.shape}."
            )
        return int(arr.shape[0])

    @staticmethod
    def _require_groups(groups: ArrayLike | None, n_samples: int) -> NDArray[Any]:
        """Validate the subject identifiers.

        Args:
            groups: Per-trial subject identifiers.
            n_samples: Expected length.

        Returns:
            The validated identifiers.

        Raises:
            ValueError: If ``groups`` is missing or the wrong length. Missing
                groups is fatal rather than a fallback to a random split,
                because a silent fallback would turn a subject-independent
                protocol into a subject-dependent one while still reporting
                the former.
        """
        if groups is None:
            raise ValueError(
                "This splitter requires `groups` giving each trial's subject. "
                "Without it a subject-independent split cannot be constructed, "
                "and falling back to a random split would report a "
                "generalisation result computed from a within-subject "
                "protocol."
            )
        arr = np.asarray(groups)
        if arr.shape != (n_samples,):
            raise ValueError(
                f"groups must have shape ({n_samples},) to match X, got {arr.shape}."
            )
        return arr


class LeaveOneSubjectOut(BaseSplitter):
    """Hold out one subject entirely per fold.

    The honest protocol for a generalisation claim, and the one this thesis
    reports under. Each fold trains on every other subject and tests on the
    held-out one, so the model has never seen the test participant's
    neurophysiology, electrode placement, or session conditions.

    Args:
        min_subjects: Minimum number of distinct subjects required. Below four,
            a leave-one-out estimate has so few folds that its variance
            dominates any difference between methods, and a comparison built
            on it cannot support a conclusion.

    Example:
        >>> import numpy as np
        >>> splitter = LeaveOneSubjectOut()
        >>> x = np.zeros((6, 2, 2))
        >>> subjects = np.array([1, 1, 2, 2, 3, 3])
        >>> splitter.get_n_splits(x, groups=subjects)
        3
    """

    def __init__(self, *, min_subjects: int = 4) -> None:
        """Store the minimum subject count.

        Args:
            min_subjects: Smallest acceptable number of distinct subjects.
        """
        self.min_subjects = min_subjects

    @property
    def info(self) -> SplitterInfo:
        """Both guarantees hold: no subject and no timepoint is shared."""
        return SplitterInfo(
            name="loso",
            subject_independent=True,
            temporally_disjoint=True,
            description=(
                "Leave-one-subject-out: each fold tests on a subject absent "
                "from training."
            ),
        )

    def _subjects(self, groups: NDArray[Any]) -> NDArray[Any]:
        """Return the sorted distinct subjects, checking there are enough."""
        subjects = np.unique(groups)
        if subjects.shape[0] < 2:
            raise ValueError(
                f"Leave-one-subject-out needs at least two subjects, found "
                f"{subjects.shape[0]}."
            )
        if subjects.shape[0] < self.min_subjects:
            logger.warning(
                "LeaveOneSubjectOut: only %d subjects, below the recommended "
                "minimum of %d. With this few folds the variance of the "
                "estimate is large enough to dominate any difference between "
                "methods; report the per-subject scores rather than the mean "
                "alone, and treat comparisons as indicative.",
                subjects.shape[0],
                self.min_subjects,
            )
        return subjects

    def split(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield one fold per subject.

        Args:
            X: Trials, indexed along the first axis.
            y: Ignored, present for API compatibility.
            groups: Per-trial subject identifiers. Required.

        Yields:
            ``(train_indices, test_indices)`` for each held-out subject.
        """
        n_samples = self._n_samples(X)
        subject_ids = self._require_groups(groups, n_samples)
        indices = np.arange(n_samples)

        for subject in self._subjects(subject_ids):
            test_mask = subject_ids == subject
            yield indices[~test_mask], indices[test_mask]

    def get_n_splits(
        self,
        X: ArrayLike | None = None,  # noqa: ARG002, N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of subjects.

        Args:
            X: Ignored.
            y: Ignored.
            groups: Per-trial subject identifiers. Required.

        Returns:
            The number of distinct subjects.
        """
        if groups is None:
            raise ValueError("LeaveOneSubjectOut.get_n_splits requires `groups`.")
        return int(np.unique(np.asarray(groups)).shape[0])

    def __repr__(self) -> str:
        """Return a representation including the configured minimum."""
        return f"LeaveOneSubjectOut(min_subjects={self.min_subjects})"


class WithinSubjectChronological(BaseSplitter):
    """Train on each subject's earlier trials, test on their later ones.

    The honest protocol for a calibration study: it answers "given some data
    from this person, can we decode the rest of their session?", which is what
    a practical BCI actually does. It is not a generalisation result and must
    not be reported as one.

    Trials are assumed to appear in recording order within each subject. That
    assumption is the whole point -- a chronological split is only temporally
    disjoint if the order is real -- so it is stated here and cannot be
    verified from the array alone.

    Args:
        train_fraction: Proportion of each subject's trials used for training.
        n_splits: Number of expanding-window folds. With ``n_splits > 1`` the
            training window grows and the test window advances, which
            characterises how performance depends on calibration length.

    Example:
        >>> import numpy as np
        >>> splitter = WithinSubjectChronological(train_fraction=0.5)
        >>> x = np.zeros((8, 2, 2))
        >>> subjects = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        >>> train, test = next(splitter.split(x, groups=subjects))
        >>> sorted(train.tolist())
        [0, 1, 4, 5]
    """

    def __init__(self, *, train_fraction: float = 0.5, n_splits: int = 1) -> None:
        """Store the split configuration.

        Args:
            train_fraction: Proportion of trials used for training.
            n_splits: Number of expanding-window folds.
        """
        self.train_fraction = train_fraction
        self.n_splits = n_splits

    @property
    def info(self) -> SplitterInfo:
        """Temporally disjoint, but the same subject appears on both sides."""
        return SplitterInfo(
            name="within_subject_chronological",
            subject_independent=False,
            temporally_disjoint=True,
            description=(
                "Within-subject chronological: trains on each subject's "
                "earlier trials and tests on their later ones."
            ),
            caveat=(
                "Measures calibration performance for a known participant, "
                "not generalisation to a new one. Do not describe results "
                "from this protocol as subject-independent."
            ),
        )

    def _validate(self) -> None:
        """Check the configuration.

        Raises:
            ValueError: If the fraction or split count is out of range.
        """
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(
                f"train_fraction must lie strictly between 0 and 1, got "
                f"{self.train_fraction!r}."
            )
        if not isinstance(self.n_splits, (int, np.integer)) or self.n_splits < 1:
            raise ValueError(
                f"n_splits must be a positive integer, got {self.n_splits!r}."
            )

    def split(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield expanding-window folds pooled across subjects.

        Each fold contains every subject's data on both sides, split at that
        subject's own chronological boundary. Pooling keeps the training set
        balanced across participants rather than letting one long recording
        dominate.

        Args:
            X: Trials in recording order within each subject.
            y: Ignored, present for API compatibility.
            groups: Per-trial subject identifiers. Required.

        Yields:
            ``(train_indices, test_indices)`` for each fold.
        """
        self._validate()
        n_samples = self._n_samples(X)
        subject_ids = self._require_groups(groups, n_samples)
        indices = np.arange(n_samples)

        for fold in range(self.n_splits):
            train_parts: list[IndexArray] = []
            test_parts: list[IndexArray] = []

            for subject in np.unique(subject_ids):
                subject_indices = indices[subject_ids == subject]
                n_subject = subject_indices.shape[0]
                if n_subject < 2:
                    raise ValueError(
                        f"Subject {subject!r} has {n_subject} trial(s); a "
                        f"chronological split needs at least two."
                    )

                # The training window grows with the fold index and the test
                # window is the block immediately after it, so no test trial
                # ever precedes a training trial for the same subject.
                base = self.train_fraction * n_subject
                remaining = n_subject - base
                cut = int(base + fold * remaining / self.n_splits)
                end = int(base + (fold + 1) * remaining / self.n_splits)
                cut = max(1, min(cut, n_subject - 1))
                end = max(cut + 1, min(end, n_subject))

                train_parts.append(subject_indices[:cut])
                test_parts.append(subject_indices[cut:end])

            yield np.concatenate(train_parts), np.concatenate(test_parts)

    def get_n_splits(
        self,
        X: ArrayLike | None = None,  # noqa: ARG002, N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,  # noqa: ARG002
    ) -> int:
        """Return the configured number of folds."""
        return int(self.n_splits)

    def __repr__(self) -> str:
        """Return a representation including the configuration."""
        return (
            f"WithinSubjectChronological(train_fraction={self.train_fraction}, "
            f"n_splits={self.n_splits})"
        )


class WithinSubjectKFold(BaseSplitter):
    """Contiguous k-fold within each subject, preserving recording order.

    Folds are contiguous blocks rather than shuffled samples, so only the
    trials at each block boundary have an autocorrelated neighbour on the
    other side. That is a far weaker leak than a shuffled split, but it is not
    zero, and the ``temporally_disjoint`` flag reports it as False for that
    reason: with ``k`` folds there are ``2k`` boundaries per subject where
    adjacent trials straddle the partition.

    Args:
        n_splits: Number of contiguous folds per subject.
    """

    def __init__(self, *, n_splits: int = 5) -> None:
        """Store the fold count.

        Args:
            n_splits: Number of contiguous folds per subject.
        """
        self.n_splits = n_splits

    @property
    def info(self) -> SplitterInfo:
        """Neither guarantee holds fully; the boundaries are the weak point."""
        return SplitterInfo(
            name="within_subject_kfold",
            subject_independent=False,
            temporally_disjoint=False,
            description=(
                "Within-subject contiguous k-fold, preserving recording order."
            ),
            caveat=(
                "Trials at each fold boundary have an autocorrelated "
                "neighbour on the other side, so a small optimistic bias "
                "remains. Use WithinSubjectChronological when a clean "
                "temporal separation is required."
            ),
        )

    def split(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield contiguous folds pooled across subjects.

        Args:
            X: Trials in recording order within each subject.
            y: Ignored, present for API compatibility.
            groups: Per-trial subject identifiers. Required.

        Yields:
            ``(train_indices, test_indices)`` for each fold.
        """
        if not isinstance(self.n_splits, (int, np.integer)) or self.n_splits < 2:
            raise ValueError(
                f"n_splits must be an integer of at least 2, got {self.n_splits!r}."
            )
        n_samples = self._n_samples(X)
        subject_ids = self._require_groups(groups, n_samples)
        indices = np.arange(n_samples)

        blocks: dict[Any, list[IndexArray]] = {}
        for subject in np.unique(subject_ids):
            subject_indices = indices[subject_ids == subject]
            if subject_indices.shape[0] < self.n_splits:
                raise ValueError(
                    f"Subject {subject!r} has {subject_indices.shape[0]} "
                    f"trials, fewer than n_splits={self.n_splits}."
                )
            blocks[subject] = np.array_split(subject_indices, self.n_splits)

        for fold in range(self.n_splits):
            test = np.concatenate([parts[fold] for parts in blocks.values()])
            train = np.concatenate(
                [
                    part
                    for parts in blocks.values()
                    for index, part in enumerate(parts)
                    if index != fold
                ]
            )
            yield np.sort(train), np.sort(test)

    def get_n_splits(
        self,
        X: ArrayLike | None = None,  # noqa: ARG002, N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,  # noqa: ARG002
    ) -> int:
        """Return the configured number of folds."""
        return int(self.n_splits)

    def __repr__(self) -> str:
        """Return a representation including the fold count."""
        return f"WithinSubjectKFold(n_splits={self.n_splits})"


class LeakyShuffleSplit(BaseSplitter):
    """Shuffled k-fold ignoring both subject and recording order.

    **This splitter is deliberately unsound.** It exists so that Paper 1 can
    measure how much accuracy the unsound protocol invents, by running it
    against an identical pipeline under leave-one-subject-out. It must never
    be used for a reported result.

    Two leaks operate at once. Trials from the same subject appear on both
    sides, so the model can key on that person's electrode montage and
    baseline neurophysiology. And because EEG is strongly autocorrelated,
    shuffling places temporally adjacent, near-duplicate epochs across the
    partition, letting a classifier score highly by recognising recording
    conditions rather than the task.

    The constructor requires ``acknowledge_leakage=True``. That argument
    carries no behaviour; it exists so that using this protocol is a decision
    someone made in writing, visible in the configuration file and the diff,
    rather than a default that nobody noticed.

    Args:
        n_splits: Number of shuffled folds.
        random_state: Seed. Required, not optional: the whole purpose is to
            quantify an inflation, and an unseeded estimate of it cannot be
            reproduced or compared across methods.
        acknowledge_leakage: Must be True.

    Example:
        >>> splitter = LeakyShuffleSplit(
        ...     n_splits=5, random_state=0, acknowledge_leakage=True
        ... )
        >>> splitter.info.is_optimistic
        True
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        random_state: int,
        acknowledge_leakage: bool = False,
    ) -> None:
        """Store the configuration, refusing construction without acknowledgement.

        Args:
            n_splits: Number of shuffled folds.
            random_state: Seed for the shuffle.
            acknowledge_leakage: Must be True.

        Raises:
            ValueError: If ``acknowledge_leakage`` is not True.
        """
        if acknowledge_leakage is not True:
            raise ValueError(
                "LeakyShuffleSplit produces inflated scores by design: it "
                "shuffles across subjects and across recording order, so "
                "autocorrelated near-duplicate trials land on both sides of "
                "the split. It exists only to quantify that inflation against "
                "an honest protocol. Pass acknowledge_leakage=True to confirm "
                "this is intentional, and never report its scores as a result."
            )
        self.n_splits = n_splits
        self.random_state = random_state
        self.acknowledge_leakage = acknowledge_leakage

    @property
    def info(self) -> SplitterInfo:
        """Neither guarantee holds. This is the point of the class."""
        return SplitterInfo(
            name="leaky_shuffle",
            subject_independent=False,
            temporally_disjoint=False,
            description=(
                "Shuffled k-fold ignoring subject and recording order. "
                "Deliberately unsound; for measuring inflation only."
            ),
            caveat=(
                "Scores from this protocol are not results. They are the "
                "control condition against which an honest protocol is "
                "compared."
            ),
        )

    def split(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,  # noqa: ARG002
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield shuffled folds, warning on every use.

        The warning fires each time rather than once per session. A single
        warning at import time scrolls away; one per fold is present in the
        log next to the inflated numbers it explains.

        Args:
            X: Trials, indexed along the first axis.
            y: Ignored, present for API compatibility.
            groups: Ignored. Deliberately so -- ignoring subject structure is
                the leak being measured.

        Yields:
            ``(train_indices, test_indices)`` for each fold.
        """
        if not isinstance(self.n_splits, (int, np.integer)) or self.n_splits < 2:
            raise ValueError(
                f"n_splits must be an integer of at least 2, got {self.n_splits!r}."
            )
        n_samples = self._n_samples(X)
        if n_samples < self.n_splits:
            raise ValueError(
                f"Cannot make {self.n_splits} folds from {n_samples} trials."
            )

        logger.warning(
            "LeakyShuffleSplit in use: subject identity and recording order "
            "are both ignored, so scores from this protocol are inflated by "
            "construction. Valid only as the control condition in a leakage "
            "comparison."
        )

        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(n_samples)
        for fold in np.array_split(shuffled, self.n_splits):
            test = np.sort(fold)
            train = np.sort(np.setdiff1d(np.arange(n_samples), test))
            yield train, test

    def get_n_splits(
        self,
        X: ArrayLike | None = None,  # noqa: ARG002, N803
        y: ArrayLike | None = None,  # noqa: ARG002
        groups: ArrayLike | None = None,  # noqa: ARG002
    ) -> int:
        """Return the configured number of folds."""
        return int(self.n_splits)

    def __repr__(self) -> str:
        """Return a representation that names the hazard."""
        return (
            f"LeakyShuffleSplit(n_splits={self.n_splits}, "
            f"random_state={self.random_state})  # DELIBERATELY LEAKY"
        )


SPLITTERS: dict[str, type[BaseSplitter]] = {
    "loso": LeaveOneSubjectOut,
    "within_subject_chronological": WithinSubjectChronological,
    "within_subject_kfold": WithinSubjectKFold,
    "leaky_shuffle": LeakyShuffleSplit,
}
"""Registry mapping configuration names to splitter classes."""


def make_splitter(name: str, **kwargs: Any) -> BaseSplitter:
    """Construct a splitter by its configuration name.

    The single entry point used by the experiment runner, so that the
    evaluation protocol is a value in a YAML file rather than a code path.

    Args:
        name: One of the keys of :data:`SPLITTERS`.
        **kwargs: Forwarded to the splitter's constructor.

    Returns:
        The constructed splitter.

    Raises:
        ValueError: If the name is unknown. Never falls back to a default: a
            typo in a protocol name must not quietly change the evaluation
            that a paper's central claim rests on.
    """
    try:
        splitter_class = SPLITTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown splitter {name!r}. Available: {sorted(SPLITTERS)}."
        ) from None

    splitter = splitter_class(**kwargs)
    info = splitter.info
    if info.is_optimistic:
        logger.warning(
            "Protocol %r does not provide both evaluation guarantees (%s). %s",
            info.name,
            f"subject_independent={info.subject_independent}, "
            f"temporally_disjoint={info.temporally_disjoint}",
            info.caveat or "Interpret its scores accordingly.",
        )
    return splitter
