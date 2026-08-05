"""The canonical EEG dataset container, and a synthetic generator.

Every loader in this framework -- MOABB, a local file, a simulation --
produces the same object: an :class:`EEGDataset`. Downstream code therefore
never learns where data came from, which is what lets a pipeline developed on
simulated trials run unchanged on BCI Competition IV 2a.

Validation is not optional
--------------------------
The container validates on construction, and the checks are chosen from the
failure modes that actually cost time in EEG work:

* **Shape and dtype**, because a transposed array of shape
  ``(n_trials, n_times, n_channels)`` has the right rank and the wrong
  meaning, and produces a covariance matrix of the wrong size several layers
  later.
* **Non-finite samples**, which usually mean an epoch overlapped a recording
  gap.
* **Group and label alignment**, because a subject vector that has drifted out
  of step with the trials silently destroys every subject-independent
  guarantee the evaluation layer provides, while leaving all of its checks
  passing.
* **Samples per channel**, warned rather than raised. The geometry layer
  measured the affine-invariant metric's error at roughly ``eps * kappa ** 2``,
  and short epochs on a wide montage put ``kappa`` in the thousands. That is a
  fact about the data worth surfacing at load time, not a defect to reject.

Why the synthetic generator lives here
--------------------------------------
Simulated data is not a stand-in for a dataset that has not arrived; it is a
tool with a distinct job. A generator with known ground truth is the only way
to check that an evaluation protocol reports what it should -- there is no way
to verify that a leaky split inflates accuracy using data whose true effect
size is unknown. Every leakage and calibration measurement in this framework
rests on it, so it is a first-class, seeded, documented component rather than
a fixture buried in a test file.

References
----------
Tangermann, M. et al. (2012). Review of the BCI Competition IV.
    *Frontiers in Neuroscience*, 6, 55.
Jayaram, V., & Barachant, A. (2018). MOABB: trustworthy algorithm
    benchmarking for BCIs. *Journal of Neural Engineering*, 15(6), 066011.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DATASETS",
    "MIN_SAMPLES_PER_CHANNEL",
    "EEGDataset",
    "load_dataset",
    "make_synthetic_eeg",
    "register_dataset",
]

logger = logging.getLogger(__name__)

MIN_SAMPLES_PER_CHANNEL: float = 2.0
"""Ratio below which a conditioning warning is emitted at load time.

Full rank needs only ``n_times > n_channels``. Usable conditioning needs far
more: measured on spatially correlated data, the sample covariance reaches a
condition number near ``3e3`` at a ratio of 2 and stays near ``1e3`` even at
45. Since affine-invariant distances carry an error near ``eps * kappa ** 2``,
the ratio is worth knowing before an experiment starts rather than after its
results look strange.
"""


@dataclass(frozen=True)
class EEGDataset:
    """Epoched EEG with the metadata every downstream layer needs.

    Attributes:
        epochs: Trials of shape ``(n_trials, n_channels, n_times)``,
            channels-first as in MNE.
        labels: Class labels of shape ``(n_trials,)``.
        subjects: Subject identifier per trial, shape ``(n_trials,)``. Used as
            ``groups`` by every subject-aware splitter.
        sessions: Session identifier per trial, shape ``(n_trials,)``. Empty
            when the dataset has a single session.
        sampling_rate: Effective sampling rate in hertz, after any resampling.
        channel_names: Channel labels, in the order of the second axis.
        name: Dataset identifier, carried into the provenance record.
        metadata: Anything else worth recording, such as the filter band.
    """

    epochs: NDArray[np.float64]
    labels: NDArray[Any]
    subjects: NDArray[Any]
    sampling_rate: float
    name: str = "unnamed"
    sessions: NDArray[Any] = field(default_factory=lambda: np.array([], dtype=object))
    channel_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes, alignment, and numerical sanity.

        Raises:
            ValueError: If any invariant is violated.
        """
        epochs = np.asarray(self.epochs, dtype=np.float64)
        object.__setattr__(self, "epochs", epochs)

        if epochs.ndim != 3:
            raise ValueError(
                f"epochs must have shape (n_trials, n_channels, n_times), got "
                f"{epochs.shape}. Channels come before time, matching MNE; a "
                f"(n_trials, n_times, n_channels) array has the right rank and "
                f"the wrong meaning, and fails much later as a covariance of "
                f"the wrong size."
            )
        n_trials, n_channels, n_times = epochs.shape
        if n_trials == 0:
            raise ValueError("epochs contains no trials.")
        if n_channels == 0 or n_times == 0:
            raise ValueError(
                f"epochs has an empty axis: {n_channels} channels, {n_times} samples."
            )
        if not np.isfinite(epochs).all():
            count = int((~np.isfinite(epochs)).sum())
            trial = int(np.argwhere(~np.isfinite(epochs))[0][0])
            raise ValueError(
                f"epochs contains {count} non-finite sample(s), first in trial "
                f"{trial}. This usually means an epoch overlapped a recording "
                f"gap, or an interpolation step failed."
            )

        for field_name in ("labels", "subjects"):
            values = np.asarray(getattr(self, field_name))
            object.__setattr__(self, field_name, values)
            if values.shape != (n_trials,):
                raise ValueError(
                    f"{field_name} must have shape ({n_trials},) to align with "
                    f"epochs, got {values.shape}. A vector out of step with "
                    f"the trials leaves every evaluation check passing while "
                    f"silently destroying the guarantee it was checking."
                )

        sessions = np.asarray(self.sessions)
        if sessions.size and sessions.shape != (n_trials,):
            raise ValueError(
                f"sessions must be empty or have shape ({n_trials},), got "
                f"{sessions.shape}."
            )
        object.__setattr__(self, "sessions", sessions)

        if np.unique(self.labels).size < 2:
            raise ValueError(
                f"Only one class present: {np.unique(self.labels).tolist()}. "
                f"A single-class dataset cannot be used for classification."
            )
        if self.sampling_rate <= 0:
            raise ValueError(
                f"sampling_rate must be positive, got {self.sampling_rate}."
            )
        if self.channel_names and len(self.channel_names) != n_channels:
            raise ValueError(
                f"channel_names has {len(self.channel_names)} entries but "
                f"epochs has {n_channels} channels."
            )

        self._warn_on_conditioning_risk(n_channels, n_times)

    @staticmethod
    def _warn_on_conditioning_risk(n_channels: int, n_times: int) -> None:
        """Warn when epochs are short relative to the channel count.

        Args:
            n_channels: Channel count.
            n_times: Samples per epoch.
        """
        ratio = n_times / n_channels
        if ratio <= 1.0:
            #  A transposed array presents here rather than at the rank check
            #  in Covariances, because the container is the first thing that
            #  sees it. Time samples vastly outnumber channels, so the
            #  transposed array looks like an enormous montage recorded for a
            #  handful of samples -- which is the tell.
            hint = (
                f" The axis sizes also look suspicious: {n_channels} "
                f"'channels' and {n_times} 'times' suggests the array is "
                f"(n_trials, n_times, n_channels) and needs transposing to "
                f"the channels-first convention MNE uses."
                if n_channels > 4 * max(n_times, 1)
                else ""
            )
            raise ValueError(
                f"n_times ({n_times}) does not exceed n_channels "
                f"({n_channels}), so every sample covariance is singular and "
                f"the SPD manifold is undefined. Lengthen the epoch, drop "
                f"channels, or raise the sampling rate.{hint}"
            )
        if ratio < MIN_SAMPLES_PER_CHANNEL:
            logger.warning(
                "Dataset has n_times/n_channels = %.1f, below %.1f. Sample "
                "covariances will be full rank but severely ill-conditioned, "
                "and affine-invariant distances carry an error near "
                "eps * kappa ** 2. Prefer a shrinkage covariance estimator.",
                ratio,
                MIN_SAMPLES_PER_CHANNEL,
            )

    # ------------------------------------------------------------------ #
    # Shape
    # ------------------------------------------------------------------ #

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return int(self.epochs.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.epochs.shape[1])

    @property
    def n_times(self) -> int:
        """Samples per epoch."""
        return int(self.epochs.shape[2])

    @property
    def n_subjects(self) -> int:
        """Number of distinct subjects."""
        return int(np.unique(self.subjects).size)

    @property
    def classes(self) -> NDArray[Any]:
        """Sorted distinct labels."""
        return np.unique(self.labels)

    @property
    def duration(self) -> float:
        """Epoch length in seconds."""
        return self.n_times / self.sampling_rate

    @property
    def samples_per_channel(self) -> float:
        """Ratio governing covariance conditioning."""
        return self.n_times / self.n_channels

    @property
    def class_balance(self) -> dict[Any, float]:
        """Label to proportion."""
        values, counts = np.unique(self.labels, return_counts=True)
        proportions = counts / counts.sum()
        return dict(zip(values.tolist(), proportions.tolist(), strict=True))

    @property
    def chance_accuracy(self) -> float:
        """Accuracy of always predicting the majority class."""
        return max(self.class_balance.values())

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def subset(self, mask: NDArray[np.bool_] | Sequence[int]) -> EEGDataset:
        """Return a new dataset containing the selected trials.

        Args:
            mask: Boolean mask or integer indices.

        Returns:
            The subset, with all metadata preserved.
        """
        index = np.asarray(mask)
        return EEGDataset(
            epochs=self.epochs[index],
            labels=self.labels[index],
            subjects=self.subjects[index],
            sessions=self.sessions[index] if self.sessions.size else self.sessions,
            sampling_rate=self.sampling_rate,
            name=self.name,
            channel_names=self.channel_names,
            metadata=self.metadata,
        )

    def select_subjects(self, subjects: Sequence[Any]) -> EEGDataset:
        """Return the trials belonging to the given subjects.

        Args:
            subjects: Subject identifiers to keep.

        Returns:
            The subset.

        Raises:
            ValueError: If any requested subject is absent. Silently returning
                fewer subjects than asked for would change the evaluation
                without changing the configuration.
        """
        requested = set(subjects)
        available = set(np.unique(self.subjects).tolist())
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                f"Subjects {missing} are not present in {self.name!r}. "
                f"Available: {sorted(available)}."
            )
        return self.subset(np.isin(self.subjects, list(requested)))

    def by_subject(self) -> Iterator[tuple[Any, EEGDataset]]:
        """Iterate over per-subject datasets, in sorted subject order.

        Yields:
            Pairs of subject identifier and that subject's trials.
        """
        for subject in np.unique(self.subjects):
            yield subject, self.subset(self.subjects == subject)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        """Return a record for the experiment's provenance file.

        Includes the chance level and the samples-per-channel ratio, because
        an accuracy without the first is uninterpretable and a geodesic
        distance without the second has unknown precision.
        """
        return {
            "dataset": self.name,
            "n_trials": self.n_trials,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_subjects": self.n_subjects,
            "n_classes": int(self.classes.size),
            "sampling_rate": self.sampling_rate,
            "epoch_duration_s": self.duration,
            "samples_per_channel": self.samples_per_channel,
            "chance_accuracy": self.chance_accuracy,
            "class_balance": {
                str(key): value for key, value in self.class_balance.items()
            },
            "trials_per_subject": {
                str(subject): int(np.sum(self.subjects == subject))
                for subject in np.unique(self.subjects)
            },
            **self.metadata,
        }

    def __len__(self) -> int:
        """Return the number of trials."""
        return self.n_trials

    def __repr__(self) -> str:
        """Return a one-line description."""
        return (
            f"EEGDataset({self.name}: {self.n_trials} trials, "
            f"{self.n_channels}ch x {self.n_times} samples @ "
            f"{self.sampling_rate:g}Hz, {self.n_subjects} subjects, "
            f"{self.classes.size} classes)"
        )


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


def make_synthetic_eeg(
    *,
    n_subjects: int = 9,
    n_trials_per_subject: int = 72,
    n_channels: int = 22,
    n_times: int = 500,
    sampling_rate: float = 250.0,
    n_classes: int = 2,
    task_effect: float = 0.35,
    subject_variability: float = 1.0,
    drift: float = 0.15,
    source_decay: float = 3.0,
    seed: int = 0,
    name: str = "synthetic",
) -> EEGDataset:
    """Generate EEG with the three structures that matter for evaluation.

    Defaults match BCI Competition IV 2a: nine subjects, 22 channels, two
    seconds at 250 Hz.

    Three ingredients, each present because omitting it makes a class of
    evaluation bug invisible:

    * **Subject identity.** Each participant's trials cluster around their own
      spatial covariance structure, which is what a leaky split lets a model
      exploit. Without it, a shuffled split and a subject-independent one give
      the same answer and the leak cannot be measured.
    * **Temporal drift.** Consecutive trials within a subject share a slowly
      changing state, reproducing the autocorrelation that makes a shuffled
      split unsound even for a single participant.
    * **Spatial correlation.** Sources are mixed through a basis with an
      exponentially decaying spectrum, as volume conduction does. White noise
      instead would give well-conditioned covariances and hide every
      conditioning problem real EEG has.

    Args:
        n_subjects: Number of participants.
        n_trials_per_subject: Trials each, split evenly across classes.
        n_channels: Channel count.
        n_times: Samples per epoch.
        sampling_rate: Sampling rate in hertz.
        n_classes: Number of classes.
        task_effect: Strength of the class-dependent component. Zero gives a
            true null, which is what a false-positive rate is measured on.
        subject_variability: Scale of between-subject differences.
        drift: Innovation weight of the AR(1) process governing within-subject
            drift, so the autoregressive coefficient is ``1 - drift``.
            *Smaller* values mean slower change and therefore stronger
            autocorrelation between neighbouring trials; ``0.05`` gives
            adjacent trials roughly half the geodesic separation of trials
            thirty apart. The name describes how fast the state drifts, not
            how much structure remains.
        source_decay: Decay constant of the source spectrum. Smaller values
            give worse-conditioned covariances.
        seed: Random seed.
        name: Dataset name.

    Returns:
        The generated dataset.

    Raises:
        ValueError: If any argument is out of range.
    """
    if n_subjects < 1 or n_trials_per_subject < n_classes:
        raise ValueError(
            f"Need at least one subject and at least {n_classes} trials each, "
            f"got {n_subjects} and {n_trials_per_subject}."
        )
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}.")
    if task_effect < 0 or drift < 0 or source_decay <= 0:
        raise ValueError(
            "task_effect and drift must be non-negative and source_decay positive."
        )

    rng = np.random.default_rng(seed)

    # Volume conduction: a fixed orthogonal basis with a decaying spectrum,
    # shared across subjects because head geometry differs in degree, not in
    # kind.
    basis, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
    spectrum = np.exp(-np.arange(n_channels) / source_decay)
    mixing = basis @ np.diag(np.sqrt(spectrum))

    # Each class has its own spatial pattern, as motor imagery classes do.
    class_patterns = [
        rng.standard_normal((n_channels, n_channels)) / np.sqrt(n_channels)
        for _ in range(n_classes)
    ]

    epochs, labels, subjects, sessions = [], [], [], []

    for subject in range(1, n_subjects + 1):
        subject_shift = (
            subject_variability
            * rng.standard_normal((n_channels, n_channels))
            / np.sqrt(n_channels)
        )
        state = np.zeros((n_channels, n_channels))

        #  Labels alternate rather than being drawn at random, so every fold
        #  of every protocol is balanced. An imbalanced simulation would make
        #  accuracy and balanced accuracy diverge for reasons unrelated to the
        #  property under test.
        order = np.tile(np.arange(n_classes), n_trials_per_subject // n_classes)
        order = np.concatenate([order, np.arange(n_trials_per_subject - order.size)])

        for index, label in enumerate(order):
            state = (1 - drift) * state + drift * rng.standard_normal(
                (n_channels, n_channels)
            ) / np.sqrt(n_channels)
            operator = (
                np.eye(n_channels)
                + subject_shift
                + state
                + task_effect * class_patterns[label]
            )
            sources = rng.standard_normal((n_channels, n_times))
            epochs.append(operator @ mixing @ sources)
            labels.append(int(label))
            subjects.append(subject)
            sessions.append(0 if index < n_trials_per_subject // 2 else 1)

    return EEGDataset(
        epochs=np.stack(epochs),
        labels=np.array(labels),
        subjects=np.array(subjects),
        sessions=np.array(sessions),
        sampling_rate=sampling_rate,
        name=name,
        channel_names=tuple(f"CH{index + 1:02d}" for index in range(n_channels)),
        metadata={
            "synthetic": True,
            "seed": seed,
            "task_effect": task_effect,
            "drift": drift,
            "source_decay": source_decay,
        },
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

DATASETS: dict[str, Callable[..., EEGDataset]] = {}
"""Registry mapping configuration names to loader callables."""


def register_dataset(name: str) -> Callable[[Callable[..., EEGDataset]], Any]:
    """Register a loader under a configuration name.

    Args:
        name: The name used in a configuration file.

    Returns:
        A decorator.

    Raises:
        ValueError: If the name is already registered. Silently replacing a
            loader would make a configuration file mean something different
            depending on import order.
    """

    def decorator(loader: Callable[..., EEGDataset]) -> Callable[..., EEGDataset]:
        if name in DATASETS:
            raise ValueError(
                f"Dataset {name!r} is already registered to {DATASETS[name].__name__}."
            )
        DATASETS[name] = loader
        return loader

    return decorator


@register_dataset("synthetic")
def _load_synthetic(**kwargs: Any) -> EEGDataset:
    """Loader entry point for the synthetic generator."""
    return make_synthetic_eeg(**kwargs)


def load_dataset(name: str, **kwargs: Any) -> EEGDataset:
    """Load a dataset by its configuration name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the loader.

    Returns:
        The loaded dataset.

    Raises:
        ValueError: If the name is unknown. Never falls back to a default: a
            typo in a dataset name must not quietly change which recordings an
            experiment ran on.
    """
    try:
        loader = DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Registered: {sorted(DATASETS)}. "
            f"Adapters register themselves on import, so a missing name may "
            f"mean the adapter module has not been imported."
        ) from None

    dataset = loader(**kwargs)
    logger.info("Loaded %r", dataset)
    return dataset
