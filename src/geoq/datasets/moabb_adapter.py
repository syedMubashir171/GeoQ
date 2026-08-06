"""Adapter from MOABB datasets to :class:`~geoq.datasets.base.EEGDataset`.

MOABB is the standard benchmark harness for BCI, and using it rather than
hand-rolled loaders is what makes results here comparable with the published
literature. This module is the bridge: it fetches, validates, caches, and
hands back the framework's canonical container.

Expectations are declared, not discovered
-----------------------------------------
Each supported dataset carries a :class:`MOABBSpec` stating the channel count,
sampling rate, subject list and classes that the published descriptions
specify. Every load is checked against it.

This matters more than it first appears. A silent upstream change -- a
re-uploaded recording, a MOABB release that alters default channel selection,
a paradigm whose defaults shift -- produces a dataset that loads cleanly and
is not the one a previous result was computed on. Comparing against a number
from six months ago would then be comparing across an invisible change. The
spec turns that into an error at load time.

Caching is of the *preprocessed epochs*
---------------------------------------
MOABB already caches raw downloads through MNE, so a second run does not
re-download. What it does repeat is filtering, epoching and resampling, which
for nine subjects of BCI Competition IV 2a is minutes rather than seconds --
long enough that an interrupted Colab session pays it again on every restart.

The cache key is a hash of every parameter that affects the output, computed
the same way as the experiment hash: canonical JSON, sorted keys. Change the
filter band and the key changes, so a stale cache cannot be served for a
different preprocessing. That is the failure this design exists to prevent,
and it is why the key is not simply the dataset name.

Labels stay as strings
----------------------
MOABB returns ``"left_hand"``, not ``0``. They are kept that way. An integer
label is one renumbering away from silently swapping two classes, and a
confusion matrix with meaningful row names is worth more than the handful of
bytes saved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geoq.datasets.base import EEGDataset, register_dataset

__all__ = [
    "MOABB_SPECS",
    "MOABBSpec",
    "available_datasets",
    "fetch_moabb_epochs",
    "load_moabb",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MOABBSpec:
    """What a supported dataset is expected to contain.

    Attributes:
        moabb_class: Class name in :mod:`moabb.datasets`.
        paradigm: Class name in :mod:`moabb.paradigms`.
        n_subjects: Expected number of participants.
        n_channels: Expected EEG channel count after the paradigm's selection.
        sampling_rate: Native sampling rate in hertz.
        classes: Expected class labels.
        default_window: Default ``(tmin, tmax)`` relative to the cue, in
            seconds, taken from the paradigm's published description.
        citation: Reference for the methods section.
    """

    moabb_class: str
    paradigm: str
    n_subjects: int
    n_channels: int
    sampling_rate: float
    classes: tuple[str, ...]
    default_window: tuple[float, float]
    citation: str


MOABB_SPECS: dict[str, MOABBSpec] = {
    #  Two entries for the same recordings, differing only in paradigm. The
    #  four-class variant is the full dataset; the left/right variant is what
    #  most published Riemannian BCI numbers are computed on, so comparing
    #  against the literature requires knowing which one produced a result.
    #  Separate names make that visible in the configuration file.
    "bci_iv_2a": MOABBSpec(
        moabb_class="BNCI2014_001",
        paradigm="MotorImagery",
        n_subjects=9,
        n_channels=22,
        sampling_rate=250.0,
        classes=("left_hand", "right_hand", "feet", "tongue"),
        default_window=(0.5, 2.5),
        citation="Tangermann et al. (2012), Review of the BCI Competition IV",
    ),
    "bci_iv_2a_lr": MOABBSpec(
        moabb_class="BNCI2014_001",
        paradigm="LeftRightImagery",
        n_subjects=9,
        n_channels=22,
        sampling_rate=250.0,
        classes=("left_hand", "right_hand"),
        default_window=(0.5, 2.5),
        citation="Tangermann et al. (2012), Review of the BCI Competition IV",
    ),
    "bci_iv_2b": MOABBSpec(
        moabb_class="BNCI2014_004",
        paradigm="LeftRightImagery",
        n_subjects=9,
        n_channels=3,
        sampling_rate=250.0,
        classes=("left_hand", "right_hand"),
        default_window=(0.5, 2.5),
        citation="Leeb et al. (2007), BCI Competition IV dataset 2b",
    ),
    "physionet_mi": MOABBSpec(
        moabb_class="PhysionetMI",
        paradigm="LeftRightImagery",
        n_subjects=109,
        n_channels=64,
        sampling_rate=160.0,
        classes=("left_hand", "right_hand"),
        default_window=(0.0, 3.0),
        citation="Schalk et al. (2004), BCI2000",
    ),
}
"""Datasets this adapter supports, with their expected contents."""


def available_datasets() -> dict[str, MOABBSpec]:
    """Return the supported datasets and their specifications."""
    return dict(MOABB_SPECS)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_moabb_epochs(
    spec: MOABBSpec,
    *,
    subjects: list[int] | None,
    tmin: float,
    tmax: float,
    low_freq: float,
    high_freq: float,
    resample: float | None,
    channels: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Download and epoch a dataset through MOABB.

    Isolated as a single function so that everything above it -- validation,
    caching, container construction -- can be tested without a network. The
    logic that decides whether a result is trustworthy should not be
    verifiable only when a 1.6 GB download succeeds.

    Args:
        spec: The dataset specification.
        subjects: Subjects to load, or None for all.
        tmin: Epoch start relative to the cue, in seconds.
        tmax: Epoch end relative to the cue, in seconds.
        low_freq: High-pass cutoff in hertz.
        high_freq: Low-pass cutoff in hertz.
        resample: Target rate in hertz, or None.
        channels: Channels to keep, or None for the paradigm's default.

    Returns:
        Tuple of epochs, labels, and MOABB's metadata table.

    Raises:
        ImportError: If MOABB is not installed.
    """
    try:
        import moabb.datasets as moabb_datasets
        import moabb.paradigms as moabb_paradigms
    except ImportError as error:  # pragma: no cover - exercised by environment
        raise ImportError(
            "Loading a MOABB dataset requires the 'eeg' extra: pip install -e '.[eeg]'"
        ) from error

    dataset = getattr(moabb_datasets, spec.moabb_class)()
    paradigm = getattr(moabb_paradigms, spec.paradigm)(
        fmin=low_freq,
        fmax=high_freq,
        tmin=tmin,
        tmax=tmax,
        resample=resample,
        channels=channels,
    )
    return paradigm.get_data(dataset=dataset, subjects=subjects)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def _cache_key(name: str, parameters: dict[str, Any]) -> str:
    """Return a stable hash of everything affecting the loaded epochs.

    Canonical JSON with sorted keys, matching
    :meth:`geoq.runtime.config.ExperimentConfig.experiment_hash`, so the key is
    identical across processes and Python versions.

    Every preprocessing parameter is included. Keying on the dataset name
    alone would serve an 8-30 Hz cache to a request for 4-8 Hz, producing
    results that silently answer a different question.

    Args:
        name: Dataset name.
        parameters: Preprocessing parameters.

    Returns:
        A sixteen-character hex digest.
    """
    canonical = json.dumps(
        {"dataset": name, **parameters},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, name: str, key: str) -> Path:
    """Return the cache file path for a dataset and key."""
    return cache_dir / f"{name}_{key}.npz"


def _pickle_free(values: np.ndarray) -> np.ndarray:
    """Return an array storable without pickling.

    Object-dtype arrays -- which pandas produces for string identifier columns
    -- cannot be saved unless pickling is enabled. Casting them to unicode
    keeps the archive readable with ``allow_pickle=False``.

    Args:
        values: The array to convert.

    Returns:
        The array, cast to unicode if it had object dtype.
    """
    return values.astype(str) if values.dtype == object else values


def _write_cache(path: Path, dataset: EEGDataset) -> None:
    """Write a dataset to the cache atomically.

    The same temporary-file-then-rename discipline as the shard store: a
    cache file truncated by a disconnect would load as a valid array of the
    wrong length, which is worse than no cache at all.

    Args:
        path: Destination.
        dataset: The dataset to cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    #  The archive is written through an open handle rather than by filename.
    #  np.savez_compressed appends '.npz' to any name that lacks it, so a
    #  temporary called 'x.npz.tmp' is silently written to 'x.npz.tmp.npz'
    #  and the rename that follows finds nothing. Passing a handle also lets
    #  the temporary carry a unique name, so parallel loaders cannot collide.
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".tmp-{os.getpid()}-", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            #  Object arrays require pickling, so identifier columns coming
            #  from pandas are cast to unicode. Subject and session values are
            #  labels rather than quantities, and a canonical string form also
            #  keeps them comparable across datasets that number subjects
            #  differently.
            np.savez_compressed(
                stream,
                epochs=dataset.epochs,
                labels=_pickle_free(dataset.labels),
                subjects=_pickle_free(dataset.subjects),
                sessions=_pickle_free(dataset.sessions),
                sampling_rate=np.array(dataset.sampling_rate),
                sidecar=np.array(
                    json.dumps(
                        {
                            "channel_names": list(dataset.channel_names),
                            "metadata": dataset.metadata,
                        },
                        default=str,
                    )
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_cache(path: Path, name: str) -> EEGDataset | None:
    """Read a cached dataset, returning None if it is unusable.

    A damaged cache is a recoverable condition: the data can always be
    recomputed. Failing the run instead would make an interrupted download
    permanently fatal.

    Args:
        path: Cache file.
        name: Dataset name.

    Returns:
        The dataset, or None.
    """
    try:
        #  allow_pickle stays off. The cache is derived data that can always be
        #  regenerated, so there is no reason for a public framework to
        #  execute arbitrary objects out of a file on a shared Drive folder.
        #  Everything is therefore stored in pickle-free forms: numeric and
        #  unicode arrays, plus one JSON string.
        with np.load(path, allow_pickle=False) as archive:
            sidecar = json.loads(str(archive["sidecar"]))
            return EEGDataset(
                epochs=archive["epochs"],
                labels=archive["labels"],
                subjects=archive["subjects"],
                sessions=archive["sessions"],
                sampling_rate=float(archive["sampling_rate"]),
                name=name,
                channel_names=tuple(sidecar["channel_names"]),
                metadata=sidecar["metadata"],
            )
    except Exception as error:
        #  Deliberately broad. NumPy signals a damaged archive through several
        #  unrelated exception types -- UnpicklingError, BadZipFile, ValueError
        #  -- and enumerating them means a future NumPy release can turn a
        #  recoverable condition into a crashed overnight run. Every failure to
        #  read a cache is recoverable by recomputing.
        logger.warning(
            "Cache file %s could not be read (%s: %s); recomputing from source.",
            path.name,
            type(error).__name__,
            error,
        )
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_against_spec(
    dataset: EEGDataset,
    spec: MOABBSpec,
    name: str,
    *,
    subjects: list[int] | None,
    channels: list[str] | None,
    resample: float | None,
    window: tuple[float, float],
) -> None:
    """Check a loaded dataset against its published description.

    Args:
        dataset: The loaded dataset.
        spec: Its specification.
        name: Dataset name.
        subjects: Subjects requested, or None for all.
        channels: Channels requested, or None for the paradigm default.
        resample: Requested sampling rate, or None.
        window: The requested ``(tmin, tmax)``, used to check the epoch length.

    Raises:
        ValueError: On a mismatch that indicates the data is not what the
            specification describes.
    """
    expected_rate = resample if resample is not None else spec.sampling_rate

    #  The epoch length is what actually reveals a rate mismatch, since the
    #  rate itself is now taken from the request rather than measured. MNE
    #  includes both endpoints, so the expected count is
    #  round(duration * rate) + 1; a tolerance of one sample absorbs MNE's
    #  rounding at awkward rate-and-window combinations without admitting a
    #  genuinely different sampling rate, which would be wrong by hundreds.
    duration = window[1] - window[0]
    expected_samples = round(duration * expected_rate) + 1
    if abs(dataset.n_times - expected_samples) > 1:
        implied_rate = (dataset.n_times - 1) / duration
        raise ValueError(
            f"{name!r} returned {dataset.n_times} samples per epoch, but a "
            f"{duration:g} s window at {expected_rate} Hz should give about "
            f"{expected_samples} (MNE includes both endpoints). The data "
            f"implies roughly {implied_rate:.1f} Hz, so the epochs are not "
            f"what the specification describes and results from them are not "
            f"comparable with the literature."
        )

    if channels is None and dataset.n_channels != spec.n_channels:
        raise ValueError(
            f"{name!r} loaded with {dataset.n_channels} channels but "
            f"{spec.n_channels} were expected. MOABB may have changed its "
            f"default channel selection; pin the channels explicitly, or "
            f"update MOABB_SPECS after confirming which is correct."
        )

    expected_subjects = spec.n_subjects if subjects is None else len(set(subjects))
    if dataset.n_subjects != expected_subjects:
        raise ValueError(
            f"{name!r} loaded {dataset.n_subjects} subjects but "
            f"{expected_subjects} were expected. A subject that failed to "
            f"download is silently missing from the analysis otherwise, and "
            f"leave-one-subject-out would report a mean over fewer folds "
            f"than the methods section claims."
        )

    unexpected = set(dataset.classes.tolist()) - set(spec.classes)
    if unexpected:
        raise ValueError(
            f"{name!r} contains unexpected class labels {sorted(unexpected)}. "
            f"Expected a subset of {list(spec.classes)}."
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_moabb(
    name: str,
    *,
    subjects: list[int] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    low_freq: float = 8.0,
    high_freq: float = 30.0,
    resample: float | None = None,
    channels: list[str] | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> EEGDataset:
    """Load a MOABB dataset as an :class:`EEGDataset`.

    Args:
        name: Key of :data:`MOABB_SPECS`.
        subjects: Subjects to load, or None for all.
        tmin: Epoch start relative to the cue. Defaults to the spec's window.
        tmax: Epoch end relative to the cue. Defaults to the spec's window.
        low_freq: High-pass cutoff in hertz.
        high_freq: Low-pass cutoff in hertz.
        resample: Target sampling rate, or None to keep the native rate.
        channels: Channels to keep, or None for the paradigm's default.
        cache_dir: Directory for the preprocessed-epoch cache. Caching is
            skipped when None.
        use_cache: Whether to read and write the cache.

    Returns:
        The loaded dataset.

    Raises:
        ValueError: If the name is unknown, the window or band is invalid, or
            the loaded data disagrees with its specification.
    """
    try:
        spec = MOABB_SPECS[name]
    except KeyError:
        raise ValueError(
            f"Unknown MOABB dataset {name!r}. Supported: {sorted(MOABB_SPECS)}."
        ) from None

    window_start = spec.default_window[0] if tmin is None else tmin
    window_end = spec.default_window[1] if tmax is None else tmax

    if window_end <= window_start:
        raise ValueError(f"tmax ({window_end}) must exceed tmin ({window_start}).")
    if high_freq <= low_freq:
        raise ValueError(f"high_freq ({high_freq}) must exceed low_freq ({low_freq}).")
    effective_rate = resample if resample is not None else spec.sampling_rate
    if effective_rate <= 2 * high_freq:
        raise ValueError(
            f"The effective sampling rate ({effective_rate} Hz) does not "
            f"exceed twice high_freq ({2 * high_freq} Hz), so the passband "
            f"aliases into the signal. The result looks like ordinary EEG and "
            f"decodes like noise."
        )

    parameters: dict[str, Any] = {
        "subjects": sorted(subjects) if subjects else None,
        "tmin": window_start,
        "tmax": window_end,
        "low_freq": low_freq,
        "high_freq": high_freq,
        "resample": resample,
        "channels": sorted(channels) if channels else None,
        "moabb_class": spec.moabb_class,
        "paradigm": spec.paradigm,
    }
    key = _cache_key(name, parameters)

    cache_file: Path | None = None
    if cache_dir is not None and use_cache:
        cache_file = _cache_path(Path(cache_dir), name, key)
        if cache_file.is_file():
            cached = _read_cache(cache_file, name)
            if cached is not None:
                logger.info("Loaded %r from cache %s", name, cache_file.name)
                return cached

    logger.info(
        "Fetching %r through MOABB (%s / %s). The first call downloads and "
        "preprocesses, which takes minutes; subsequent calls use the cache.",
        name,
        spec.moabb_class,
        spec.paradigm,
    )
    started = time.perf_counter()
    epochs, labels, table = fetch_moabb_epochs(
        spec,
        subjects=subjects,
        tmin=window_start,
        tmax=window_end,
        low_freq=low_freq,
        high_freq=high_freq,
        resample=resample,
        channels=channels,
    )
    elapsed = time.perf_counter() - started

    dataset = _build_dataset(
        name=name,
        spec=spec,
        epochs=epochs,
        labels=labels,
        table=table,
        parameters=parameters,
        load_seconds=elapsed,
    )
    _validate_against_spec(
        dataset,
        spec,
        name,
        subjects=subjects,
        channels=channels,
        resample=resample,
        window=(window_start, window_end),
    )

    if cache_file is not None:
        _write_cache(cache_file, dataset)
        logger.info("Cached %r to %s", name, cache_file.name)

    return dataset


def _build_dataset(
    *,
    name: str,
    spec: MOABBSpec,
    epochs: np.ndarray,
    labels: np.ndarray,
    table: Any,
    parameters: dict[str, Any],
    load_seconds: float,
) -> EEGDataset:
    """Assemble an :class:`EEGDataset` from MOABB's return values.

    Args:
        name: Dataset name.
        spec: Its specification.
        epochs: Epoch array from MOABB.
        labels: Label array from MOABB.
        table: MOABB's metadata table, carrying subject and session columns.
        parameters: Preprocessing parameters, recorded in the metadata.
        load_seconds: Fetch duration, recorded for the provenance record.

    Returns:
        The dataset.

    Raises:
        ValueError: If MOABB's metadata table lacks a subject column. Falling
            back to a single pseudo-subject would silently turn every
            subject-independent protocol into a within-subject one.
    """
    if not hasattr(table, "columns") or "subject" not in table.columns:
        raise ValueError(
            "MOABB returned metadata without a 'subject' column, so trials "
            "cannot be grouped by participant. Continuing would turn every "
            "subject-independent protocol into a within-subject one while "
            "leaving its checks passing."
        )

    subjects = np.asarray(table["subject"])
    sessions = (
        np.asarray(table["session"])
        if "session" in table.columns
        else np.array([], dtype=object)
    )

    #  The rate is taken from the request, not derived from the epoch length.
    #
    #  Deriving it looks safer and is wrong: MNE's epoch window is inclusive of
    #  both endpoints, so tmin=0.5 to tmax=2.5 at 250 Hz yields 501 samples
    #  rather than 500, and n_times / duration reports 250.5 Hz. The authority
    #  on the rate is the resample argument, or the dataset's native rate when
    #  no resampling was requested. The sample count is cross-checked against
    #  that rate in _validate_against_spec instead.
    sampling_rate = float(
        parameters["resample"]
        if parameters["resample"] is not None
        else spec.sampling_rate
    )

    return EEGDataset(
        epochs=np.asarray(epochs, dtype=np.float64),
        labels=np.asarray(labels),
        subjects=subjects,
        sessions=sessions,
        sampling_rate=sampling_rate,
        name=name,
        channel_names=tuple(f"CH{index + 1:02d}" for index in range(epochs.shape[1])),
        metadata={
            "source": "moabb",
            "moabb_class": spec.moabb_class,
            "paradigm": spec.paradigm,
            "citation": spec.citation,
            "load_seconds": round(load_seconds, 2),
            **parameters,
        },
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def _make_loader(name: str):
    """Return a registry loader bound to one dataset name."""

    def loader(**kwargs: Any) -> EEGDataset:
        return load_moabb(name, **kwargs)

    loader.__name__ = f"load_{name}"
    loader.__doc__ = f"Load the {name} dataset through MOABB."
    return loader


for _name in MOABB_SPECS:
    register_dataset(_name)(_make_loader(_name))
