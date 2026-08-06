"""Tests for :mod:`geoq.datasets.moabb_adapter`.

Testing without a network
-------------------------
:func:`geoq.datasets.moabb_adapter.fetch_moabb_epochs` is the single function
that touches the network, and every test here replaces it. That is deliberate:
the logic determining whether a loaded dataset is trustworthy -- specification
validation, cache keying, subject grouping -- should not be verifiable only
when a 1.6 GB download succeeds.

What that leaves unverified is stated plainly rather than glossed over: these
tests establish that the adapter behaves correctly *given* what MOABB returns.
They cannot establish that MOABB returns what the specifications claim. That
check belongs in a one-off validation run against the real files, and the
specifications in :data:`MOABB_SPECS` are what turn its outcome into a
permanent guard.

What is being defended
----------------------
* **Cache keys cover every preprocessing parameter.** ``TestCacheKey`` checks
  each one individually. A key on the dataset name alone would serve 8-30 Hz
  epochs to a request for 4-8 Hz, and the results would silently answer a
  different question.
* **Specification mismatches are errors.** ``TestSpecValidation`` covers a
  changed channel count, a wrong sampling rate, a missing subject, and an
  unexpected class. Each corresponds to a real way an upstream change makes
  today's data quietly different from last month's.
* **Subject grouping is mandatory.** Without it every subject-independent
  protocol silently becomes a within-subject one, with all of its checks
  still passing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pandas", reason="requires the 'ml' extra")

import pandas as pd

from geoq.datasets import moabb_adapter
from geoq.datasets.base import DATASETS, EEGDataset, load_dataset
from geoq.datasets.moabb_adapter import (
    MOABB_SPECS,
    MOABBSpec,
    _cache_key,
    available_datasets,
    load_moabb,
)

BASE_PARAMETERS: dict[str, Any] = {
    "subjects": None,
    "tmin": 0.5,
    "tmax": 2.5,
    "low_freq": 8.0,
    "high_freq": 30.0,
    "resample": None,
    "channels": None,
    "moabb_class": "BNCI2014_001",
    "paradigm": "MotorImagery",
}


def make_fetcher(
    *,
    n_trials_per_subject: int = 16,
    n_channels: int | None = None,
    n_subjects: int | None = None,
    classes: tuple[str, ...] | None = None,
    include_session: bool = True,
    include_subject: bool = True,
    seed: int = 0,
):
    """Build a stand-in for the network fetch.

    Args:
        n_trials_per_subject: Trials generated per subject.
        n_channels: Override the specification's channel count, to simulate an
            upstream change.
        n_subjects: Override the subject count, to simulate a failed download.
        classes: Override the class labels.
        include_session: Whether the metadata table has a session column.
        include_subject: Whether it has a subject column.
        seed: Random seed.

    Returns:
        A callable with the signature of ``fetch_moabb_epochs``.
    """

    def fetch(spec: MOABBSpec, **kwargs: Any):
        rng = np.random.default_rng(seed)
        subjects = kwargs["subjects"] or list(
            range(1, (n_subjects or spec.n_subjects) + 1)
        )
        channels = n_channels or spec.n_channels
        rate = kwargs["resample"] or spec.sampling_rate
        #  MNE epochs include both endpoints, so a two-second window at 250 Hz
        #  is 501 samples rather than 500. The mock reproduces that exactly:
        #  an earlier version used the exclusive count and let a wrong
        #  sampling-rate derivation pass every test here while failing on the
        #  first real download.
        n_times = round((kwargs["tmax"] - kwargs["tmin"]) * rate) + 1
        n_trials = len(subjects) * n_trials_per_subject

        labels = classes or spec.classes[:2]
        table: dict[str, Any] = {}
        if include_subject:
            table["subject"] = np.repeat(subjects, n_trials_per_subject)
        if include_session:
            table["session"] = np.tile(
                np.repeat([0, 1], n_trials_per_subject // 2), len(subjects)
            )
        if not table:
            table["run"] = np.zeros(n_trials)

        return (
            rng.standard_normal((n_trials, channels, n_times)),
            np.array([labels[index % len(labels)] for index in range(n_trials)]),
            pd.DataFrame(table),
        )

    return fetch


@pytest.fixture
def mocked(monkeypatch: pytest.MonkeyPatch):
    """Replace the network boundary with a configurable stand-in."""

    def install(**kwargs: Any) -> None:
        monkeypatch.setattr(moabb_adapter, "fetch_moabb_epochs", make_fetcher(**kwargs))

    install()
    return install


# --------------------------------------------------------------------------- #
# 1. Specifications and registration
# --------------------------------------------------------------------------- #


class TestSpecifications:
    """Expectations are declared up front."""

    def test_known_datasets_are_registered(self) -> None:
        for name in MOABB_SPECS:
            assert name in DATASETS

    def test_two_class_variant_is_registered(self) -> None:
        """Same recordings, different paradigm, different published numbers.

        Most Riemannian BCI results are computed on the left/right subset, so
        comparing against the literature requires knowing which variant
        produced a result. Separate registry names make that visible in the
        configuration file rather than implicit in a paradigm argument.
        """
        four_class = MOABB_SPECS["bci_iv_2a"]
        two_class = MOABB_SPECS["bci_iv_2a_lr"]
        assert four_class.moabb_class == two_class.moabb_class
        assert four_class.paradigm != two_class.paradigm
        assert len(two_class.classes) == 2

    def test_bci_iv_2a_matches_the_published_description(self) -> None:
        """The dataset this thesis is built on.

        Nine subjects, 22 EEG channels, 250 Hz, four motor-imagery classes.
        Encoding it here turns a future upstream change from an invisible
        difference into a load-time error.
        """
        spec = MOABB_SPECS["bci_iv_2a"]
        assert spec.moabb_class == "BNCI2014_001"
        assert spec.n_subjects == 9
        assert spec.n_channels == 22
        assert spec.sampling_rate == 250.0
        assert set(spec.classes) == {"left_hand", "right_hand", "feet", "tongue"}

    def test_every_spec_is_self_consistent(self) -> None:
        for name, spec in MOABB_SPECS.items():
            assert spec.n_subjects > 0, name
            assert spec.n_channels > 0, name
            assert spec.sampling_rate > 0, name
            assert len(spec.classes) >= 2, name
            assert spec.default_window[1] > spec.default_window[0], name
            assert spec.citation, name

    def test_available_datasets_returns_a_copy(self) -> None:
        """Mutating the returned mapping must not change the registry."""
        available_datasets().pop("bci_iv_2a", None)
        assert "bci_iv_2a" in MOABB_SPECS

    def test_unknown_dataset_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown MOABB dataset"):
            load_moabb("bci_iv_2c")

    def test_reachable_through_the_dataset_registry(self, mocked) -> None:
        """A configuration file names a dataset; this is the path it takes."""
        data = load_dataset("bci_iv_2a", subjects=[1, 2])
        assert isinstance(data, EEGDataset)
        assert data.n_subjects == 2


# --------------------------------------------------------------------------- #
# 2. Loading
# --------------------------------------------------------------------------- #


class TestLoading:
    """Assembly of the canonical container."""

    def test_shape_and_metadata(self, mocked) -> None:
        data = load_moabb("bci_iv_2a")
        assert data.n_subjects == 9
        assert data.n_channels == 22
        # 501, not 500: MNE's window includes both endpoints.
        assert data.n_times == 501
        assert data.sampling_rate == pytest.approx(250.0)
        assert data.metadata["source"] == "moabb"
        assert data.metadata["moabb_class"] == "BNCI2014_001"
        assert "citation" in data.metadata

    def test_labels_remain_strings(self, mocked) -> None:
        """An integer label is one renumbering away from swapping two classes.

        Keeping the names also makes a confusion matrix readable, which is
        worth more than the bytes saved.
        """
        data = load_moabb("bci_iv_2a")
        assert data.classes.dtype.kind in "US"
        assert set(data.classes.tolist()) == {"left_hand", "right_hand"}

    def test_subject_grouping_is_preserved(self, mocked) -> None:
        data = load_moabb("bci_iv_2a", subjects=[3, 5, 7])
        assert sorted(set(data.subjects.tolist())) == [3, 5, 7]
        for subject in (3, 5, 7):
            assert int(np.sum(data.subjects == subject)) == 16

    def test_sessions_are_carried_through(self, mocked) -> None:
        assert set(load_moabb("bci_iv_2a").sessions.tolist()) == {0, 1}

    def test_missing_session_column_is_tolerated(self, mocked) -> None:
        """Not every dataset has sessions; the container allows an empty one."""
        mocked(include_session=False)
        assert load_moabb("bci_iv_2a").sessions.size == 0

    def test_missing_subject_column_is_fatal(self, mocked) -> None:
        """The failure that silently invalidates every protocol.

        Without subject identifiers, leave-one-subject-out degenerates into an
        arbitrary partition while every one of its checks continues to pass.
        Falling back to a single pseudo-subject would be the worst possible
        default.
        """
        mocked(include_subject=False, include_session=False)
        with pytest.raises(ValueError, match="'subject' column"):
            load_moabb("bci_iv_2a")

    def test_sampling_rate_comes_from_the_request(self, mocked) -> None:
        """Taken from the resample argument, not measured from the epochs.

        Measuring it looks safer and is wrong. MNE's window includes both
        endpoints, so ``n_times / duration`` reports 250.5 Hz for a two-second
        window at 250 Hz -- which is what the specification check caught on
        the first real download.
        """
        data = load_moabb("bci_iv_2a", resample=128.0, high_freq=30.0)
        assert data.sampling_rate == pytest.approx(128.0)
        assert data.n_times == 257

    def test_native_rate_is_reported_exactly(self, mocked) -> None:
        """No resampling means the dataset's own rate, to the digit."""
        assert load_moabb("bci_iv_2a").sampling_rate == 250.0

    def test_default_window_comes_from_the_spec(self, mocked) -> None:
        data = load_moabb("bci_iv_2a")
        assert data.metadata["tmin"] == 0.5
        assert data.metadata["tmax"] == 2.5

    def test_preprocessing_parameters_are_recorded(self, mocked) -> None:
        """The provenance record must describe how the epochs were made."""
        data = load_moabb("bci_iv_2a", low_freq=4.0, high_freq=40.0)
        assert data.metadata["low_freq"] == 4.0
        assert data.metadata["high_freq"] == 40.0
        assert "load_seconds" in data.metadata


class TestParameterValidation:
    """Impossible requests fail before any download starts."""

    def test_inverted_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="must exceed tmin"):
            load_moabb("bci_iv_2a", tmin=2.0, tmax=1.0)

    def test_inverted_band_rejected(self) -> None:
        with pytest.raises(ValueError, match="must exceed low_freq"):
            load_moabb("bci_iv_2a", low_freq=30.0, high_freq=8.0)

    def test_nyquist_violation_rejected(self) -> None:
        """Aliasing produces something that looks like EEG and is not.

        Caught before the download rather than diagnosed afterwards from
        results that decode like noise.
        """
        with pytest.raises(ValueError, match="aliases"):
            load_moabb("bci_iv_2a", high_freq=30.0, resample=50.0)

    def test_native_rate_against_a_wide_band_rejected(self) -> None:
        """250 Hz cannot represent a band up to 130 Hz."""
        with pytest.raises(ValueError, match="aliases"):
            load_moabb("bci_iv_2a", low_freq=8.0, high_freq=130.0)


# --------------------------------------------------------------------------- #
# 3. Specification validation
# --------------------------------------------------------------------------- #


class TestSpecValidation:
    """A load that disagrees with the published description is an error."""

    def test_changed_channel_count_rejected(self, mocked) -> None:
        """A MOABB release altering default channel selection.

        The data would load cleanly and not be what an earlier result was
        computed on, so comparing across the change would be comparing across
        something invisible.
        """
        mocked(n_channels=25)
        with pytest.raises(ValueError, match="channels but 22 were expected"):
            load_moabb("bci_iv_2a")

    def test_explicit_channel_selection_bypasses_the_check(self, mocked) -> None:
        """Asking for fewer channels is a deliberate choice, not a mismatch."""
        mocked(n_channels=3)
        assert load_moabb("bci_iv_2a", channels=["C3", "Cz", "C4"]).n_channels == 3

    def test_missing_subject_rejected(self, mocked) -> None:
        """A subject whose download failed must not vanish silently.

        Otherwise leave-one-subject-out reports a mean over fewer folds than
        the methods section claims.
        """
        mocked(n_subjects=7)
        with pytest.raises(ValueError, match="7 subjects but 9 were expected"):
            load_moabb("bci_iv_2a")

    def test_subject_subset_is_checked_against_the_request(self, mocked) -> None:
        assert load_moabb("bci_iv_2a", subjects=[1, 2, 3]).n_subjects == 3

    def test_unexpected_class_rejected(self, mocked) -> None:
        mocked(classes=("left_hand", "rest"))
        with pytest.raises(ValueError, match="unexpected class labels"):
            load_moabb("bci_iv_2a")

    @pytest.mark.parametrize(
        ("offset", "accepted"),
        [(0, True), (1, True), (-1, True), (2, False), (-250, False)],
    )
    def test_epoch_length_is_cross_checked(
        self, offset: int, accepted: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sample count is what reveals a rate mismatch.

        The rate itself now comes from the request rather than being measured,
        so the epoch length is the quantity that can disagree. A tolerance of
        one sample absorbs MNE's rounding at awkward rate-and-window
        combinations; it cannot absorb a genuinely different sampling rate,
        which is wrong by hundreds of samples. The ``-250`` case is 125 Hz
        masquerading as 250.
        """

        def shifted(spec: MOABBSpec, **kwargs: Any):
            rate = kwargs["resample"] or spec.sampling_rate
            n_times = round((kwargs["tmax"] - kwargs["tmin"]) * rate) + 1 + offset
            n_trials = spec.n_subjects * 8
            return (
                np.random.default_rng(0).standard_normal(
                    (n_trials, spec.n_channels, n_times)
                ),
                np.array([spec.classes[index % 2] for index in range(n_trials)]),
                pd.DataFrame({"subject": np.repeat(range(1, spec.n_subjects + 1), 8)}),
            )

        monkeypatch.setattr(moabb_adapter, "fetch_moabb_epochs", shifted)
        if accepted:
            assert load_moabb("bci_iv_2a").n_subjects == 9
        else:
            with pytest.raises(ValueError, match="samples per epoch"):
                load_moabb("bci_iv_2a")


# --------------------------------------------------------------------------- #
# 4. Caching
# --------------------------------------------------------------------------- #


class TestCacheKey:
    """Every preprocessing parameter must reach the key."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("low_freq", 4.0),
            ("high_freq", 13.0),
            ("tmin", 1.0),
            ("tmax", 3.0),
            ("resample", 128.0),
            ("subjects", [1, 2]),
            ("channels", ["C3", "Cz"]),
            ("paradigm", "LeftRightImagery"),
        ],
    )
    def test_each_parameter_changes_the_key(self, field: str, value: Any) -> None:
        """A key on the dataset name alone is the failure this prevents.

        It would serve 8-30 Hz epochs to a request for 4-8 Hz, and every
        result computed from them would silently answer a different question
        than its configuration describes.
        """
        base = _cache_key("bci_iv_2a", BASE_PARAMETERS)
        assert _cache_key("bci_iv_2a", {**BASE_PARAMETERS, field: value}) != base

    def test_dataset_name_changes_the_key(self) -> None:
        assert _cache_key("bci_iv_2b", BASE_PARAMETERS) != _cache_key(
            "bci_iv_2a", BASE_PARAMETERS
        )

    def test_key_order_is_irrelevant(self) -> None:
        reordered = dict(reversed(list(BASE_PARAMETERS.items())))
        assert _cache_key("bci_iv_2a", reordered) == _cache_key(
            "bci_iv_2a", BASE_PARAMETERS
        )

    def test_key_is_a_short_hex_digest(self) -> None:
        key = _cache_key("bci_iv_2a", BASE_PARAMETERS)
        assert len(key) == 16
        assert all(character in "0123456789abcdef" for character in key)


class TestCaching:
    """Reading, writing, and refusing to serve the wrong thing."""

    def test_round_trip(self, mocked, tmp_path: Path) -> None:
        first = load_moabb("bci_iv_2a", cache_dir=tmp_path)
        second = load_moabb("bci_iv_2a", cache_dir=tmp_path)
        assert np.array_equal(first.epochs, second.epochs)
        assert np.array_equal(first.labels, second.labels)
        assert np.array_equal(first.subjects, second.subjects)
        assert first.sampling_rate == second.sampling_rate
        assert first.metadata["moabb_class"] == second.metadata["moabb_class"]

    def test_second_load_does_not_refetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The point of the cache: preprocessing runs once."""
        calls: list[int] = []
        fetcher = make_fetcher()

        def counting(spec: MOABBSpec, **kwargs: Any):
            calls.append(1)
            return fetcher(spec, **kwargs)

        monkeypatch.setattr(moabb_adapter, "fetch_moabb_epochs", counting)
        load_moabb("bci_iv_2a", cache_dir=tmp_path)
        load_moabb("bci_iv_2a", cache_dir=tmp_path)
        assert len(calls) == 1

    def test_different_preprocessing_gets_its_own_cache(
        self, mocked, tmp_path: Path
    ) -> None:
        load_moabb("bci_iv_2a", cache_dir=tmp_path)
        load_moabb("bci_iv_2a", low_freq=4.0, high_freq=8.0, cache_dir=tmp_path)
        load_moabb("bci_iv_2a", cache_dir=tmp_path)
        assert len(list(tmp_path.glob("*.npz"))) == 2

    def test_use_cache_false_refetches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[int] = []
        fetcher = make_fetcher()

        def counting(spec: MOABBSpec, **kwargs: Any):
            calls.append(1)
            return fetcher(spec, **kwargs)

        monkeypatch.setattr(moabb_adapter, "fetch_moabb_epochs", counting)
        load_moabb("bci_iv_2a", cache_dir=tmp_path, use_cache=False)
        load_moabb("bci_iv_2a", cache_dir=tmp_path, use_cache=False)
        assert len(calls) == 2

    def test_no_cache_dir_means_no_files(self, mocked, tmp_path: Path) -> None:
        load_moabb("bci_iv_2a")
        assert list(tmp_path.glob("*.npz")) == []

    def test_corrupt_cache_is_recomputed(
        self, mocked, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A damaged cache must not be fatal.

        The data can always be recomputed, so failing would make one
        interrupted write permanently break an experiment.
        """
        load_moabb("bci_iv_2a", cache_dir=tmp_path)
        cache_file = next(tmp_path.glob("*.npz"))
        cache_file.write_bytes(b"not an npz archive")

        logger_name = "geoq.datasets.moabb_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            data = load_moabb("bci_iv_2a", cache_dir=tmp_path)
        assert data.n_subjects == 9
        assert "recomputing" in caplog.text

    def test_no_temporary_files_remain(self, mocked, tmp_path: Path) -> None:
        """A leftover temporary would accumulate across every disconnect."""
        load_moabb("bci_iv_2a", cache_dir=tmp_path)
        assert list(tmp_path.glob(".tmp-*")) == []

    def test_cache_directory_is_created(self, mocked, tmp_path: Path) -> None:
        target = tmp_path / "deeply" / "nested"
        load_moabb("bci_iv_2a", cache_dir=target)
        assert len(list(target.glob("*.npz"))) == 1


# --------------------------------------------------------------------------- #
# 5. Downstream compatibility
# --------------------------------------------------------------------------- #


class TestPipelineCompatibility:
    """A MOABB dataset feeds the framework unchanged."""

    def test_full_pipeline_runs_on_string_labels(self, mocked) -> None:
        """String labels must work end to end, not only in the container."""
        pytest.importorskip("sklearn")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline

        from geoq.evaluation.protocol import evaluate
        from geoq.evaluation.splitters import LeaveOneSubjectOut
        from geoq.features.covariance import Covariances
        from geoq.features.tangent_space import TangentSpace

        data = load_moabb("bci_iv_2a", subjects=[1, 2, 3, 4])
        pipeline = make_pipeline(
            Covariances(estimator="oas", audit_conditioning=False),
            TangentSpace(),
            LinearDiscriminantAnalysis(),
        )
        result = evaluate(
            pipeline,
            data.epochs,
            data.labels,
            groups=data.subjects,
            splitter=LeaveOneSubjectOut(min_subjects=2),
        )
        assert len(result.folds) == 4
        assert result.protocol.subject_independent

    def test_summary_is_provenance_ready(self, mocked) -> None:
        summary = load_moabb("bci_iv_2a").summary()
        assert summary["dataset"] == "bci_iv_2a"
        assert summary["n_channels"] == 22
        assert summary["source"] == "moabb"
        assert summary["citation"]
