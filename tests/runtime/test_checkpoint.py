"""Tests for :mod:`geoq.runtime.checkpoint`.

What is being defended
----------------------
The whole module exists for one scenario: a Colab session drops in hour five
of an overnight sweep, and the next session must resume rather than restart.
Three properties make that work, and each is tested against a simulated
failure rather than assumed.

* **Atomicity.** ``TestAtomicWrite`` interrupts a write partway and asserts
  that no shard appears at the destination and no temporary file is left
  behind. A half-written shard that parses would be the worst outcome: the
  unit would be skipped on resume and its result silently wrong.
* **Corruption means recompute.** ``TestCorruption`` damages shards in the
  three ways a interrupted network filesystem actually produces -- truncation,
  a wrong checksum, an empty file -- and asserts each is reported as pending.
* **No central index.** ``TestNoCentralIndex`` deletes and adds shard files
  behind the store's back and confirms the reported progress follows the
  directory. A manifest would be a single point of failure whose disagreement
  with reality causes finished work to be redone or unfinished work skipped.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from geoq.runtime.checkpoint import (
    SHARD_SUFFIX,
    CorruptShardError,
    Shard,
    ShardStore,
    to_jsonable,
)


def _raise_os_error(*_args: Any, **_kwargs: Any) -> None:
    """Stand in for a write that fails partway through."""
    raise OSError("boom")


@pytest.fixture
def store(tmp_path: Path) -> ShardStore:
    """An empty store in a temporary directory."""
    return ShardStore(tmp_path / "shards")


@pytest.fixture
def populated(store: ShardStore) -> ShardStore:
    """A store holding five shards."""
    for index in range(5):
        store.write(
            f"unit{index:04d}",
            {"kappa": 0.4 + 0.05 * index, "n_test": 288},
            fold=index,
            subject=index,
        )
    return store


# --------------------------------------------------------------------------- #
# 1. Round trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    """What goes in comes back out."""

    def test_write_then_read(self, store: ShardStore) -> None:
        store.write("unit0001", {"kappa": 0.42}, fold=3)
        shard = store.read("unit0001")
        assert isinstance(shard, Shard)
        assert shard.unit_id == "unit0001"
        assert shard.payload == {"kappa": 0.42}
        assert shard.metadata == {"fold": 3}
        assert shard.verify()

    def test_missing_unit_reads_as_none(self, store: ShardStore) -> None:
        assert store.read("absent0001") is None

    def test_numpy_payload_survives(self, store: ShardStore) -> None:
        store.write(
            "unit0001",
            {
                "scores": np.array([0.4, 0.5, 0.6]),
                "kappa": np.float64(0.42),
                "folds": np.int64(9),
                "converged": np.bool_(True),
            },
        )
        payload = store.read("unit0001").payload
        assert payload["scores"] == [0.4, 0.5, 0.6]
        assert payload["kappa"] == pytest.approx(0.42)
        assert payload["folds"] == 9
        assert payload["converged"] is True

    def test_timestamp_recorded(self, store: ShardStore) -> None:
        store.write("unit0001", {"kappa": 0.4})
        assert store.read("unit0001").written_at.endswith("+00:00")

    def test_rewrite_replaces(self, store: ShardStore) -> None:
        """Recomputing a unit must overwrite, not append or duplicate."""
        store.write("unit0001", {"kappa": 0.4})
        store.write("unit0001", {"kappa": 0.9})
        assert store.read("unit0001").payload == {"kappa": 0.9}
        assert len(store) == 1

    def test_shard_file_is_human_readable(self, store: ShardStore) -> None:
        """A results file a reader must be able to open in five years.

        JSON is slower and larger than a binary format and chosen anyway: a
        provenance artefact that needs this codebase to read it is a
        liability.
        """
        store.write("unit0001", {"kappa": 0.42}, fold=3)
        text = store.path_for("unit0001").read_text(encoding="utf-8")
        assert "kappa" in text
        assert json.loads(text)["payload"]["kappa"] == 0.42


class TestSerialisation:
    """Conversion to JSON-safe values."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.int32(7), 7),
            (np.float32(0.5), 0.5),
            (np.bool_(False), False),
            (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
            ({"a": np.array([1.0])}, {"a": [1.0]}),
            ((1, np.int8(2)), [1, 2]),
            (Path("a/b"), "a/b"),
            (None, None),
        ],
    )
    def test_conversions(self, value: Any, expected: Any) -> None:
        assert to_jsonable(value) == expected

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_non_finite_becomes_a_string(self, value: float) -> None:
        """JSON has no NaN or infinity.

        Emitting the bare literals produces a file that Python's parser
        accepts and no other does, so the loss is made explicit instead.
        """
        converted = to_jsonable(np.float64(value))
        assert isinstance(converted, str)
        assert json.dumps({"x": converted})

    def test_windows_path_uses_forward_slashes(self) -> None:
        """A results file must read the same on every platform."""
        assert "\\" not in to_jsonable(Path("a") / "b" / "c")

    def test_objects_with_to_dict_are_supported(self) -> None:
        class Result:
            def to_dict(self) -> dict[str, float]:
                return {"kappa": 0.4}

        assert to_jsonable(Result()) == {"kappa": 0.4}

    def test_unsupported_type_raises(self, store: ShardStore) -> None:
        """Stringifying silently would put pseudo-data into a results file."""
        with pytest.raises(TypeError, match="Cannot serialise"):
            store.write("unit0001", {"model": object()})


# --------------------------------------------------------------------------- #
# 2. Atomicity
# --------------------------------------------------------------------------- #


class TestAtomicWrite:
    """A shard is either absent or complete."""

    def test_interrupted_write_leaves_nothing(
        self, store: ShardStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scenario this module exists for.

        A half-written shard that happened to parse would be the worst
        possible outcome: the unit would be skipped on resume and its result
        silently wrong. Writing to a temporary file and moving it into place
        makes that state unreachable.
        """
        import geoq.runtime.checkpoint as module

        def explode(*args: Any, **kwargs: Any) -> None:
            raise OSError("simulated disconnect mid-write")

        monkeypatch.setattr(module.json, "dump", explode)
        with pytest.raises(OSError, match="simulated disconnect"):
            store.write("unit0001", {"kappa": 0.4})

        assert not store.path_for("unit0001").exists()
        assert store.read("unit0001") is None
        assert list(store.root.glob(".tmp-*")) == []

    def test_content_is_never_written_to_the_destination_directly(
        self, store: ShardStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mechanism itself, not just its visible effect.

        Added after a mutation test: replacing the temp-file-plus-rename with
        a direct write to the destination was caught by none of the tests
        above, because they interrupt the write before either implementation
        reaches the destination. This one observes the filesystem at the
        moment serialisation happens and asserts that the destination is still
        absent while a temporary file is present.

        Without that ordering the atomicity claim is unverified, and a
        disconnect between opening the file and finishing the write would
        leave a shard that exists and is truncated -- which resume would skip.
        """
        import geoq.runtime.checkpoint as module

        observations: dict[str, bool] = {}
        real_dump = module.json.dump

        def observing_dump(obj: Any, stream: Any, **kwargs: Any) -> None:
            observations["destination_absent"] = not store.path_for("unit0001").exists()
            observations["temporary_present"] = bool(list(store.root.glob(".tmp-*")))
            return real_dump(obj, stream, **kwargs)

        monkeypatch.setattr(module.json, "dump", observing_dump)
        store.write("unit0001", {"kappa": 0.42})

        assert observations["destination_absent"]
        assert observations["temporary_present"]
        assert store.read("unit0001").payload == {"kappa": 0.42}

    def test_failure_between_write_and_rename_leaves_nothing(
        self, store: ShardStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The narrow window a direct write would widen to the whole operation.

        Serialisation succeeds, the move fails. The destination must remain
        absent and the completed temporary file must be cleaned up.
        """

        def failing_replace(self: Path, target: Any) -> None:
            raise OSError("simulated failure during rename")

        monkeypatch.setattr(Path, "replace", failing_replace)
        with pytest.raises(OSError, match="during rename"):
            store.write("unit0001", {"kappa": 0.42})

        assert not store.path_for("unit0001").exists()
        assert list(store.root.glob(".tmp-*")) == []

    def test_interrupted_write_does_not_damage_an_existing_shard(
        self, store: ShardStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recomputation that fails must leave the previous result intact."""
        store.write("unit0001", {"kappa": 0.42})
        import geoq.runtime.checkpoint as module

        monkeypatch.setattr(
            module.json,
            "dump",
            _raise_os_error,
        )
        with pytest.raises(OSError):
            store.write("unit0001", {"kappa": 0.99})
        assert store.read("unit0001").payload == {"kappa": 0.42}

    def test_temporary_files_are_ignored_by_discovery(self, store: ShardStore) -> None:
        """A write in progress must not be counted as completed work."""
        store.write("unit0001", {"kappa": 0.4})
        (store.root / f".tmp-999-1-abc{SHARD_SUFFIX}").write_text("{}")
        assert store.completed_units() == {"unit0001"}
        assert len(store) == 1

    def test_concurrent_writers_do_not_collide(self, store: ShardStore) -> None:
        """Parallel workers each write their own temporary file.

        A shared temporary name would let one worker's partial write land
        under another's identifier -- a corruption that checksums cannot
        detect, because the file would be internally consistent and simply
        belong to the wrong unit.
        """
        identifiers = [f"unit{index:05d}" for index in range(200)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda unit: store.write(unit, {"value": int(unit[4:])}),
                    identifiers,
                )
            )
        assert store.completed_units() == set(identifiers)
        assert list(store.root.glob(".tmp-*")) == []
        for unit in identifiers[:20]:
            assert store.read(unit).payload["value"] == int(unit[4:])


# --------------------------------------------------------------------------- #
# 3. Corruption
# --------------------------------------------------------------------------- #


class TestCorruption:
    """Damage means recompute, not crash."""

    @pytest.mark.parametrize(
        ("content", "label"),
        [
            ('{"unit_id":"unit0001","payl', "truncated mid-write"),
            ("", "empty file"),
            (
                '{"unit_id":"unit0001","payload":{"k":1},"checksum":"wrong"}',
                "bad checksum",
            ),
            ('{"payload":{"k":1}}', "missing unit_id"),
            ("not json at all", "not json"),
        ],
    )
    def test_damaged_shard_reads_as_missing(
        self,
        content: str,
        label: str,
        store: ShardStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every damage mode an interrupted network filesystem produces.

        Failing the run instead would turn one damaged file into a lost
        overnight sweep, which is exactly what this module exists to prevent.
        """
        store.path_for("unit0001").write_text(content, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="geoq.runtime.checkpoint"):
            assert store.read("unit0001") is None
        assert "recomputed" in caplog.text

    def test_damaged_shard_appears_in_pending(self, store: ShardStore) -> None:
        """The property that makes recomputation actually happen."""
        store.write("unit0001", {"kappa": 0.4})
        store.write("unit0002", {"kappa": 0.5})
        store.path_for("unit0002").write_text("{corrupt", encoding="utf-8")
        assert store.pending(["unit0001", "unit0002"]) == ["unit0002"]

    def test_payload_tampering_is_detected(self, store: ShardStore) -> None:
        """The checksum covers the payload, so an edit invalidates the shard."""
        store.write("unit0001", {"kappa": 0.4})
        path = store.path_for("unit0001")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["payload"]["kappa"] = 0.99
        path.write_text(json.dumps(record), encoding="utf-8")
        assert store.read("unit0001") is None

    def test_strict_mode_raises(self, tmp_path: Path) -> None:
        """For auditing a completed run, where silence would hide damage."""
        strict = ShardStore(tmp_path / "s", strict=True)
        strict.path_for("unit0001").write_text("{bad", encoding="utf-8")
        with pytest.raises(CorruptShardError, match="could not be parsed"):
            strict.read("unit0001")

    def test_strict_mode_message_names_the_file(self, tmp_path: Path) -> None:
        strict = ShardStore(tmp_path / "s", strict=True)
        strict.write("unit0001", {"kappa": 0.4})
        record = json.loads(strict.path_for("unit0001").read_text(encoding="utf-8"))
        record["checksum"] = "0" * 64
        strict.path_for("unit0001").write_text(json.dumps(record))
        with pytest.raises(CorruptShardError, match=r"unit0001\.json"):
            strict.read("unit0001")

    def test_clear_corrupt_removes_only_damaged_shards(
        self, populated: ShardStore
    ) -> None:
        populated.path_for("unit0002").write_text("{bad", encoding="utf-8")
        removed = populated.clear_corrupt()
        assert removed == [f"unit0002{SHARD_SUFFIX}"]
        assert len(populated) == 4

    def test_clear_corrupt_is_not_automatic(self, populated: ShardStore) -> None:
        """Deleting damaged shards silently would erase the evidence.

        A cluster of corrupt shards is a signal about the storage, worth
        investigating before it is destroyed.
        """
        populated.path_for("unit0002").write_text("{bad", encoding="utf-8")
        populated.pending([f"unit{i:04d}" for i in range(5)])
        assert populated.path_for("unit0002").exists()

    def test_clear_temporary(self, store: ShardStore) -> None:
        for index in range(3):
            (store.root / f".tmp-1-{index}-x{SHARD_SUFFIX}").write_text("{}")
        assert len(store.clear_temporary()) == 3
        assert list(store.root.glob(".tmp-*")) == []


# --------------------------------------------------------------------------- #
# 4. Resumption
# --------------------------------------------------------------------------- #


class TestNoCentralIndex:
    """The filesystem is the only record of progress."""

    def test_progress_follows_the_directory(self, populated: ShardStore) -> None:
        """Files removed behind the store's back must show as pending.

        A manifest would disagree with reality here, and a manifest that
        disagrees is worse than none: it either redoes finished work or, far
        worse, skips work that was never done.
        """
        units = [f"unit{index:04d}" for index in range(5)]
        assert populated.pending(units) == []
        populated.path_for("unit0003").unlink()
        assert populated.pending(units) == ["unit0003"]

    def test_a_second_store_sees_the_same_progress(self, populated: ShardStore) -> None:
        """A new session must find the previous one's work.

        No in-memory state carries over, which is the point: the next Colab
        session is a fresh process with a fresh object.
        """
        reopened = ShardStore(populated.root)
        assert reopened.completed_units() == populated.completed_units()

    def test_pending_preserves_the_requested_order(self, store: ShardStore) -> None:
        """A resumed run proceeds in the original sequence, so logs compare."""
        units = [f"unit{index:04d}" for index in range(10)]
        for index in (1, 4, 7):
            store.write(units[index], {"kappa": 0.4})
        assert store.pending(units) == [units[index] for index in (0, 2, 3, 5, 6, 8, 9)]

    def test_pending_of_an_empty_store_is_everything(self, store: ShardStore) -> None:
        units = [f"unit{index:04d}" for index in range(4)]
        assert store.pending(units) == units

    def test_resume_after_simulated_disconnect(self, store: ShardStore) -> None:
        """The end-to-end scenario, as a single test.

        Six units, a disconnect after three, then a fresh store object that
        completes only the remaining three.
        """
        units = [f"unit{index:04d}" for index in range(6)]
        computed: list[str] = []

        def run(target: ShardStore) -> None:
            for unit in target.pending(units):
                computed.append(unit)
                target.write(unit, {"kappa": 0.4})
                if len(computed) == 3:
                    raise KeyboardInterrupt("simulated disconnect")

        with pytest.raises(KeyboardInterrupt):
            run(store)
        assert len(computed) == 3

        run(ShardStore(store.root))
        assert computed == units
        assert len(set(computed)) == 6  # nothing recomputed

    def test_exists_without_verification_is_cheaper_and_looser(
        self, store: ShardStore
    ) -> None:
        """The documented trade-off, made explicit.

        Skipping verification trusts the filename alone, so a corrupt shard
        reads as complete. Acceptable only for a progress display.
        """
        store.path_for("unit0001").write_text("{corrupt", encoding="utf-8")
        assert store.exists("unit0001", verify=False)
        assert not store.exists("unit0001", verify=True)


# --------------------------------------------------------------------------- #
# 5. Views and maintenance
# --------------------------------------------------------------------------- #


class TestViews:
    """Aggregating a completed sweep."""

    def test_load_all_is_sorted_and_complete(self, populated: ShardStore) -> None:
        shards = populated.load_all()
        assert len(shards) == 5
        assert [shard.unit_id for shard in shards] == sorted(
            shard.unit_id for shard in shards
        )

    def test_load_all_skips_damaged(self, populated: ShardStore) -> None:
        populated.path_for("unit0002").write_text("{bad", encoding="utf-8")
        assert len(populated.load_all()) == 4

    def test_to_frame_merges_payload_and_metadata(self, populated: ShardStore) -> None:
        """A completed sweep becomes a table with no separate aggregation."""
        pytest.importorskip("pandas")
        frame = populated.to_frame()
        assert len(frame) == 5
        for column in ("unit_id", "written_at", "fold", "subject", "kappa"):
            assert column in frame.columns
        assert frame["kappa"].max() == pytest.approx(0.6)

    def test_len_and_contains(self, populated: ShardStore) -> None:
        assert len(populated) == 5
        assert "unit0003" in populated
        assert "unit0099" not in populated
        assert 42 not in populated

    def test_repr_names_the_directory(self, populated: ShardStore) -> None:
        assert "5 shards" in repr(populated)

    def test_delete(self, populated: ShardStore) -> None:
        assert populated.delete("unit0002")
        assert not populated.delete("unit0002")
        assert len(populated) == 4


class TestPathValidation:
    """Identifiers must be safe filenames on every platform."""

    @pytest.mark.parametrize(
        "unit_id", ["", "a/b", "a\\b", "a:b", "a*b", "a?b", "a b", "a.b", "a|b"]
    )
    def test_unsafe_identifier_rejected(self, unit_id: str, store: ShardStore) -> None:
        """Colons and backslashes are legal on POSIX and not on Windows.

        Rejecting them everywhere keeps a workspace written on one platform
        readable on another, which matters because development happens on
        Windows and runs happen on Colab.
        """
        with pytest.raises(ValueError):
            store.path_for(unit_id)

    def test_hex_digest_identifiers_are_accepted(self, store: ShardStore) -> None:
        """The form ExperimentConfig.unit_id actually produces."""
        assert store.path_for("a3f9c1d2e4b60718").name.startswith("a3f9c1d2")

    def test_root_is_created(self, tmp_path: Path) -> None:
        root = tmp_path / "deeply" / "nested" / "shards"
        assert ShardStore(root).root.is_dir()


class TestIntegrationWithConfig:
    """Unit identifiers come from the configuration layer."""

    def test_config_unit_ids_are_valid_shard_names(self, store: ShardStore) -> None:
        """The two modules must agree on what an identifier looks like."""
        pytest.importorskip("pydantic")
        from geoq.runtime.config import ExperimentConfig

        config = ExperimentConfig(
            name="exp",
            dataset={"name": "bci_iv_2a"},
            pipeline={"steps": [{"name": "mdm"}]},
            protocol={"name": "loso"},
        )
        for fold in range(5):
            unit = config.unit_id(fold=fold)
            store.write(unit, {"kappa": 0.4 + fold / 100}, fold=fold)
        assert len(store.completed_units()) == 5

    def test_changing_the_seed_produces_different_units(
        self, store: ShardStore
    ) -> None:
        """A new seed must not be mistaken for completed work."""
        pytest.importorskip("pydantic")
        from geoq.runtime.config import ExperimentConfig

        base = {
            "name": "exp",
            "dataset": {"name": "bci_iv_2a"},
            "pipeline": {"steps": [{"name": "mdm"}]},
            "protocol": {"name": "loso"},
        }
        first = ExperimentConfig(**base, runtime={"seed": 0})
        second = ExperimentConfig(**base, runtime={"seed": 1})
        units = [first.unit_id(fold=index) for index in range(3)]
        for unit in units:
            store.write(unit, {"kappa": 0.4})
        assert store.pending([second.unit_id(fold=index) for index in range(3)]) == [
            second.unit_id(fold=index) for index in range(3)
        ]
