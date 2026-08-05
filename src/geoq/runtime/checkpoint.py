"""Atomic, content-addressed storage for resumable experiments.

Colab disconnects. A long leave-one-subject-out sweep across datasets, models
and seeds is thousands of independent units of work, and losing all of it
because the session dropped in hour five is unacceptable. This module is what
makes the work survivable.

No central index
----------------
Progress is not tracked in a manifest. Completion is discovered by listing the
shard directory, so the record of what is done *is* the set of files that
exist. A manifest would be a single point of failure: a disconnect while
appending to it leaves a file that is either truncated or disagrees with the
directory, and a wrong manifest is worse than none -- it causes finished work
to be redone or, far worse, unfinished work to be skipped.

Writes are atomic
-----------------
Each shard is written to a temporary file in the same directory, flushed and
``fsync``-ed, then moved into place with :func:`os.replace`, which is atomic on
POSIX and on Windows for same-volume moves. A shard is therefore either absent
or complete; a half-written one cannot exist. The temporary name includes the
process and thread identifier so that parallel workers cannot collide.

Corruption means recompute, not crash
-------------------------------------
Every shard carries a checksum of its payload. Google Drive is a network
filesystem with its own sync semantics, and atomicity guarantees that hold on
a local disk are not guaranteed to survive it. So a shard that fails
verification is treated as *missing*: it is logged, and the unit is recomputed.
The alternative -- raising -- would turn one damaged file into a failed
overnight run, which is precisely the outcome this module exists to prevent.

Payloads are JSON
-----------------
Slower and larger than a binary format, and chosen anyway. A results shard is
a provenance artefact that a reader may need to inspect in five years, and a
format that requires this codebase to read it is a liability. NumPy arrays and
scalars are converted on the way out; the conversion is lossy for dtype, so
anything that must round-trip exactly belongs in a separate array file rather
than in a shard.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SHARD_SUFFIX",
    "CorruptShardError",
    "Shard",
    "ShardStore",
    "to_jsonable",
]

logger = logging.getLogger(__name__)

SHARD_SUFFIX: str = ".json"
"""Extension used for shard files."""

_TEMP_PREFIX: str = ".tmp-"
"""Prefix for in-progress writes, so they are ignored by directory listing."""


class CorruptShardError(RuntimeError):
    """Raised when a shard fails verification and the caller asked to know."""


def to_jsonable(value: Any) -> Any:
    """Convert NumPy and Path objects into JSON-serialisable equivalents.

    Applied recursively. Arrays become nested lists and NumPy scalars become
    Python scalars, which loses dtype: a float32 array reloads as float64.
    That is acceptable for the summary statistics a shard holds, and is the
    reason anything requiring bit-exact round-tripping should be stored as a
    separate array file instead.

    Args:
        value: The object to convert.

    Returns:
        A JSON-serialisable equivalent.

    Raises:
        TypeError: If the object has no JSON equivalent. Raised rather than
            silently stringified, because a stringified object in a results
            file looks like data and is not.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    #  Checked before the general scalar branch, and covering plain Python
    #  floats as well as NumPy ones. np.float64 subclasses float, so an
    #  isinstance check against float alone would swallow NaN and infinity
    #  and write them out as bare literals -- a file that Python's json
    #  module reads back and no other parser accepts.
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return converted if np.isfinite(converted) else str(converted)
    if isinstance(value, int):
        return value
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    raise TypeError(
        f"Cannot serialise {type(value).__name__} to JSON. Convert it "
        f"explicitly, or give it a to_dict method. Stringifying it here would "
        f"put something that looks like data, and is not, into a results file."
    )


def _checksum(payload: Any) -> str:
    """Return a stable SHA-256 digest of a JSON-serialisable payload.

    Args:
        payload: Already converted by :func:`to_jsonable`.

    Returns:
        The hex digest.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Shard:
    """One completed unit of work.

    Attributes:
        unit_id: Content-addressed identifier from
            :meth:`geoq.runtime.config.ExperimentConfig.unit_id`.
        payload: The result itself.
        metadata: Context recorded alongside it, such as the fold index or the
            held-out subject.
        checksum: SHA-256 of the payload, verified on read.
        written_at: UTC ISO-8601 timestamp.
    """

    unit_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    written_at: str = ""

    def verify(self) -> bool:
        """Return whether the payload still matches its recorded checksum."""
        return bool(self.checksum) and _checksum(self.payload) == self.checksum


class ShardStore:
    """A directory of shards, one file per unit of work.

    Args:
        root: Directory holding the shards. Created if absent.
        strict: If True, reading a corrupt shard raises
            :class:`CorruptShardError` instead of treating it as missing.
            Leave False for experiment runs, where recomputing one unit is
            always better than failing the whole sweep; set True in tests and
            when auditing a completed run.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     store = ShardStore(directory)
        ...     _ = store.write("unit0001", {"kappa": 0.42}, fold=0)
        ...     store.exists("unit0001")
        True
    """

    def __init__(self, root: str | Path, *, strict: bool = False) -> None:
        """Create the store, making the directory if necessary.

        Args:
            root: Directory holding the shards.
            strict: Whether corrupt shards raise instead of being ignored.
        """
        self.root = Path(root)
        self.strict = strict
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def path_for(self, unit_id: str) -> Path:
        """Return the file path for a unit.

        Args:
            unit_id: The unit identifier.

        Returns:
            The shard path.

        Raises:
            ValueError: If the identifier is empty or contains characters that
                are unsafe in a filename. Identifiers come from a hex digest,
                so anything else indicates a caller constructing them by hand.
        """
        if not unit_id:
            raise ValueError("unit_id must not be empty.")
        unsafe = set(unit_id) & set('/\\:*?"<>| .')
        if unsafe:
            raise ValueError(
                f"unit_id {unit_id!r} contains characters unsafe in a "
                f"filename: {sorted(unsafe)}. Identifiers should come from "
                f"ExperimentConfig.unit_id, which returns a hex digest."
            )
        return self.root / f"{unit_id}{SHARD_SUFFIX}"

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def write(self, unit_id: str, payload: dict[str, Any], **metadata: Any) -> Path:
        """Write one shard atomically.

        The file is created under a temporary name in the same directory,
        flushed to disk, then moved into place. Same directory matters: a move
        across filesystems is a copy and is not atomic, and on Colab the
        system temporary directory is on a different device from Drive.

        Args:
            unit_id: The unit identifier.
            payload: The result. Must be JSON-serialisable after
                :func:`to_jsonable`.
            **metadata: Context stored alongside the payload.

        Returns:
            The path written.
        """
        destination = self.path_for(unit_id)
        serialisable = to_jsonable(payload)

        record = {
            "unit_id": unit_id,
            "payload": serialisable,
            "metadata": to_jsonable(metadata),
            "checksum": _checksum(serialisable),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }

        # A unique temporary name per process and thread, so parallel workers
        # writing different units cannot collide on the same partial file.
        handle, temporary = tempfile.mkstemp(
            prefix=f"{_TEMP_PREFIX}{os.getpid()}-{threading.get_ident()}-",
            suffix=SHARD_SUFFIX,
            dir=self.root,
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                #  allow_nan=False makes it impossible to write a file that
                #  only Python can read. If a non-finite value ever reaches
                #  here, the write fails loudly rather than producing a
                #  results file that a reviewer's tooling cannot parse.
                json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.flush()
                # Without fsync the data may sit in the page cache when the
                # rename lands, so a crash between the two leaves a shard that
                # exists and is empty -- which is exactly the state the atomic
                # write is meant to make impossible.
                os.fsync(stream.fileno())
            #  Path.replace delegates to os.replace: atomic on POSIX, and on
            #  Windows for a same-volume move, which is why the temporary
            #  file is created in the destination directory.
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        return destination

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def read(self, unit_id: str) -> Shard | None:
        """Read one shard, returning None if it is absent or corrupt.

        Args:
            unit_id: The unit identifier.

        Returns:
            The shard, or None.

        Raises:
            CorruptShardError: If the store is strict and the shard fails to
                parse or verify.
        """
        path = self.path_for(unit_id)
        if not path.is_file():
            return None
        return self._load(path)

    def _load(self, path: Path) -> Shard | None:
        """Parse and verify a shard file.

        Args:
            path: The shard path.

        Returns:
            The shard, or None when it is unusable and the store is lenient.

        Raises:
            CorruptShardError: If the store is strict.
        """
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            shard = Shard(
                unit_id=record["unit_id"],
                payload=record["payload"],
                metadata=record.get("metadata", {}),
                checksum=record.get("checksum", ""),
                written_at=record.get("written_at", ""),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            return self._handle_damage(path, f"could not be parsed: {error}")

        if not shard.verify():
            return self._handle_damage(path, "failed checksum verification")
        return shard

    def _handle_damage(self, path: Path, reason: str) -> None:
        """Log or raise for a damaged shard.

        Args:
            path: The shard path.
            reason: Human description of the damage.

        Returns:
            None, when the store is lenient.

        Raises:
            CorruptShardError: When the store is strict.
        """
        message = (
            f"Shard {path.name} {reason}. Treating the unit as incomplete so "
            f"it will be recomputed. Damaged shards are usually a Drive sync "
            f"interrupted mid-write; if many appear at once, check the "
            f"workspace path and available quota."
        )
        if self.strict:
            raise CorruptShardError(message)
        logger.warning("%s", message)
        return None

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _shard_paths(self) -> list[Path]:
        """Return the shard files, excluding in-progress temporaries."""
        return sorted(
            path
            for path in self.root.glob(f"*{SHARD_SUFFIX}")
            if not path.name.startswith(_TEMP_PREFIX)
        )

    def exists(self, unit_id: str, *, verify: bool = True) -> bool:
        """Return whether a usable shard exists for a unit.

        Args:
            unit_id: The unit identifier.
            verify: Whether to parse and checksum the file. Verification costs
                a read per unit, which is negligible against recomputing a
                cross-validation fold. Disable it only when scanning tens of
                thousands of shards for a progress display.

        Returns:
            True when the unit can be skipped.
        """
        path = self.path_for(unit_id)
        if not path.is_file():
            return False
        if not verify:
            return True
        return self.read(unit_id) is not None

    def completed_units(self, *, verify: bool = True) -> set[str]:
        """Return the identifiers of every usable shard.

        Discovered by listing the directory, so the filesystem is the single
        source of truth and there is no index to fall out of step with it.

        Args:
            verify: Whether to checksum each shard.

        Returns:
            The set of completed unit identifiers.
        """
        completed: set[str] = set()
        for path in self._shard_paths():
            if not verify:
                completed.add(path.stem)
                continue
            shard = self._load(path)
            if shard is not None:
                completed.add(shard.unit_id)
        return completed

    def pending(self, unit_ids: list[str], *, verify: bool = True) -> list[str]:
        """Return the units still to be computed, in the order given.

        The call a runner makes on startup. Order is preserved so a resumed
        run proceeds in the same sequence as the original, which keeps logs
        comparable across sessions.

        Args:
            unit_ids: Every unit the experiment requires.
            verify: Whether to checksum existing shards.

        Returns:
            The subset not yet completed.
        """
        completed = self.completed_units(verify=verify)
        return [unit_id for unit_id in unit_ids if unit_id not in completed]

    def load_all(self) -> list[Shard]:
        """Return every usable shard, sorted by identifier.

        Returns:
            The shards.
        """
        shards = [self._load(path) for path in self._shard_paths()]
        return [shard for shard in shards if shard is not None]

    def to_frame(self):
        """Return a tidy table with one row per shard.

        Payload and metadata keys become columns, so a completed sweep can be
        analysed directly without a separate aggregation step.

        Returns:
            A :class:`pandas.DataFrame`.
        """
        import pandas as pd

        rows = []
        for shard in self.load_all():
            row: dict[str, Any] = {
                "unit_id": shard.unit_id,
                "written_at": shard.written_at,
            }
            row.update(shard.metadata)
            row.update(shard.payload)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def delete(self, unit_id: str) -> bool:
        """Remove one shard.

        Args:
            unit_id: The unit identifier.

        Returns:
            True if a file was removed.
        """
        path = self.path_for(unit_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def clear_corrupt(self) -> list[str]:
        """Delete every shard that fails verification.

        Not called automatically. A run that quietly deletes damaged results
        destroys the evidence of whatever damaged them, and a cluster of
        corrupt shards is a signal about the storage worth investigating
        before it is erased.

        Returns:
            The names of the files removed.
        """
        removed: list[str] = []
        for path in self._shard_paths():
            try:
                with path.open(encoding="utf-8") as stream:
                    record = json.load(stream)
                shard = Shard(
                    unit_id=record["unit_id"],
                    payload=record["payload"],
                    checksum=record.get("checksum", ""),
                )
                healthy = shard.verify()
            except (OSError, ValueError, KeyError, TypeError):
                healthy = False
            if not healthy:
                path.unlink()
                removed.append(path.name)
        if removed:
            logger.warning("Removed %d corrupt shard(s): %s", len(removed), removed)
        return removed

    def clear_temporary(self) -> list[str]:
        """Delete leftover temporary files from interrupted writes.

        These are inert -- directory listing already ignores them -- but they
        accumulate across disconnects and clutter a Drive folder.

        Returns:
            The names of the files removed.
        """
        removed: list[str] = []
        for path in self.root.glob(f"{_TEMP_PREFIX}*"):
            path.unlink(missing_ok=True)
            removed.append(path.name)
        return removed

    def __len__(self) -> int:
        """Return the number of shard files, without verifying them."""
        return len(self._shard_paths())

    def __contains__(self, unit_id: object) -> bool:
        """Return whether a usable shard exists for ``unit_id``."""
        return isinstance(unit_id, str) and self.exists(unit_id)

    def __repr__(self) -> str:
        """Return a representation naming the directory and shard count."""
        return f"ShardStore({self.root!s}, {len(self)} shards)"
