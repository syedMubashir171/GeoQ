"""Validated experiment configuration and deterministic identity.

An experiment in this framework is a YAML file. This module turns that file
into a validated, frozen object and gives it a stable identity, which is what
makes three things possible: reproducibility, resumability, and provenance.

Identity is semantic, not textual
---------------------------------
Two configurations describing the same computation must produce the same hash,
and two describing different computations must not. That rules out hashing the
file: reordering keys, reformatting, adding a comment, or writing ``0.10``
instead of ``0.1`` all change the text without changing the experiment, and
each would silently invalidate hours of completed work on Colab.

So the hash is taken over a canonical JSON rendering of the validated model,
with keys sorted and floats normalised. And it deliberately excludes fields
that describe *how* the computation is executed rather than *what* is
computed:

* ``n_jobs``, ``log_level`` and ``output_root`` are excluded. Raising the
  worker count after a disconnect must resume the run, not restart it.
* ``name`` and ``description`` are excluded. Renaming an experiment does not
  change its science, and forcing a recompute for a typo in a label would
  teach you to never fix labels.
* ``seed`` is **included**. A different seed is a different experiment; a
  framework that treated it as incidental would silently merge results across
  seeds and destroy the variance decomposition that Paper 1 depends on.

Unknown keys are errors
-----------------------
Every model sets ``extra="forbid"``. A YAML file containing ``mtrics:`` or
``protocl:`` fails immediately with the offending key named, rather than
running to completion under the defaults and producing results that answer a
different question than the file appears to specify. This is the single most
valuable thing configuration validation does, and it only works if unknown
keys are rejected rather than ignored.

References
----------
Sculley, D. et al. (2015). Hidden technical debt in machine learning systems.
    *NeurIPS*.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "NON_SEMANTIC_FIELDS",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "PipelineConfig",
    "ProtocolConfig",
    "RuntimeConfig",
    "StatisticsConfig",
    "StepConfig",
    "load_config",
    "save_config",
]

logger = logging.getLogger(__name__)

#: Dotted paths excluded from the experiment hash. Everything here describes
#: how a computation runs, not what it computes.
NON_SEMANTIC_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "runtime.output_root",
        "runtime.n_jobs",
        "runtime.log_level",
        "runtime.overwrite",
    }
)

#: Pipeline step names the framework currently provides. Validated early so a
#: misspelt step fails at load time rather than after the dataset has been
#: downloaded and preprocessed.
KNOWN_STEPS: frozenset[str] = frozenset(
    {
        "covariances",
        "tangent_space",
        "mdm",
        "lda",
        "svm",
        "logistic_regression",
        "scaler",
    }
)


class _Base(BaseModel):
    """Shared configuration behaviour.

    Frozen, because a configuration mutated after its hash was computed would
    describe an experiment that never ran. Extra keys forbidden, so typos are
    errors.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DatasetConfig(_Base):
    """Which recordings to load and how to epoch them.

    Attributes:
        name: Dataset identifier, e.g. ``"bci_iv_2a"``.
        subjects: Subject identifiers to include. Empty means all.
        channels: Channel names to keep. Empty means all.
        tmin: Epoch start relative to the cue, in seconds.
        tmax: Epoch end relative to the cue, in seconds.
        low_freq: High-pass cutoff in hertz.
        high_freq: Low-pass cutoff in hertz.
        resample: Target sampling rate in hertz, or None to keep the original.
    """

    name: str
    subjects: tuple[int, ...] = ()
    channels: tuple[str, ...] = ()
    tmin: float = 0.5
    tmax: float = 2.5
    low_freq: float = 8.0
    high_freq: float = 30.0
    resample: float | None = None

    @model_validator(mode="after")
    def _check_windows(self) -> DatasetConfig:
        """Validate the epoch window and the filter band.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the window or band is inverted, or if the sampling
                rate cannot represent the requested band.
        """
        if self.tmax <= self.tmin:
            raise ValueError(
                f"tmax ({self.tmax}) must exceed tmin ({self.tmin}); the epoch "
                f"window would otherwise be empty or reversed."
            )
        if self.high_freq <= self.low_freq:
            raise ValueError(
                f"high_freq ({self.high_freq}) must exceed low_freq ({self.low_freq})."
            )
        if self.low_freq <= 0:
            raise ValueError(f"low_freq must be positive, got {self.low_freq}.")
        if self.resample is not None and self.resample <= 2 * self.high_freq:
            raise ValueError(
                f"resample ({self.resample} Hz) must exceed twice high_freq "
                f"({2 * self.high_freq} Hz) to satisfy Nyquist. Resampling "
                f"below that aliases the passband into the signal, and the "
                f"result looks like ordinary EEG."
            )
        return self

    @property
    def duration(self) -> float:
        """Epoch length in seconds."""
        return self.tmax - self.tmin

    def expected_samples(self, sampling_rate: float) -> int:
        """Return the number of time samples an epoch will contain.

        Args:
            sampling_rate: Effective sampling rate in hertz.

        Returns:
            Sample count.
        """
        return round(self.duration * sampling_rate)


class StepConfig(_Base):
    """One step of a pipeline.

    Attributes:
        name: Registry key of the step.
        params: Keyword arguments passed to its constructor.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_step(cls, value: str) -> str:
        """Reject unknown step names at load time.

        Args:
            value: The step name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is not registered.
        """
        if value not in KNOWN_STEPS:
            raise ValueError(
                f"Unknown pipeline step {value!r}. Known steps: "
                f"{sorted(KNOWN_STEPS)}. Failing here rather than at build "
                f"time means a typo costs a second, not a dataset download."
            )
        return value


class PipelineConfig(_Base):
    """An ordered sequence of pipeline steps.

    Attributes:
        steps: The steps, in order. The last must be a classifier.
    """

    steps: tuple[StepConfig, ...]

    @field_validator("steps")
    @classmethod
    def _non_empty(cls, value: tuple[StepConfig, ...]) -> tuple[StepConfig, ...]:
        """Require at least one step.

        Args:
            value: The steps.

        Returns:
            The validated steps.

        Raises:
            ValueError: If empty.
        """
        if not value:
            raise ValueError("A pipeline needs at least one step.")
        return value

    @property
    def step_names(self) -> tuple[str, ...]:
        """Names of the steps, in order."""
        return tuple(step.name for step in self.steps)


class ProtocolConfig(_Base):
    """The evaluation protocol.

    Attributes:
        name: Registry key of the splitter.
        params: Keyword arguments passed to its constructor.
    """

    name: Literal[
        "loso",
        "within_subject_chronological",
        "within_subject_kfold",
        "leaky_shuffle",
    ]
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _guard_leaky(self) -> ProtocolConfig:
        """Require an explicit acknowledgement for the unsound protocol.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the leaky protocol is requested without
                acknowledgement. The barrier is repeated here as well as in
                the splitter so that a configuration file cannot select it
                without the acknowledgement being visible in the file itself.
        """
        if self.name == "leaky_shuffle":
            if self.params.get("acknowledge_leakage") is not True:
                raise ValueError(
                    "Protocol 'leaky_shuffle' produces inflated scores by "
                    "design and exists only to quantify that inflation. Set "
                    "params.acknowledge_leakage: true in the configuration to "
                    "confirm this is deliberate, so that the choice is visible "
                    "in the file and in its diff."
                )
            if "random_state" not in self.params:
                raise ValueError(
                    "Protocol 'leaky_shuffle' requires params.random_state. An "
                    "inflation estimate that cannot be reproduced is not a "
                    "measurement."
                )
        return self

    @property
    def is_deliberately_leaky(self) -> bool:
        """Whether this protocol is the unsound control condition."""
        return self.name == "leaky_shuffle"


class EvaluationConfig(_Base):
    """Metrics and hyperparameter selection.

    Attributes:
        metrics: Metric names to compute.
        selection_metric: Metric optimised by the inner search.
        param_grid: Hyperparameter grid, empty for no search.
        inner_protocol: Protocol for the inner search, required when
            ``param_grid`` is non-empty.
        return_train_scores: Whether to score the training folds too.
    """

    metrics: tuple[str, ...] = ("accuracy", "balanced_accuracy", "kappa")
    selection_metric: str = "balanced_accuracy"
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    inner_protocol: ProtocolConfig | None = None
    return_train_scores: bool = False

    @field_validator("metrics")
    @classmethod
    def _non_empty_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require at least one metric, with no duplicates.

        Args:
            value: Metric names.

        Returns:
            The validated names.

        Raises:
            ValueError: If empty or containing duplicates.
        """
        if not value:
            raise ValueError("At least one metric must be requested.")
        if len(set(value)) != len(value):
            raise ValueError(f"Duplicate metrics in {list(value)}.")
        return value

    @model_validator(mode="after")
    def _search_needs_inner_protocol(self) -> EvaluationConfig:
        """Refuse a grid without an inner protocol.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a grid is given without an inner protocol, or if
                the selection metric is not among the computed metrics.
        """
        if self.param_grid and self.inner_protocol is None:
            raise ValueError(
                "param_grid requires inner_protocol. Selecting "
                "hyperparameters once on the whole dataset lets the choice see "
                "the test folds, and inflates every score that follows by an "
                "amount that grows with the size of the grid."
            )
        if self.param_grid and self.selection_metric not in self.metrics:
            raise ValueError(
                f"selection_metric {self.selection_metric!r} is not among the "
                f"computed metrics {list(self.metrics)}, so the reported "
                f"scores would not include the quantity the search optimised."
            )
        return self


class StatisticsConfig(_Base):
    """Inference settings.

    Attributes:
        alpha: Significance level.
        equivalence_bound: TOST margin in the metric's units, or None to skip
            equivalence testing.
        n_permutations: Permutations for the significance test, zero to skip.
        n_bootstrap: Bootstrap resamples, zero to skip.
        power: Power used for the minimum detectable effect.
        correct_multiple_comparisons: Whether to apply Holm-Bonferroni across
            the family of comparisons.
    """

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    equivalence_bound: float | None = Field(default=None, gt=0.0)
    n_permutations: int = Field(default=0, ge=0)
    n_bootstrap: int = Field(default=10000, ge=0)
    power: float = Field(default=0.8, gt=0.0, lt=1.0)
    correct_multiple_comparisons: bool = True


class RuntimeConfig(_Base):
    """Execution settings.

    Every field except ``seed`` is excluded from the experiment hash: they
    describe how the work is carried out, not what is computed. Changing
    ``n_jobs`` after a Colab disconnect must resume the run, not restart it.

    Attributes:
        seed: Master random seed. Semantic, and therefore hashed.
        output_root: Directory holding results, normally on Drive.
        n_jobs: Worker processes.
        log_level: Logging verbosity.
        overwrite: Whether to recompute units that already exist on disk.
    """

    seed: int = 0
    output_root: Path = Path("results")
    n_jobs: int = Field(default=1, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    overwrite: bool = False


class ExperimentConfig(_Base):
    """A complete experiment specification.

    Attributes:
        name: Short identifier, used for the output directory.
        description: Free text. Excluded from the hash.
        dataset: Data selection and preprocessing.
        pipeline: The model pipeline.
        protocol: The evaluation protocol.
        evaluation: Metrics and hyperparameter selection.
        statistics: Inference settings.
        runtime: Execution settings.
    """

    name: str
    description: str = ""
    dataset: DatasetConfig
    pipeline: PipelineConfig
    protocol: ProtocolConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("name")
    @classmethod
    def _filesystem_safe(cls, value: str) -> str:
        """Require a name usable as a directory on every platform.

        Args:
            value: The experiment name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is empty or contains path characters.
                Rejected rather than sanitised, because silently rewriting a
                name would make the output directory disagree with the
                configuration file that produced it.
        """
        if not value.strip():
            raise ValueError("name must not be empty.")
        forbidden = set('/\\:*?"<>| ')
        offending = sorted(forbidden & set(value))
        if offending:
            raise ValueError(
                f"name {value!r} contains characters that are unsafe in a "
                f"directory name: {offending}. Use letters, digits, hyphens "
                f"and underscores."
            )
        return value

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    def semantic_payload(self) -> dict[str, Any]:
        """Return the configuration content that defines the computation.

        Non-semantic fields are removed, so a run can be resumed after
        changing the worker count, the log level, or the output directory.

        Returns:
            A JSON-serialisable dictionary.
        """
        payload = json.loads(self.model_dump_json())
        for dotted in NON_SEMANTIC_FIELDS:
            head, _, tail = dotted.partition(".")
            if not tail:
                payload.pop(head, None)
            elif isinstance(payload.get(head), dict):
                payload[head].pop(tail, None)
        return payload

    @property
    def experiment_hash(self) -> str:
        """Stable SHA-256 hash of the semantic content.

        Canonical JSON with sorted keys, so dictionary insertion order and
        YAML formatting cannot change it. Stable across processes and Python
        versions, which matters because the hash names a directory that a
        later session must find again.

        Returns:
            The hex digest.
        """
        canonical = json.dumps(
            self.semantic_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        """First twelve hex characters of :attr:`experiment_hash`.

        Twelve characters give roughly 2.8e14 possibilities, so a collision is
        implausible across any realistic number of experiments while staying
        short enough to type and to read in a directory listing.
        """
        return self.experiment_hash[:12]

    @property
    def output_dir(self) -> Path:
        """Directory holding this experiment's results.

        Laid out as ``<output_root>/<name>/<short_hash>``. The name is for a
        human scanning the directory; the hash is the identity. Two configs
        differing only in name therefore land in different folders while
        keeping the same hash, so a rename starts a fresh directory rather
        than silently mixing results.
        """
        return self.runtime.output_root / self.name / self.short_hash

    def unit_id(self, **components: Any) -> str:
        """Return a deterministic identifier for one unit of work.

        A unit is the smallest resumable piece of an experiment -- typically a
        dataset, fold and seed. Its identifier is content-addressed, so the
        runner can list the shards already on disk and skip them without
        keeping any global progress state that could be corrupted by a
        disconnect mid-write.

        Args:
            **components: Fields identifying the unit, such as ``fold=3``.

        Returns:
            A sixteen-character hex identifier.
        """
        canonical = json.dumps(
            {"experiment": self.experiment_hash, **components},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """Return a one-line human summary for logs."""
        leaky = " [LEAKY PROTOCOL]" if self.protocol.is_deliberately_leaky else ""
        return (
            f"{self.name} ({self.short_hash}): {self.dataset.name} -> "
            f"{' -> '.join(self.pipeline.step_names)} under "
            f"{self.protocol.name}, seed {self.runtime.seed}{leaky}"
        )


# --------------------------------------------------------------------------- #
# Loading and saving
# --------------------------------------------------------------------------- #


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, is not a mapping, fails validation,
            or records a hash that disagrees with its contents. Validation
            errors name the offending key, which is the point of forbidding
            unknown fields.
    """
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No configuration file at {config_path}.")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"Configuration file {config_path} is empty.")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Configuration file {config_path} must contain a mapping at the "
            f"top level, got {type(raw).__name__}."
        )

    #  Keys beginning with an underscore are metadata written by
    #  save_config -- currently the recorded hash. They are stripped here
    #  rather than declared as fields, so that the archived copy beside a set
    #  of results reloads cleanly. Without this the file the framework itself
    #  writes would be rejected by the framework's own loader, which would
    #  defeat the point of archiving it.
    recorded_hash = raw.pop("_experiment_hash", None)
    raw = {key: value for key, value in raw.items() if not key.startswith("_")}

    config = ExperimentConfig(**raw)

    if recorded_hash is not None and recorded_hash != config.experiment_hash:
        #  The file was hand-edited after the run, or the hashing rules
        #  changed between framework versions. Either way the results in the
        #  neighbouring directory were produced by a different specification
        #  than this file now describes, and silently continuing would attach
        #  those numbers to the wrong configuration.
        raise ValueError(
            f"Configuration file {config_path} records hash "
            f"{recorded_hash[:12]} but its contents hash to "
            f"{config.short_hash}. The file has been edited since it was "
            f"written, or was produced by a different version of the "
            f"framework. Results stored alongside it do not correspond to "
            f"this specification."
        )
    logger.info("Loaded configuration: %s", config.describe())
    if config.protocol.is_deliberately_leaky:
        logger.warning(
            "Configuration %r selects the deliberately leaky protocol. Its "
            "scores are a control condition, not a result.",
            config.name,
        )
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write a configuration to YAML, creating parent directories.

    Every experiment saves its own configuration beside its results. The file
    the run was launched from can be edited, moved, or lost; the copy in the
    output directory is what makes the numbers reproducible a year later.

    Args:
        config: The configuration.
        path: Destination path.

    Returns:
        The path written.
    """
    import yaml

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(config.model_dump_json())
    payload["_experiment_hash"] = config.experiment_hash

    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True, default_flow_style=False)
    return destination
