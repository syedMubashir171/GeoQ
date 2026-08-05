"""Tests for :mod:`geoq.runtime.config`.

What is being defended
----------------------
* **Hash stability.** ``TestHashStability`` fixes exactly which changes alter
  an experiment's identity. Getting this wrong is expensive in both
  directions: an over-sensitive hash discards completed Colab work every time
  a worker count changes, and an under-sensitive one silently merges results
  from different seeds or different pipelines into one directory.
* **Typos are errors.** ``TestUnknownKeys`` checks that a misspelt key fails
  with the key named. A configuration system that ignores unknown fields is
  worse than none: the run completes, the results look plausible, and they
  answer a different question than the file appears to ask.
* **The leaky protocol needs acknowledgement in the file.** The barrier is
  duplicated from the splitter into the configuration layer so the decision is
  visible in the YAML and in its diff, not buried in a constructor call.
"""

from __future__ import annotations

import copy
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pydantic", reason="requires the 'ml' extra")
pytest.importorskip("yaml", reason="requires the 'ml' extra")

from pydantic import ValidationError

from geoq.runtime.config import (
    NON_SEMANTIC_FIELDS,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    PipelineConfig,
    ProtocolConfig,
    RuntimeConfig,
    StatisticsConfig,
    StepConfig,
    load_config,
    save_config,
)

BASE: dict[str, Any] = {
    "name": "paper1_reckoning",
    "description": "LOSO against the leaky control",
    "dataset": {"name": "bci_iv_2a", "tmin": 0.5, "tmax": 2.5},
    "pipeline": {
        "steps": [
            {"name": "covariances", "params": {"estimator": "oas"}},
            {"name": "tangent_space"},
            {"name": "lda"},
        ]
    },
    "protocol": {"name": "loso"},
}


def variant(**overrides: Any) -> ExperimentConfig:
    """Build a configuration from ``BASE`` with the given overrides."""
    payload = copy.deepcopy(BASE)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return ExperimentConfig(**payload)


@pytest.fixture
def config() -> ExperimentConfig:
    """The reference configuration."""
    return ExperimentConfig(**copy.deepcopy(BASE))


# --------------------------------------------------------------------------- #
# 1. Hash stability
# --------------------------------------------------------------------------- #


class TestHashStability:
    """Exactly which changes alter an experiment's identity."""

    def test_identical_configs_agree(self, config: ExperimentConfig) -> None:
        assert ExperimentConfig(**copy.deepcopy(BASE)).experiment_hash == (
            config.experiment_hash
        )

    def test_key_order_is_irrelevant(self, config: ExperimentConfig) -> None:
        """Reformatting a YAML file must not discard completed work."""
        reordered = {key: BASE[key] for key in reversed(list(BASE))}
        assert ExperimentConfig(**reordered).experiment_hash == (config.experiment_hash)

    def test_float_spelling_is_irrelevant(self, config: ExperimentConfig) -> None:
        """``0.5``, ``0.50`` and ``5e-1`` are the same number."""
        for spelling in (0.50, 5e-1):
            assert variant(dataset={"tmin": spelling}).experiment_hash == (
                config.experiment_hash
            )

    @pytest.mark.parametrize(
        "runtime_override",
        [
            {"n_jobs": 16},
            {"log_level": "DEBUG"},
            {"output_root": "/content/drive/MyDrive/elsewhere"},
            {"overwrite": True},
        ],
    )
    def test_execution_settings_do_not_change_identity(
        self, runtime_override: dict, config: ExperimentConfig
    ) -> None:
        """The property Colab resumability depends on.

        Reconnecting with a different worker count, a louder log, or a
        relocated Drive folder must resume the run. If any of these changed
        the hash, every disconnect would restart the experiment from zero.
        """
        assert variant(runtime=runtime_override).experiment_hash == (
            config.experiment_hash
        )

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_labels_do_not_change_identity(
        self, field: str, config: ExperimentConfig
    ) -> None:
        """Renaming an experiment does not change its science."""
        assert variant(**{field: "something_else"}).experiment_hash == (
            config.experiment_hash
        )

    def test_seed_does_change_identity(self, config: ExperimentConfig) -> None:
        """A different seed is a different experiment.

        Treating it as incidental would merge results across seeds into one
        directory and destroy the variance decomposition that separates
        seed-driven variation from method-driven variation.
        """
        assert variant(runtime={"seed": 1}).experiment_hash != config.experiment_hash

    @pytest.mark.parametrize(
        "override",
        [
            {"dataset": {"tmin": 1.0}},
            {"dataset": {"low_freq": 4.0}},
            {"dataset": {"name": "bci_iv_2b"}},
            {"protocol": {"name": "within_subject_kfold"}},
            {"evaluation": {"metrics": ["accuracy"]}},
            {"statistics": {"alpha": 0.01}},
        ],
    )
    def test_scientific_settings_change_identity(
        self, override: dict, config: ExperimentConfig
    ) -> None:
        assert variant(**override).experiment_hash != config.experiment_hash

    def test_pipeline_changes_identity(self, config: ExperimentConfig) -> None:
        changed = copy.deepcopy(BASE)
        changed["pipeline"]["steps"][0]["params"]["estimator"] = "scm"
        assert ExperimentConfig(**changed).experiment_hash != config.experiment_hash

    def test_step_order_changes_identity(self, config: ExperimentConfig) -> None:
        """A pipeline is ordered; reversing it is a different computation."""
        reversed_steps = copy.deepcopy(BASE)
        reversed_steps["pipeline"]["steps"] = reversed_steps["pipeline"]["steps"][::-1]
        assert ExperimentConfig(**reversed_steps).experiment_hash != (
            config.experiment_hash
        )

    def test_stable_across_processes(self, config: ExperimentConfig) -> None:
        """Verified in a subprocess with a randomised hash seed.

        Python randomises string hashing per process, so an implementation
        built on ``hash()`` or on set iteration order would produce a
        different digest in every session -- and a Colab run resumed the next
        morning would find no completed work. Canonical JSON avoids that, and
        this test proves it rather than assuming it.
        """
        script = (
            "import json,sys;"
            "from geoq.runtime.config import ExperimentConfig;"
            "print(ExperimentConfig(**json.loads(sys.argv[1])).experiment_hash)"
        )
        import json as json_module
        import os

        #  Inherit the environment and override only the hash seed. Replacing
        #  it wholesale fails on Windows, where the interpreter needs
        #  SYSTEMROOT and its own PATH entries to start at all: the subprocess
        #  exits 1 and the test reports a portability problem as a hashing
        #  problem. On POSIX a stripped environment happens to work, which is
        #  exactly why this went unnoticed until it ran on Windows.
        environment = {**os.environ, "PYTHONHASHSEED": "random"}

        completed = subprocess.run(
            [sys.executable, "-c", script, json_module.dumps(BASE)],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        assert completed.stdout.strip() == config.experiment_hash

    def test_short_hash_is_a_prefix(self, config: ExperimentConfig) -> None:
        assert config.experiment_hash.startswith(config.short_hash)
        assert len(config.short_hash) == 12

    def test_non_semantic_fields_are_absent_from_the_payload(
        self, config: ExperimentConfig
    ) -> None:
        payload = config.semantic_payload()
        assert "name" not in payload
        assert "description" not in payload
        for dotted in NON_SEMANTIC_FIELDS:
            head, _, tail = dotted.partition(".")
            if tail:
                assert tail not in payload.get(head, {})
        assert "seed" in payload["runtime"]


class TestOutputLayout:
    """Where results land, and why."""

    def test_directory_includes_name_and_hash(self, config: ExperimentConfig) -> None:
        assert config.output_dir == Path("results") / BASE["name"] / config.short_hash

    def test_rename_gives_a_new_directory_but_the_same_hash(
        self, config: ExperimentConfig
    ) -> None:
        """The name is for humans; the hash is the identity."""
        renamed = variant(name="renamed_experiment")
        assert renamed.experiment_hash == config.experiment_hash
        assert renamed.output_dir != config.output_dir

    def test_output_root_is_respected(self) -> None:
        """Compared as paths, not as strings.

        ``Path`` renders with the host separator, so a POSIX-style literal
        never prefixes a ``WindowsPath``'s string form. Comparing through
        ``parents`` keeps the assertion about path structure rather than about
        the platform the tests happen to run on.
        """
        drive = Path("/content/drive/MyDrive/GeoQ_workspace/results")
        output_dir = variant(runtime={"output_root": str(drive)}).output_dir
        assert drive in output_dir.parents
        assert output_dir.parts[: len(drive.parts)] == drive.parts


class TestUnitId:
    """Content-addressed identifiers for resumable work."""

    def test_deterministic(self, config: ExperimentConfig) -> None:
        assert config.unit_id(fold=3, subject=7) == config.unit_id(fold=3, subject=7)

    def test_keyword_order_is_irrelevant(self, config: ExperimentConfig) -> None:
        assert config.unit_id(fold=3, subject=7) == config.unit_id(subject=7, fold=3)

    def test_distinct_units_differ(self, config: ExperimentConfig) -> None:
        identifiers = {config.unit_id(fold=index) for index in range(50)}
        assert len(identifiers) == 50

    def test_depends_on_the_experiment(self, config: ExperimentConfig) -> None:
        """Otherwise two experiments would collide on their shard filenames."""
        assert config.unit_id(fold=1) != variant(runtime={"seed": 9}).unit_id(fold=1)

    def test_survives_execution_setting_changes(self, config: ExperimentConfig) -> None:
        """The resumability property, at the unit level."""
        assert config.unit_id(fold=1) == variant(runtime={"n_jobs": 8}).unit_id(fold=1)

    def test_handles_non_string_components(self, config: ExperimentConfig) -> None:
        assert isinstance(config.unit_id(subject=Path("s01"), fold=None), str)


# --------------------------------------------------------------------------- #
# 2. Unknown keys and validation
# --------------------------------------------------------------------------- #


class TestUnknownKeys:
    """A typo must fail loudly, naming the key."""

    @pytest.mark.parametrize(
        ("section", "payload"),
        [
            ("dataset", {"name": "d", "tmim": 0.5}),
            ("evaluation", {"mtrics": ["accuracy"]}),
            ("statistics", {"alpah": 0.05}),
            ("runtime", {"n_job": 4}),
            ("protocol", {"name": "loso", "parms": {}}),
        ],
    )
    def test_misspelt_key_rejected(self, section: str, payload: dict) -> None:
        """The single most valuable thing this layer does.

        Ignoring the key would let the run complete under defaults and produce
        results that answer a different question than the file specifies --
        undetectable from the outputs alone.
        """
        with pytest.raises(ValidationError) as excinfo:
            variant(**{section: payload})
        assert "Extra inputs" in str(excinfo.value)

    def test_top_level_typo_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            ExperimentConfig(**{**BASE, "pipline": {}})

    def test_unknown_step_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown pipeline step"):
            variant(pipeline={"steps": [{"name": "tangentspace"}]})

    def test_unknown_protocol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            variant(protocol={"name": "loso_kfold"})


class TestDatasetValidation:
    """Windows and bands must be physically sensible."""

    def test_inverted_epoch_window_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must exceed tmin"):
            variant(dataset={"tmin": 2.0, "tmax": 1.0})

    def test_inverted_band_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must exceed low_freq"):
            variant(dataset={"low_freq": 30.0, "high_freq": 8.0})

    def test_nyquist_violation_rejected(self) -> None:
        """Resampling below twice the passband aliases it into the signal.

        The result looks like ordinary EEG and decodes like noise, which is
        why this has to be caught in the configuration rather than diagnosed
        from the outputs.
        """
        with pytest.raises(ValidationError, match="Nyquist"):
            variant(dataset={"high_freq": 30.0, "resample": 50.0})

    def test_valid_resample_accepted(self) -> None:
        assert variant(dataset={"high_freq": 30.0, "resample": 128.0})

    def test_duration_and_sample_count(self) -> None:
        dataset = DatasetConfig(name="d", tmin=0.5, tmax=2.5)
        assert dataset.duration == pytest.approx(2.0)
        assert dataset.expected_samples(250.0) == 500


class TestEvaluationValidation:
    """Metric and search consistency."""

    def test_empty_metrics_rejected(self) -> None:
        with pytest.raises(ValidationError, match="At least one metric"):
            variant(evaluation={"metrics": []})

    def test_duplicate_metrics_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate metrics"):
            variant(evaluation={"metrics": ["accuracy", "accuracy"]})

    def test_grid_without_inner_protocol_rejected(self) -> None:
        """Nested selection is not optional when a grid is present."""
        with pytest.raises(ValidationError, match="requires inner_protocol"):
            variant(evaluation={"param_grid": {"mdm__metric": ["airm"]}})

    def test_selection_metric_must_be_computed(self) -> None:
        with pytest.raises(ValidationError, match="not among the computed"):
            variant(
                evaluation={
                    "metrics": ["accuracy"],
                    "selection_metric": "kappa",
                    "param_grid": {"mdm__metric": ["airm"]},
                    "inner_protocol": {"name": "within_subject_kfold"},
                }
            )

    def test_valid_nested_search_accepted(self) -> None:
        assert variant(
            evaluation={
                "param_grid": {"mdm__metric": ["airm", "logeuclid"]},
                "inner_protocol": {"name": "within_subject_kfold"},
            }
        )


class TestProtocolValidation:
    """The leaky protocol needs acknowledgement in the file."""

    def test_leaky_without_acknowledgement_rejected(self) -> None:
        with pytest.raises(ValidationError, match="acknowledge_leakage"):
            variant(protocol={"name": "leaky_shuffle", "params": {"random_state": 0}})

    def test_leaky_without_seed_rejected(self) -> None:
        """An inflation estimate that cannot be reproduced is not a measurement."""
        with pytest.raises(ValidationError, match="random_state"):
            variant(
                protocol={
                    "name": "leaky_shuffle",
                    "params": {"acknowledge_leakage": True},
                }
            )

    def test_truthy_acknowledgement_is_not_enough(self) -> None:
        """``1`` in a YAML file reads as an unrelated integer setting."""
        with pytest.raises(ValidationError, match="acknowledge_leakage"):
            variant(
                protocol={
                    "name": "leaky_shuffle",
                    "params": {"acknowledge_leakage": 1, "random_state": 0},
                }
            )

    def test_properly_acknowledged_leaky_protocol_accepted(self) -> None:
        config = variant(
            protocol={
                "name": "leaky_shuffle",
                "params": {"acknowledge_leakage": True, "random_state": 0},
            }
        )
        assert config.protocol.is_deliberately_leaky
        assert "LEAKY PROTOCOL" in config.describe()

    def test_honest_protocol_is_not_flagged(self, config: ExperimentConfig) -> None:
        assert not config.protocol.is_deliberately_leaky
        assert "LEAKY" not in config.describe()


class TestFieldConstraints:
    """Numeric ranges enforced by the schema."""

    @pytest.mark.parametrize(
        "override",
        [
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"equivalence_bound": 0.0},
            {"equivalence_bound": -0.1},
            {"n_permutations": -1},
            {"power": 1.5},
        ],
    )
    def test_out_of_range_statistics_rejected(self, override: dict) -> None:
        with pytest.raises(ValidationError):
            variant(statistics=override)

    def test_zero_workers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            variant(runtime={"n_jobs": 0})

    def test_unknown_log_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            variant(runtime={"log_level": "TRACE"})

    @pytest.mark.parametrize("name", ["", "   ", "a/b", "a b", "a:b", "a*b"])
    def test_unsafe_experiment_name_rejected(self, name: str) -> None:
        """Rejected rather than sanitised.

        Silently rewriting the name would make the output directory disagree
        with the configuration file that produced it, so a reader could not
        find the results a config refers to.
        """
        with pytest.raises(ValidationError):
            variant(name=name)

    def test_empty_pipeline_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one step"):
            variant(pipeline={"steps": []})


class TestImmutability:
    """Configurations are frozen once validated."""

    def test_cannot_mutate(self, config: ExperimentConfig) -> None:
        """A config mutated after hashing describes an experiment that never ran."""
        with pytest.raises(ValidationError):
            config.name = "changed"  # type: ignore[misc]

    def test_nested_models_are_frozen(self, config: ExperimentConfig) -> None:
        with pytest.raises(ValidationError):
            config.dataset.tmin = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 3. Round trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    """Saving and loading preserves identity."""

    def test_save_then_load_preserves_the_hash(
        self, config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """The property that makes an archived config reproducible.

        The launching file can be edited or lost; the copy written beside the
        results is what lets a reader re-run the experiment a year later and
        land in the same output directory.
        """
        written = save_config(config, tmp_path / "nested" / "config.yaml")
        assert written.is_file()
        assert load_config(written).experiment_hash == config.experiment_hash

    def test_saved_file_records_the_hash(
        self, config: ExperimentConfig, tmp_path: Path
    ) -> None:
        import yaml

        path = save_config(config, tmp_path / "config.yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["_experiment_hash"] == config.experiment_hash

    def test_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        original = variant(
            statistics={"n_permutations": 500, "equivalence_bound": 0.02},
            runtime={"seed": 7, "n_jobs": 4},
        )
        path = save_config(original, tmp_path / "c.yaml")
        reloaded = load_config(path)
        assert reloaded.statistics == original.statistics
        assert reloaded.runtime.seed == original.runtime.seed

    def test_archived_config_reloads_directly(
        self, config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """The file the framework writes must load in the framework.

        ``save_config`` records the hash as ``_experiment_hash``. Under
        ``extra="forbid"`` that key would be rejected by the loader, so the
        archived copy beside a set of results would be unreadable -- defeating
        the reason for archiving it. Underscore-prefixed metadata is therefore
        stripped on load.
        """
        path = save_config(config, tmp_path / "archived.yaml")
        assert load_config(path).experiment_hash == config.experiment_hash

    def test_edited_archive_is_rejected(
        self, config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """A hand-edited archive must not silently claim the old results.

        If the recorded hash disagrees with the contents, the numbers stored
        in the neighbouring directory were produced by a different
        specification than the file now describes.
        """
        import yaml

        path = save_config(config, tmp_path / "archived.yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["dataset"]["tmin"] = 1.25
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="records hash"):
            load_config(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.yaml")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="is empty"):
            load_config(path)

    def test_non_mapping_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping at the top level"):
            load_config(path)

    def test_load_warns_for_the_leaky_protocol(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = variant(
            protocol={
                "name": "leaky_shuffle",
                "params": {"acknowledge_leakage": True, "random_state": 0},
            }
        )
        path = save_config(config, tmp_path / "c.yaml")
        with caplog.at_level(logging.WARNING, logger="geoq.runtime.config"):
            load_config(path)
        assert "leaky protocol" in caplog.text

    def test_realistic_yaml_loads(self, tmp_path: Path) -> None:
        """A hand-written file in the form a user would actually create."""
        path = tmp_path / "paper1.yaml"
        path.write_text(
            """
name: paper1_loso_baseline
description: |
  Classical Riemannian baseline under strict LOSO.
dataset:
  name: bci_iv_2a
  tmin: 0.5
  tmax: 2.5
  low_freq: 8.0
  high_freq: 30.0
pipeline:
  steps:
    - name: covariances
      params:
        estimator: oas
    - name: tangent_space
      params:
        metric: airm
    - name: lda
protocol:
  name: loso
evaluation:
  metrics: [accuracy, balanced_accuracy, kappa]
statistics:
  alpha: 0.05
  equivalence_bound: 0.02
  n_permutations: 1000
runtime:
  seed: 42
  output_root: /content/drive/MyDrive/GeoQ_workspace/results
""",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.pipeline.step_names == (
            "covariances",
            "tangent_space",
            "lda",
        )
        assert config.statistics.n_permutations == 1000
        assert config.runtime.seed == 42


class TestComponentDefaults:
    """Defaults are the honest choices, not the convenient ones."""

    def test_metrics_include_kappa_by_default(self) -> None:
        """Accuracy alone is uninterpretable across class balances."""
        assert "kappa" in EvaluationConfig().metrics

    def test_permutations_default_to_zero(self) -> None:
        """Expensive, so opt-in rather than a surprise in a first run."""
        assert StatisticsConfig().n_permutations == 0

    def test_overwrite_defaults_to_false(self) -> None:
        """Resuming is the default; destroying prior work must be requested."""
        assert RuntimeConfig().overwrite is False

    def test_components_construct_standalone(self) -> None:
        assert StepConfig(name="mdm").params == {}
        assert PipelineConfig(steps=[StepConfig(name="mdm")]).step_names == ("mdm",)
        assert ProtocolConfig(name="loso").params == {}
