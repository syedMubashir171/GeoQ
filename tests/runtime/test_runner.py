"""Tests for :mod:`geoq.runtime.runner`.

What is being defended
----------------------
* **Resumption is exact.** ``TestResumption`` interrupts a run partway,
  restarts it, and asserts the assembled result is bitwise identical to one
  computed in a single session. A resume that produced *nearly* the same
  numbers would be worse than one that failed: the difference would appear
  only in a table, months later, with no way to tell which session produced
  which row.
* **One code path.** The runner and
  :func:`geoq.evaluation.protocol.evaluate` must agree, because one produces
  the quick answer in a notebook and the other the archived one.
  ``TestAgreementWithEvaluate`` pins that.
* **Provenance is written before the work.** A run that dies in hour five
  should still leave behind enough to know what it was doing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the 'ml' extra")
pytest.importorskip("pydantic", reason="requires the 'ml' extra")

from geoq.datasets.base import make_synthetic_eeg
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import make_splitter
from geoq.runtime.checkpoint import ShardStore
from geoq.runtime.config import ExperimentConfig, PipelineConfig, StepConfig
from geoq.runtime.runner import (
    STEP_BUILDERS,
    build_pipeline,
    collect_provenance,
    run_experiment,
)


@pytest.fixture
def dataset():
    """A small synthetic dataset, so runs finish in seconds."""
    return make_synthetic_eeg(
        n_subjects=4, n_trials_per_subject=24, n_channels=6, n_times=128, seed=0
    )


def make_config(output_root: Path, **overrides: Any) -> ExperimentConfig:
    """Build a runnable configuration."""
    payload: dict[str, Any] = {
        "name": "test_run",
        "dataset": {"name": "synthetic"},
        "pipeline": {
            "steps": [
                {
                    "name": "covariances",
                    "params": {"estimator": "oas", "audit_conditioning": False},
                },
                {"name": "tangent_space"},
                {"name": "lda"},
            ]
        },
        "protocol": {"name": "loso", "params": {"min_subjects": 2}},
        "runtime": {"seed": 0, "output_root": str(output_root)},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return ExperimentConfig(**payload)


# --------------------------------------------------------------------------- #
# 1. Pipeline construction
# --------------------------------------------------------------------------- #


class TestBuildPipeline:
    """Configuration to estimator, failing early on mistakes."""

    def test_builds_the_named_steps_in_order(self) -> None:
        pipeline = build_pipeline(
            PipelineConfig(
                steps=[
                    StepConfig(name="covariances"),
                    StepConfig(name="tangent_space"),
                    StepConfig(name="lda"),
                ]
            )
        )
        assert [name for name, _ in pipeline.steps] == [
            "0_covariances",
            "1_tangent_space",
            "2_lda",
        ]

    def test_parameters_reach_the_estimator(self) -> None:
        pipeline = build_pipeline(
            PipelineConfig(
                steps=[
                    StepConfig(name="covariances", params={"estimator": "ledoit_wolf"}),
                    StepConfig(name="mdm", params={"metric": "logeuclid"}),
                ]
            )
        )
        assert pipeline.steps[0][1].estimator == "ledoit_wolf"
        assert pipeline.steps[1][1].metric == "logeuclid"

    def test_bad_parameters_fail_before_any_data_is_loaded(self) -> None:
        """A typo should cost a second, not a download."""
        with pytest.raises(ValueError, match="rejected its parameters"):
            build_pipeline(
                PipelineConfig(
                    steps=[StepConfig(name="mdm", params={"metrik": "airm"})]
                )
            )

    def test_every_registered_step_builds(self) -> None:
        for name in STEP_BUILDERS:
            assert build_pipeline(PipelineConfig(steps=[StepConfig(name=name)]))

    def test_config_known_steps_all_have_builders(self) -> None:
        """The two registries must agree.

        A step the config accepts but the runner cannot build would pass
        validation and fail at launch, which is the worst place for it.
        """
        from geoq.runtime.config import KNOWN_STEPS

        assert set(KNOWN_STEPS) == set(STEP_BUILDERS)


# --------------------------------------------------------------------------- #
# 2. A complete run
# --------------------------------------------------------------------------- #


class TestRun:
    """What one execution produces."""

    def test_produces_one_fold_per_subject(self, tmp_path: Path, dataset) -> None:
        summary = run_experiment(make_config(tmp_path), dataset=dataset)
        assert summary.n_folds == 4
        assert summary.n_computed == 4
        assert summary.n_resumed == 0
        assert len(summary.result.folds) == 4

    def test_writes_the_expected_artefacts(self, tmp_path: Path, dataset) -> None:
        """Enough to reconstruct the run from the directory alone."""
        summary = run_experiment(make_config(tmp_path), dataset=dataset)
        for name in ("config.yaml", "provenance.json", "summary.json", "shards"):
            assert (summary.output_dir / name).exists()

    def test_output_directory_is_named_by_the_hash(
        self, tmp_path: Path, dataset
    ) -> None:
        config = make_config(tmp_path)
        summary = run_experiment(config, dataset=dataset)
        assert summary.output_dir.name == config.short_hash
        assert summary.output_dir.parent.name == config.name

    def test_archived_config_reloads_to_the_same_hash(
        self, tmp_path: Path, dataset
    ) -> None:
        """The archived copy must be usable, not just present."""
        from geoq.runtime.config import load_config

        config = make_config(tmp_path)
        summary = run_experiment(config, dataset=dataset)
        reloaded = load_config(summary.output_dir / "config.yaml")
        assert reloaded.experiment_hash == config.experiment_hash

    def test_one_shard_per_fold(self, tmp_path: Path, dataset) -> None:
        summary = run_experiment(make_config(tmp_path), dataset=dataset)
        store = ShardStore(summary.output_dir / "shards")
        assert len(store.completed_units()) == 4
        for shard in store.load_all():
            assert {"n_train", "n_test", "kappa"} <= set(shard.payload)

    def test_summary_json_matches_the_result(self, tmp_path: Path, dataset) -> None:
        summary = run_experiment(make_config(tmp_path), dataset=dataset)
        written = json.loads((summary.output_dir / "summary.json").read_text())
        assert written["protocol"] == "loso"
        assert written["kappa_mean"] == pytest.approx(summary.result.mean("kappa"))

    def test_protocol_travels_into_the_result(self, tmp_path: Path, dataset) -> None:
        summary = run_experiment(make_config(tmp_path), dataset=dataset)
        assert summary.result.protocol.name == "loso"
        assert not summary.result.protocol.is_optimistic


class TestAgreementWithEvaluate:
    """The runner and the direct call must not diverge."""

    def test_identical_results(self, tmp_path: Path, dataset) -> None:
        """One produces the notebook answer, the other the archived one.

        They share :func:`geoq.evaluation.protocol.evaluate_fold` precisely so
        that this holds; the test is what keeps the sharing real.
        """
        config = make_config(tmp_path)
        via_runner = run_experiment(config, dataset=dataset).result
        via_evaluate = evaluate(
            build_pipeline(config.pipeline),
            dataset.epochs,
            dataset.labels,
            groups=dataset.subjects,
            splitter=make_splitter("loso", min_subjects=2),
            metrics=config.evaluation.metrics,
        )
        assert np.array_equal(via_runner.scores("kappa"), via_evaluate.scores("kappa"))
        assert via_runner.fold_sizes == via_evaluate.fold_sizes


# --------------------------------------------------------------------------- #
# 3. Resumption
# --------------------------------------------------------------------------- #


class TestResumption:
    """A dead session must resume, not restart."""

    def test_second_run_computes_nothing(self, tmp_path: Path, dataset) -> None:
        config = make_config(tmp_path)
        run_experiment(config, dataset=dataset)
        second = run_experiment(config, dataset=dataset)
        assert second.n_computed == 0
        assert second.n_resumed == 4

    def test_resumed_result_is_bitwise_identical(self, tmp_path: Path, dataset) -> None:
        """Nearly-identical would be worse than failing.

        A discrepancy would surface only as a number in a table months later,
        with no way to tell which session produced which row.
        """
        config = make_config(tmp_path)
        first = run_experiment(config, dataset=dataset).result
        second = run_experiment(config, dataset=dataset).result
        for metric in first.metrics:
            assert np.array_equal(first.scores(metric), second.scores(metric))

    def test_partial_run_resumes_from_the_right_fold(
        self, tmp_path: Path, dataset
    ) -> None:
        """The scenario the whole design exists for.

        Two folds are deleted to simulate a session that died after the
        second, and the restart must recompute exactly those two.
        """
        config = make_config(tmp_path)
        complete = run_experiment(config, dataset=dataset).result
        store = ShardStore(config.output_dir / "shards")
        for fold in (2, 3):
            assert store.delete(config.unit_id(fold=fold))

        resumed = run_experiment(config, dataset=dataset)
        assert resumed.n_computed == 2
        assert resumed.n_resumed == 2
        assert np.array_equal(resumed.result.scores("kappa"), complete.scores("kappa"))

    def test_execution_settings_do_not_invalidate_shards(
        self, tmp_path: Path, dataset
    ) -> None:
        """Reconnecting with a different worker count must resume.

        This is the property that makes a Colab disconnect survivable, and it
        rests on the experiment hash excluding non-semantic fields.
        """
        run_experiment(make_config(tmp_path), dataset=dataset)
        second = run_experiment(
            make_config(tmp_path, runtime={"n_jobs": 4, "log_level": "DEBUG"}),
            dataset=dataset,
        )
        assert second.n_computed == 0

    def test_a_different_seed_starts_a_new_experiment(
        self, tmp_path: Path, dataset
    ) -> None:
        """A new seed is a new experiment, not resumable work."""
        first = run_experiment(make_config(tmp_path), dataset=dataset)
        second = run_experiment(
            make_config(tmp_path, runtime={"seed": 1}), dataset=dataset
        )
        assert second.n_computed == 4
        assert second.output_dir != first.output_dir

    def test_overwrite_discards_existing_shards(self, tmp_path: Path, dataset) -> None:
        run_experiment(make_config(tmp_path), dataset=dataset)
        again = run_experiment(
            make_config(tmp_path, runtime={"overwrite": True}), dataset=dataset
        )
        assert again.n_computed == 4

    def test_corrupt_shard_is_recomputed(self, tmp_path: Path, dataset) -> None:
        """Damage on Drive must cost one fold, not the whole run."""
        config = make_config(tmp_path)
        run_experiment(config, dataset=dataset)
        store = ShardStore(config.output_dir / "shards")
        store.path_for(config.unit_id(fold=1)).write_text("{corrupt")

        resumed = run_experiment(config, dataset=dataset)
        assert resumed.n_computed == 1
        assert len(resumed.result.folds) == 4

    def test_missing_fold_after_the_run_is_fatal(
        self, tmp_path: Path, dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assembling from an incomplete set would report a partial mean.

        A result computed over three of four folds that presents itself as
        complete is the kind of error no downstream check can catch.
        """
        config = make_config(tmp_path)
        run_experiment(config, dataset=dataset)

        import geoq.runtime.runner as runner_module

        original = runner_module.ShardStore.load_all
        monkeypatch.setattr(
            runner_module.ShardStore,
            "load_all",
            lambda self: original(self)[:-1],
        )
        with pytest.raises(RuntimeError, match="folds are missing"):
            run_experiment(config, dataset=dataset)


# --------------------------------------------------------------------------- #
# 4. Provenance
# --------------------------------------------------------------------------- #


class TestProvenance:
    """Which code produced which number, with one answer."""

    def test_records_the_essentials(self, tmp_path: Path) -> None:
        record = collect_provenance(make_config(tmp_path))
        for key in (
            "experiment_hash",
            "python_version",
            "platform",
            "git_commit",
            "git_dirty",
            "packages",
            "seed",
        ):
            assert key in record
        assert "numpy" in record["packages"]

    def test_hash_matches_the_config(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        assert collect_provenance(config)["experiment_hash"] == (config.experiment_hash)

    def test_is_json_serialisable(self, tmp_path: Path) -> None:
        assert json.loads(json.dumps(collect_provenance(make_config(tmp_path))))

    def test_dirty_tree_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A recorded commit that does not describe the running code.

        The most common reason a result cannot be reproduced later, and silent
        unless it is said out loud at launch.
        """
        import geoq.runtime.runner as runner_module

        original = runner_module._git

        def dirty(command: list[str], cwd: Any) -> str:
            return (
                "M src/geoq/x.py" if command[0] == "status" else original(command, cwd)
            )

        runner_module._git = dirty
        try:
            with caplog.at_level(logging.WARNING, logger="geoq.runtime.runner"):
                record = collect_provenance(make_config(tmp_path))
        finally:
            runner_module._git = original

        assert record["git_dirty"] is True
        assert "uncommitted changes" in caplog.text

    def test_written_before_the_first_fold(
        self, tmp_path: Path, dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that dies early still says what it was doing."""
        import geoq.runtime.runner as runner_module

        config = make_config(tmp_path)
        seen: dict[str, bool] = {}

        def failing_fold(*args: Any, **kwargs: Any):
            seen["config"] = (config.output_dir / "config.yaml").is_file()
            seen["provenance"] = (config.output_dir / "provenance.json").is_file()
            raise KeyboardInterrupt("simulated disconnect")

        monkeypatch.setattr(runner_module, "evaluate_fold", failing_fold)
        with pytest.raises(KeyboardInterrupt):
            run_experiment(config, dataset=dataset)

        assert seen == {"config": True, "provenance": True}


# --------------------------------------------------------------------------- #
# 5. Configuration coverage
# --------------------------------------------------------------------------- #


class TestConfigurationVariants:
    """Settings in the file reach the run."""

    def test_metrics_are_respected(self, tmp_path: Path, dataset) -> None:
        summary = run_experiment(
            make_config(tmp_path, evaluation={"metrics": ["accuracy", "f1_macro"]}),
            dataset=dataset,
        )
        assert summary.result.metrics == ("accuracy", "f1_macro")

    def test_protocol_is_respected(self, tmp_path: Path, dataset) -> None:
        summary = run_experiment(
            make_config(
                tmp_path,
                protocol={"name": "within_subject_kfold", "params": {"n_splits": 3}},
            ),
            dataset=dataset,
        )
        assert summary.n_folds == 3
        assert summary.result.protocol.is_optimistic

    def test_leaky_protocol_runs_when_acknowledged(
        self, tmp_path: Path, dataset
    ) -> None:
        """Paper 1's control condition, driven from a config file.

        The acknowledgement lives in the YAML, so a reader can see from the
        configuration alone that the run was a deliberate control.
        """
        summary = run_experiment(
            make_config(
                tmp_path,
                protocol={
                    "name": "leaky_shuffle",
                    "params": {"acknowledge_leakage": True, "random_state": 0},
                },
            ),
            dataset=dataset,
        )
        assert summary.result.protocol.is_optimistic
        assert summary.n_folds == 5

    def test_nested_search_records_chosen_parameters(
        self, tmp_path: Path, dataset
    ) -> None:
        summary = run_experiment(
            make_config(
                tmp_path,
                pipeline={
                    "steps": [
                        {
                            "name": "covariances",
                            "params": {"audit_conditioning": False},
                        },
                        {"name": "mdm"},
                    ]
                },
                evaluation={
                    "param_grid": {"1_mdm__metric": ["airm", "logeuclid"]},
                    "inner_protocol": {
                        "name": "within_subject_kfold",
                        "params": {"n_splits": 2},
                    },
                },
            ),
            dataset=dataset,
        )
        assert all(fold.best_params is not None for fold in summary.result.folds)

    def test_dataset_is_loaded_from_the_registry(self, tmp_path: Path) -> None:
        """No preloaded data: the config names the dataset and that is enough."""
        summary = run_experiment(make_config(tmp_path))
        assert summary.n_folds == 9
        assert summary.result.n_samples == 648
