"""Config-driven experiment runner with per-fold resumption.

Turns a YAML file into an archived result. Everything the earlier layers built
converges here: a validated configuration names the dataset, pipeline,
protocol and seed; its hash names the output directory; each fold is
checkpointed as it completes; and a session that dies partway through resumes
from the fold after the last one that finished.

Folds are the unit of work
--------------------------
:func:`geoq.evaluation.protocol.evaluate` runs every fold in one call, which
is right for a notebook and wrong for an overnight sweep: a disconnect in fold
eight discards seven completed folds. The runner therefore drives the splitter
itself and writes a shard per fold, reusing
:func:`geoq.evaluation.protocol.evaluate_fold` so that the quick path and the
archived path cannot diverge.

Fold identity is content-addressed. A shard is named by
``config.unit_id(fold=i)``, which depends on the experiment hash and therefore
on the seed, the dataset, the pipeline and the protocol -- but not on the
worker count or the output directory. Changing how a run executes resumes it;
changing what it computes starts a new one.

What lands beside the results
-----------------------------
Three files, written before the first fold rather than after the last, because
a run that dies in hour five should still leave behind enough to know what it
was doing:

* ``config.yaml`` -- the specification, including its hash.
* ``provenance.json`` -- interpreter, platform, package versions, git commit
  and whether the working tree was dirty.
* ``shards/`` -- one file per completed fold.

The git commit is the load-bearing one. Six months later, "which code produced
this number" has exactly one answer, and a dirty working tree is recorded as
such so the answer is not quietly wrong.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geoq.datasets.base import EEGDataset, load_dataset
from geoq.evaluation.protocol import EvaluationResult, FoldResult, evaluate_fold
from geoq.evaluation.splitters import make_splitter
from geoq.runtime.checkpoint import ShardStore
from geoq.runtime.config import ExperimentConfig, PipelineConfig, save_config

ALIGNMENT_STEP = "alignment"

__all__ = [
    "ALIGNMENT_STEP",
    "STEP_BUILDERS",
    "RunSummary",
    "build_pipeline",
    "collect_provenance",
    "run_experiment",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #


def _covariances(**params: Any):
    from geoq.features.covariance import Covariances

    return Covariances(**params)


def _alignment(**params: Any):
    from geoq.features.alignment import RiemannianAlignment

    return RiemannianAlignment(**{"assume_calibration_data": True, **params})


def _tangent_space(**params: Any):
    from geoq.features.tangent_space import TangentSpace

    return TangentSpace(**params)


def _mdm(**params: Any):
    from geoq.models.classical.mdm import MDM

    return MDM(**params)


def _lda(**params: Any):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return LinearDiscriminantAnalysis(**params)


def _svm(**params: Any):
    from sklearn.svm import SVC

    return SVC(**{"probability": True, **params})


def _logistic_regression(**params: Any):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**{"max_iter": 1000, **params})


def _scaler(**params: Any):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler(**params)


STEP_BUILDERS: dict[str, Any] = {
    "covariances": _covariances,
    "alignment": _alignment,
    "tangent_space": _tangent_space,
    "mdm": _mdm,
    "lda": _lda,
    "svm": _svm,
    "logistic_regression": _logistic_regression,
    "scaler": _scaler,
}
"""Mapping from configuration step name to a constructor.

Imports are deferred into the builders so that this module loads without
scikit-learn present, keeping the runtime layer importable in the same bare
environment the geometry layer is.
"""


def build_pipeline(config: PipelineConfig):
    """Construct a scikit-learn pipeline from its configuration.

    Args:
        config: The pipeline specification.

    Returns:
        A :class:`sklearn.pipeline.Pipeline`.

    Raises:
        ValueError: If a step name is unknown, or its parameters are rejected
            by the constructor. Both fail here, before the dataset is loaded,
            because a typo should cost a second rather than a download.
    """
    from sklearn.pipeline import Pipeline

    steps = []
    for index, step in enumerate(config.steps):
        try:
            builder = STEP_BUILDERS[step.name]
        except KeyError:
            raise ValueError(
                f"Pipeline step {step.name!r} has no builder. Available: "
                f"{sorted(STEP_BUILDERS)}."
            ) from None
        try:
            estimator = builder(**step.params)
        except TypeError as error:
            raise ValueError(
                f"Step {step.name!r} rejected its parameters {step.params}: {error}"
            ) from error
        steps.append((f"{index}_{step.name}", estimator))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def _git(command: list[str], cwd: Path | None) -> str:
    """Run a git command, returning empty string on any failure."""
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def collect_provenance(
    config: ExperimentConfig, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Record everything needed to reproduce a run.

    Args:
        config: The experiment specification.
        repo_root: Repository to query for the git commit. Defaults to the
            directory containing the installed package.

    Returns:
        A JSON-serialisable provenance record.
    """
    from importlib.metadata import PackageNotFoundError, version

    root = repo_root or Path(__file__).resolve().parents[3]
    commit = _git(["rev-parse", "HEAD"], root)
    dirty = bool(_git(["status", "--porcelain"], root))

    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "pyriemann",
        "mne",
        "moabb",
        "geoq",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"

    record = {
        "experiment_hash": config.experiment_hash,
        "experiment_name": config.name,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
        "packages": packages,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": config.runtime.seed,
    }

    if dirty:
        logger.warning(
            "The working tree has uncommitted changes, so commit %s does not "
            "describe the code about to run. This result will not be "
            "reproducible from the repository alone; commit before launching a "
            "run whose numbers will be reported.",
            (commit or "unknown")[:12],
        )
    return record


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunSummary:
    """What one call to :func:`run_experiment` did.

    Attributes:
        result: The assembled evaluation result.
        output_dir: Directory holding the config, provenance and shards.
        n_folds: Total folds in the experiment.
        n_computed: Folds computed by this call.
        n_resumed: Folds already on disk and skipped.
        seconds: Wall-clock duration of this call.
    """

    result: EvaluationResult
    output_dir: Path
    n_folds: int
    n_computed: int
    n_resumed: int
    seconds: float

    def describe(self) -> str:
        """Return a one-line summary for a log."""
        headline = self.result.metrics[0]
        return (
            f"{self.n_computed} computed / {self.n_resumed} resumed of "
            f"{self.n_folds} folds in {self.seconds:.1f}s; "
            f"{headline}={self.result.mean(headline):.3f} -> {self.output_dir}"
        )


def run_experiment(
    config: ExperimentConfig,
    *,
    dataset: EEGDataset | None = None,
    repo_root: Path | None = None,
) -> RunSummary:
    """Execute an experiment, resuming any folds already on disk.

    Args:
        config: The validated specification.
        dataset: Preloaded data. When None the dataset named in the config is
            loaded through the registry.
        repo_root: Repository for the git commit in the provenance record.

    Returns:
        A summary including the assembled result.

    Raises:
        ValueError: If the pipeline or protocol cannot be constructed, or the
            dataset is unsuitable.
    """
    started = time.perf_counter()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    provenance = collect_provenance(config, repo_root=repo_root)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    store = ShardStore(output_dir / "shards")
    if config.runtime.overwrite:
        removed = [store.delete(unit) for unit in store.completed_units()]
        logger.info("overwrite=True: discarded %d existing shard(s)", sum(removed))

    #  Built before the data is loaded: a typo in the pipeline should fail in a
    #  second rather than after a download.
    prefix_config, pipeline_config, alignment_step = _split_at_alignment(
        config.pipeline
    )
    pipeline = build_pipeline(pipeline_config)
    splitter = make_splitter(config.protocol.name, **config.protocol.params)

    data = dataset if dataset is not None else _load_from_config(config)
    logger.info("%s | %s", config.describe(), data)

    features = _apply_preprocessing(data, prefix_config, alignment_step)
    folds = list(splitter.split(features, data.labels, data.subjects))
    unit_ids = [config.unit_id(fold=index) for index in range(len(folds))]
    pending = set(store.pending(unit_ids))

    n_computed = 0
    for index, (train_index, test_index) in enumerate(folds):
        unit = unit_ids[index]
        if unit not in pending:
            continue

        fold_started = time.perf_counter()
        fold_result = evaluate_fold(
            pipeline,
            features,
            data.labels,
            train_index=train_index,
            test_index=test_index,
            fold=index,
            groups=data.subjects,
            metrics=config.evaluation.metrics,
            param_grid=dict(config.evaluation.param_grid) or None,
            inner_splitter=(
                None
                if config.evaluation.inner_protocol is None
                else make_splitter(
                    config.evaluation.inner_protocol.name,
                    **config.evaluation.inner_protocol.params,
                )
            ),
            selection_metric=config.evaluation.selection_metric,
            return_train_scores=config.evaluation.return_train_scores,
        )

        #  Written immediately, not batched at the end. A shard held in memory
        #  until the sweep finishes is a shard lost to the first disconnect.
        store.write(
            unit,
            _fold_payload(fold_result),
            fold=index,
            experiment=config.experiment_hash,
        )
        n_computed += 1
        logger.info(
            "fold %d/%d done in %.1fs (%s)",
            index + 1,
            len(folds),
            time.perf_counter() - fold_started,
            ", ".join(
                f"{key}={value:.3f}" for key, value in fold_result.scores.items()
            ),
        )

    result = _assemble(config, store, unit_ids, splitter, data, pipeline)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    summary = RunSummary(
        result=result,
        output_dir=output_dir,
        n_folds=len(folds),
        n_computed=n_computed,
        n_resumed=len(folds) - n_computed,
        seconds=time.perf_counter() - started,
    )
    logger.info("%s", summary.describe())
    return summary


def _split_at_alignment(
    config: PipelineConfig,
) -> tuple[PipelineConfig | None, PipelineConfig, Any]:
    """Separate the steps that run before per-subject alignment.

    Alignment cannot be an ordinary pipeline step: it needs each trial's
    subject, and ``Pipeline`` does not forward ``groups`` to ``transform``.
    Rather than smuggle that through a global, the runner splits the pipeline
    at the alignment step, applies the prefix plus alignment once, and treats
    the remainder as the estimator.

    Applying alignment before folding is not a shortcut. A subject's reference
    point is computed from that subject's trials alone, so aligning the whole
    dataset gives bitwise the same result as recomputing it inside every fold
    -- the property asserted by
    ``tests/features/test_alignment.py::test_domains_are_independent``. It is
    still transductive, which is what ``assume_calibration_data`` declares.

    Args:
        config: The full pipeline specification.

    Returns:
        Tuple of the prefix configuration (or None), the remaining pipeline,
        and the alignment step (or None).

    Raises:
        ValueError: If alignment appears more than once, or is not preceded
            solely by covariance estimation. Alignment operates on SPD
            matrices, so anything else before it would hand it the wrong kind
            of object.
    """
    names = config.step_names
    if ALIGNMENT_STEP not in names:
        return None, config, None
    if names.count(ALIGNMENT_STEP) > 1:
        raise ValueError(
            f"Pipeline contains {names.count(ALIGNMENT_STEP)} alignment steps. "
            f"Re-centring twice is a no-op the second time and signals a "
            f"configuration mistake."
        )

    index = names.index(ALIGNMENT_STEP)
    prefix = config.steps[:index]
    if [step.name for step in prefix] != ["covariances"]:
        raise ValueError(
            f"Alignment must be preceded by exactly one 'covariances' step, "
            f"got {[step.name for step in prefix]}. Re-centring operates on "
            f"SPD matrices; any other preceding step would hand it something "
            f"that is not one."
        )
    return (
        PipelineConfig(steps=prefix),
        PipelineConfig(steps=config.steps[index + 1 :]),
        config.steps[index],
    )


def _apply_preprocessing(
    data: EEGDataset, prefix: PipelineConfig | None, alignment_step: Any
) -> Any:
    """Run the pre-alignment steps and align, or return the raw epochs.

    Args:
        data: The dataset.
        prefix: Steps to apply before alignment, or None when no alignment is
            configured.
        alignment_step: The alignment step configuration, or None.

    Returns:
        The array the folds will be computed over.
    """
    if prefix is None:
        return data.epochs

    covariances = build_pipeline(prefix).fit_transform(data.epochs)
    aligner = _alignment(**alignment_step.params)
    aligned = aligner.fit_transform(covariances, domains=data.subjects)

    from geoq.features.alignment import alignment_quality

    quality = alignment_quality(aligned, data.subjects)
    logger.info(
        "Aligned %d subjects; residual distance to the identity: max %.2e. "
        "Alignment is transductive: each subject's reference uses that "
        "subject's own unlabelled trials.",
        int(quality["n_domains"]),
        quality["max_residual"],
    )
    return aligned


def _load_from_config(config: ExperimentConfig) -> EEGDataset:
    """Load the dataset named in a configuration.

    Args:
        config: The experiment specification.

    Returns:
        The dataset.
    """
    dataset_config = config.dataset
    kwargs: dict[str, Any] = {
        "tmin": dataset_config.tmin,
        "tmax": dataset_config.tmax,
        "low_freq": dataset_config.low_freq,
        "high_freq": dataset_config.high_freq,
        "resample": dataset_config.resample,
    }
    if dataset_config.subjects:
        kwargs["subjects"] = list(dataset_config.subjects)
    if dataset_config.channels:
        kwargs["channels"] = list(dataset_config.channels)
    if dataset_config.name == "synthetic":
        kwargs = {"seed": config.runtime.seed}
    return load_dataset(dataset_config.name, **kwargs)


def _fold_payload(fold: FoldResult) -> dict[str, Any]:
    """Convert a fold result into a shard payload."""
    payload: dict[str, Any] = {
        "fold": fold.fold,
        "n_train": fold.n_train,
        "n_test": fold.n_test,
        "test_groups": list(fold.test_groups),
        "fit_seconds": fold.fit_seconds,
        "score_seconds": fold.score_seconds,
        **fold.scores,
    }
    if fold.train_scores is not None:
        payload.update({f"train_{k}": v for k, v in fold.train_scores.items()})
    if fold.best_params is not None:
        payload["best_params"] = {str(k): v for k, v in fold.best_params.items()}
    return payload


def _assemble(
    config: ExperimentConfig,
    store: ShardStore,
    unit_ids: list[str],
    splitter: Any,
    data: EEGDataset,
    pipeline: Any,
) -> EvaluationResult:
    """Rebuild an :class:`EvaluationResult` from the shards on disk.

    Reading back rather than accumulating in memory is deliberate: it means a
    resumed run and a run that completed in one session produce the same
    object, assembled the same way, so the two cannot silently differ.

    Args:
        config: The experiment specification.
        store: The shard store.
        unit_ids: Expected unit identifiers, in fold order.
        splitter: The protocol, for its guarantees.
        data: The dataset, for class balance.
        pipeline: The estimator, for its representation.

    Returns:
        The assembled result.

    Raises:
        RuntimeError: If a fold is missing after the run.
    """
    by_unit = {shard.unit_id: shard for shard in store.load_all()}
    missing = [unit for unit in unit_ids if unit not in by_unit]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(unit_ids)} folds are missing from "
            f"{store.root} after the run. Shards may have been damaged after "
            f"being written; re-run to recompute them."
        )

    metrics = tuple(config.evaluation.metrics)
    folds = []
    for index, unit in enumerate(unit_ids):
        payload = by_unit[unit].payload
        folds.append(
            FoldResult(
                fold=index,
                scores={name: float(payload[name]) for name in metrics},
                n_train=int(payload["n_train"]),
                n_test=int(payload["n_test"]),
                test_groups=tuple(payload.get("test_groups", ())),
                fit_seconds=float(payload.get("fit_seconds", 0.0)),
                score_seconds=float(payload.get("score_seconds", 0.0)),
                best_params=payload.get("best_params"),
            )
        )

    return EvaluationResult(
        folds=tuple(folds),
        protocol=splitter.info,
        estimator_repr=repr(pipeline),
        metrics=metrics,
        n_samples=data.n_trials,
        n_classes=int(data.classes.size),
        class_balance={str(k): v for k, v in data.class_balance.items()},
        chance_accuracy=data.chance_accuracy,
        param_grid=dict(config.evaluation.param_grid) or None,
    )
