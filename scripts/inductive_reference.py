"""Fold-wise Riemannian reference: removing the transductive step.

What was transductive
---------------------
Every earlier Paper 2 run projected covariances to the tangent space at the
Riemannian mean of *all* epochs, including those of the held-out subject.
No labels were used and every model saw the same projection, so the
comparisons were not biased, but the pipeline was not strictly inductive.

What this script does
---------------------
The tangent-space reference is fitted on the training subjects of each
fold only, and the held-out subject is projected at that reference. The
in-fold reduction (standardise, PCA, MaxAbs) is unchanged: it already lives
inside each classifier. Everything else -- models, seeds, schedule -- is
identical to ``seeds_stats.py`` and ``readout_control.py``.

Stages (run in order; each is resumable at the level of a single fold)
----------------------------------------------------------------------
core      mlp_24, mlp_8, hybrid                 both datasets, 5 seeds
controls  product_readout, hybrid_no_entangle   both datasets, 5 seeds
readout   hybrid_allz (entangled, mean <Z_j>)   both datasets, 5 seeds
depth     hybrid at 5, 7, 11 layers             2a only, seeds 0-2

``core`` is required for the paper. ``controls`` makes every headline
comparison inductive. ``readout`` removes the readout confound between the
circuit and the product-state control. ``depth`` gives the depth sweep
three seeds under the same pipeline (9 layers comes from ``core``).

Usage (from the repository root, Drive mounted)
-----------------------------------------------
    !python scripts/inductive_reference.py core
    !python scripts/inductive_reference.py summary

Outputs (paper2 results folder)
-------------------------------
inductive_{dataset}.csv           one row per (model, seed, subject)
inductive_vs_transductive.csv     mean kappa per model under both pipelines
inductive_tests.csv               paired tests on the inductive results
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

#  seeds_stats reads sys.argv[1] as its results folder at import time, so
#  the stage argument is removed before anything imports it.
STAGE = sys.argv[1] if len(sys.argv) > 1 else "core"
sys.argv = sys.argv[:1]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hybrid import ReuploadingClassifier  # noqa: E402
from scripts.readout_control import (  # noqa: E402
    ProductReadoutReuploading,
    paired,
)
from scripts.seeds_stats import (  # noqa: E402
    CACHE,
    LAYERS,
    RESULTS,
    SEEDS,
    WIDTHS,
    holm,
    models,
)

from geoq.datasets import load_dataset  # noqa: E402
from geoq.features.covariance import Covariances  # noqa: E402
from geoq.features.tangent_space import TangentSpace  # noqa: E402

DATASETS = tuple(WIDTHS)
DEPTH_SEEDS = (0, 1, 2)

#: Paired comparisons tested on the inductive results, Holm-corrected
#: together within each dataset, matching the manuscript's Table 3.
COMPARISONS = (
    ("hybrid", "hybrid_no_entangle"),
    ("hybrid", "product_readout"),
    ("hybrid", "mlp_24"),
    ("hybrid", "mlp_8"),
)

#: Secondary family for the readout stage, corrected separately: entanglement
#: with the readout held fixed, and the readout with entanglement held fixed.
READOUT_COMPARISONS = (
    ("hybrid_allz", "product_readout"),
    ("hybrid_allz", "hybrid"),
)


class AllQubitReuploading(ReuploadingClassifier):
    """The entangled circuit read out as the mean of every ``<Z_j>``.

    Identical to the main circuit except for the observable. Comparing it
    with the product-state control isolates entanglement with the readout
    held fixed; comparing it with the main circuit isolates the readout.
    """

    def _build(self):
        """Return a callable with the parent's ``circuit(params, x)`` API."""
        import pennylane as qml
        from pennylane import numpy as pnp

        device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(device, interface="autograd", diff_method="backprop")
        def expectations(params, x):
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.RY(x[..., wire % x.shape[-1]], wires=wire)
                for wire in range(self.n_qubits):
                    qml.Rot(*params[layer, wire], wires=wire)
                for wire in range(self.n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        def circuit(params, x):
            return pnp.mean(pnp.stack(expectations(params, x)), axis=0)

        return circuit


def build(name: str, n_qubits: int, n_features: int, seed: int):
    """Return a fresh, unfitted model by name."""
    if name in ("hybrid", "hybrid_no_entangle", "mlp_24", "mlp_8"):
        return models(n_qubits, n_features, seed)[name]
    if name == "product_readout":
        return ProductReadoutReuploading(n_qubits=n_qubits, n_layers=LAYERS, seed=seed)
    if name == "hybrid_allz":
        return AllQubitReuploading(n_qubits=n_qubits, n_layers=LAYERS, seed=seed)
    if name.startswith("hybrid_L"):
        return ReuploadingClassifier(
            n_qubits=n_qubits, n_layers=int(name[len("hybrid_L") :]), seed=seed
        )
    raise ValueError(f"Unknown model {name!r}.")


def plan(stage: str) -> list[tuple[str, str, tuple[int, ...]]]:
    """(dataset, model, seeds) triples for a stage, cheapest models first."""
    groups = {
        "core": ["mlp_24", "mlp_8", "hybrid"],
        "controls": ["product_readout", "hybrid_no_entangle"],
        "readout": ["hybrid_allz"],
    }
    if stage == "depth":
        return [
            ("bci_iv_2a_lr", f"hybrid_L{depth}", DEPTH_SEEDS) for depth in (5, 7, 11)
        ]
    if stage not in groups:
        raise SystemExit(
            f"Unknown stage {stage!r}; use core, controls, readout, depth or summary."
        )
    return [(d, m, SEEDS) for d in DATASETS for m in groups[stage]]


def check_tangent_space(covariances: np.ndarray) -> None:
    """Confirm that fit-then-transform reproduces the pipeline used so far.

    If this fails, the fold-wise projection would differ from the earlier
    one for a reason other than the reference point, and the comparison
    between pipelines would be meaningless.
    """
    subset = covariances[:200]
    joint = TangentSpace().fit_transform(subset)
    split = TangentSpace().fit(subset).transform(subset)
    gap = float(np.abs(joint - split).max())
    print(
        f"  tangent-space check: max |fit_transform - fit.transform| = {gap:.2e}",
        flush=True,
    )
    if gap > 1e-8:
        raise RuntimeError("TangentSpace fit/transform is inconsistent.")


def run(dataset: str, jobs: list[tuple[str, tuple[int, ...]]]) -> None:
    """Evaluate the requested models with a fold-wise reference point."""
    output = RESULTS / f"inductive_{dataset}.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {(r["model"], r["seed"], r["subject"]) for r in records}

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    labels = LabelEncoder().fit_transform(data.labels)
    subjects = np.asarray(data.subjects)
    n_qubits = WIDTHS[dataset]
    n_features = covariances.shape[1] * (covariances.shape[1] + 1) // 2
    check_tangent_space(covariances)

    #  The projection depends only on the fold, not on the model or seed,
    #  so it is computed once per held-out subject and reused.
    folds = {}
    for subject in np.unique(subjects):
        train, test = subjects != subject, subjects == subject
        reference = TangentSpace().fit(covariances[train])
        folds[subject] = (
            train,
            test,
            reference.transform(covariances[train]),
            reference.transform(covariances[test]),
        )

    for name, seeds in jobs:
        for seed in seeds:
            pending = [s for s in folds if (name, seed, s) not in done]
            if not pending:
                print(f"    {name:20s} seed {seed} cached", flush=True)
                continue
            started = time.perf_counter()
            for subject in pending:
                train, test, x_train, x_test = folds[subject]
                model = build(name, n_qubits, n_features, seed)
                model.fit(x_train, labels[train])
                kappa = cohen_kappa_score(labels[test], model.predict(x_test))
                records.append(
                    {
                        "dataset": dataset,
                        "model": name,
                        "seed": seed,
                        "subject": subject,
                        "kappa": float(kappa),
                    }
                )
                #  Written after every fold: a timeout loses one fold at most.
                pd.DataFrame(records).to_csv(output, index=False)
            scores = [
                r["kappa"] for r in records if r["model"] == name and r["seed"] == seed
            ]
            print(
                f"    {name:20s} seed {seed} kappa {np.mean(scores):+.3f} "
                f"({(time.perf_counter() - started) / 60:.1f} min)",
                flush=True,
            )


def transductive(dataset: str) -> pd.DataFrame:
    """The earlier results, for comparison."""
    frames = [pd.read_csv(RESULTS / f"seeds_{dataset}.csv")]
    readout = RESULTS / f"readout_{dataset}.csv"
    if readout.exists():
        frames.append(pd.read_csv(readout))
    return pd.concat(frames, ignore_index=True)


def summary() -> None:
    """Compare pipelines and test the inductive results."""
    shifts, tests = [], []
    for dataset in DATASETS:
        path = RESULTS / f"inductive_{dataset}.csv"
        if not path.exists():
            continue
        new = pd.read_csv(path)
        old = transductive(dataset)
        for name in sorted(new.model.unique()):
            per_subject_new = new[new.model == name].groupby("subject").kappa
            row = {
                "dataset": dataset,
                "model": name,
                "seeds": new[new.model == name].seed.nunique(),
                "inductive": per_subject_new.mean().mean(),
                "inductive_seed_sd": new[new.model == name]
                .groupby("seed")
                .kappa.mean()
                .std(),
            }
            if name in set(old.model):
                a = old[old.model == name].groupby("subject").kappa.mean()
                b = per_subject_new.mean()
                row["transductive"] = a.mean()
                row["shift"] = b.mean() - a.mean()
                row["max_subject_shift"] = float((b - a).abs().max())
            shifts.append(row)

        means = new.groupby(["model", "subject"]).kappa.mean().unstack("model")
        for family, pairs in (
            ("planned", COMPARISONS),
            ("readout", READOUT_COMPARISONS),
        ):
            available = [
                (a, b) for a, b in pairs if a in means.columns and b in means.columns
            ]
            batch = [
                {
                    "dataset": dataset,
                    "family": family,
                    "comparison": f"{a} - {b}",
                    **paired(means[a].to_numpy(), means[b].to_numpy()),
                }
                for a, b in available
            ]
            adjusted_p = holm([t["p_perm"] for t in batch]) if batch else []
            for test, adjusted in zip(batch, adjusted_p, strict=True):
                test["p_holm"] = adjusted
            tests.extend(batch)

    shift_table = pd.DataFrame(shifts)
    shift_table.to_csv(RESULTS / "inductive_vs_transductive.csv", index=False)
    print("\nMean kappa, transductive vs fold-wise reference")
    print(shift_table.round(4).to_string(index=False))
    if tests:
        test_table = pd.DataFrame(tests)
        test_table.to_csv(RESULTS / "inductive_tests.csv", index=False)
        print(
            "\nPaired tests, fold-wise reference (Holm within each dataset and family)"
        )
        print(test_table.round(4).to_string(index=False))


def main() -> None:
    """Run one stage, then print the summary."""
    if STAGE != "summary":
        jobs = plan(STAGE)
        for dataset in DATASETS:
            selected = [(m, s) for d, m, s in jobs if d == dataset]
            if selected:
                print(f"{dataset}  [{STAGE}]", flush=True)
                run(dataset, selected)
    summary()


if __name__ == "__main__":
    main()
