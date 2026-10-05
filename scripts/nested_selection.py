"""Nested selection of the circuit depth and the MLP width.

The problem this fixes
----------------------
The main experiments fix the circuit at nine layers and the MLP at 24
hidden units. Both numbers came from sweeps scored on the same
leave-one-subject-out folds that the comparison reports, so each model's
architecture was chosen with knowledge of the test folds. The direction of
that bias favours whichever family was swept more carefully, and no amount
of argument removes it from the reported numbers.

Here the architecture is chosen inside each outer training fold and never
sees the held-out subject.

Protocol
--------
Outer loop: leave one subject out, nine folds.
Inner loop: of the eight training subjects, six train and two validate.
            Every candidate is fitted on the six and scored on the two;
            the best is refitted on all eight and applied to the held-out
            subject.

The two inner validation subjects rotate with the seed, so a configuration
is not selected against one arbitrary split. Both families use the same
inner protocol, so neither is selected more finely than the other.

Candidates
----------
circuit : 5, 7, 9 and 11 re-uploading layers at the dataset's qubit count
mlp     : 8, 16, 24 and 64 hidden units, with L2 penalty 1e-4 or 1e-2

Cost
----
The circuit dominates: per outer fold, four inner fits on six subjects
plus one refit on eight. With three seeds and both datasets this is
roughly 30 hours, checkpointed after every outer fold.

Usage (repository root, Drive mounted)
--------------------------------------
    !python scripts/nested_selection.py           # runs both families
    !python scripts/nested_selection.py mlp       # classical only (minutes)
    !python scripts/nested_selection.py summary

Outputs (paper2 results folder)
-------------------------------
nested_{dataset}.csv        chosen configuration and outer kappa per fold
nested_inner_{dataset}.csv  every inner validation score
nested_tests.csv            circuit against MLP under nested selection
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = sys.argv[:1]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    LabelEncoder,
    MaxAbsScaler,
    StandardScaler,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hybrid import ReuploadingClassifier  # noqa: E402
from scripts.readout_control import paired  # noqa: E402
from scripts.seeds_stats import CACHE, RESULTS, WIDTHS  # noqa: E402

from geoq.datasets import load_dataset  # noqa: E402
from geoq.features.covariance import Covariances  # noqa: E402
from geoq.features.tangent_space import TangentSpace  # noqa: E402

#: Three seeds keep the circuit's cost near thirty hours. The seed sets
#: both the initialisation and which two subjects validate.
SEEDS = (0, 1, 2)

DEPTHS = (5, 7, 9, 11)
HIDDEN = (8, 16, 24, 64)
ALPHAS = (1e-4, 1e-2)
N_VALIDATION = 2


def circuit_candidates(n_qubits: int, seed: int) -> dict:
    """Candidate circuits, keyed by a readable configuration name."""
    return {
        f"L{depth}": ReuploadingClassifier(n_qubits=n_qubits, n_layers=depth, seed=seed)
        for depth in DEPTHS
    }


def mlp_candidates(n_qubits: int, n_features: int, seed: int) -> dict:
    """Candidate MLPs on exactly the inputs the circuit receives."""
    return {
        f"h{hidden}_a{alpha:g}": make_pipeline(
            StandardScaler(),
            PCA(min(n_qubits, n_features), random_state=0),
            MaxAbsScaler(),
            MLPClassifier(
                hidden_layer_sizes=(hidden,),
                alpha=alpha,
                max_iter=800,
                early_stopping=True,
                random_state=seed,
            ),
        )
        for hidden in HIDDEN
        for alpha in ALPHAS
    }


def inner_split(training: np.ndarray, seed: int) -> np.ndarray:
    """Pick the validation subjects for this seed, rotating through them.

    Args:
        training: The eight training subjects of the outer fold, sorted.
        seed: Run seed; consecutive seeds validate on different subjects.

    Returns:
        The subjects held out for inner validation.
    """
    start = (N_VALIDATION * seed) % training.size
    order = np.concatenate([training, training])
    return order[start : start + N_VALIDATION]


def score(model, x_train, y_train, x_test, y_test) -> float:
    """Fit on the training part and return Cohen's kappa on the test part."""
    model.fit(x_train, y_train)
    return float(cohen_kappa_score(y_test, model.predict(x_test)))


def run(dataset: str, families: list[str]) -> None:
    """Nested selection for one dataset, resuming at the fold level."""
    output = RESULTS / f"nested_{dataset}.csv"
    inner_output = RESULTS / f"nested_inner_{dataset}.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    inner_records = (
        pd.read_csv(inner_output).to_dict("records") if inner_output.exists() else []
    )
    done = {(r["family"], r["seed"], r["subject"]) for r in records}

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    labels = LabelEncoder().fit_transform(data.labels)
    subjects = np.asarray(data.subjects)
    n_qubits = WIDTHS[dataset]
    n_features = covariances.shape[1] * (covariances.shape[1] + 1) // 2

    for seed in SEEDS:
        for held_out in np.unique(subjects):
            #  The reference point is fitted on the outer training subjects,
            #  as in the main experiments, and reused by every candidate.
            outer_train = subjects != held_out
            reference = TangentSpace().fit(covariances[outer_train])
            x_outer = reference.transform(covariances[outer_train])
            x_held = reference.transform(covariances[subjects == held_out])
            y_outer = labels[outer_train]
            y_held = labels[subjects == held_out]
            groups = subjects[outer_train]

            training = np.unique(groups)
            validation = inner_split(training, seed)
            inner_is_validation = np.isin(groups, validation)

            for family in families:
                if (family, seed, held_out) in done:
                    print(
                        f"    {family:8s} seed {seed} subject {held_out} cached",
                        flush=True,
                    )
                    continue
                started = time.perf_counter()
                candidates = (
                    circuit_candidates(n_qubits, seed)
                    if family == "circuit"
                    else mlp_candidates(n_qubits, n_features, seed)
                )
                inner = {}
                for name, model in candidates.items():
                    inner[name] = score(
                        model,
                        x_outer[~inner_is_validation],
                        y_outer[~inner_is_validation],
                        x_outer[inner_is_validation],
                        y_outer[inner_is_validation],
                    )
                    inner_records.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "seed": seed,
                            "subject": held_out,
                            "candidate": name,
                            "inner_kappa": inner[name],
                        }
                    )
                chosen = max(inner, key=inner.get)

                #  Refit the chosen configuration on all eight training
                #  subjects before touching the held-out subject.
                final = (
                    circuit_candidates(n_qubits, seed)[chosen]
                    if family == "circuit"
                    else mlp_candidates(n_qubits, n_features, seed)[chosen]
                )
                kappa = score(final, x_outer, y_outer, x_held, y_held)
                records.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "seed": seed,
                        "subject": held_out,
                        "chosen": chosen,
                        "inner_kappa": inner[chosen],
                        "kappa": kappa,
                    }
                )
                pd.DataFrame(records).to_csv(output, index=False)
                pd.DataFrame(inner_records).to_csv(inner_output, index=False)
                print(
                    f"    {family:8s} seed {seed} subject {held_out}: "
                    f"chose {chosen:10s} inner {inner[chosen]:+.3f} "
                    f"outer {kappa:+.3f} "
                    f"({(time.perf_counter() - started) / 60:.1f} min)",
                    flush=True,
                )


def summary() -> None:
    """Report nested accuracy, the paired test and selection stability."""
    rows = []
    for dataset in WIDTHS:
        path = RESULTS / f"nested_{dataset}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        print(f"\n{dataset}")
        print("  mean kappa under nested selection")
        print(frame.groupby("family").kappa.mean().round(4).to_string())
        print("  configurations chosen (count over folds and seeds)")
        print(frame.groupby(["family", "chosen"]).size().rename("folds").to_string())

        means = frame.groupby(["family", "subject"]).kappa.mean()
        if {"circuit", "mlp"} <= set(frame.family.unique()):
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": "circuit - mlp",
                    **paired(means["circuit"].to_numpy(), means["mlp"].to_numpy()),
                }
            )
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS / "nested_tests.csv", index=False)
        print(
            "\nCircuit against MLP, both nested "
            "(paired by held-out record, averaged over seeds)"
        )
        print(table.round(4).to_string(index=False))


def main() -> None:
    """Run the requested family on both datasets, then summarise."""
    families = (
        ["circuit", "mlp"]
        if FAMILY == "all"
        else []
        if FAMILY == "summary"
        else [FAMILY]
    )
    if families and not set(families) <= {"circuit", "mlp"}:
        raise SystemExit("Use: circuit, mlp, all or summary.")
    for dataset in WIDTHS:
        if families:
            print(f"{dataset}  [{', '.join(families)}]", flush=True)
            run(dataset, families)
    summary()


if __name__ == "__main__":
    main()
