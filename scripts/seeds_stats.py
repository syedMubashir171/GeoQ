"""Seed variability, exact paired tests, and the per-subject table.

Three things the results so far do not establish
-------------------------------------------------
The circuit was trained once per fold from a single random initialisation.
Variational models are sensitive to where they start, and a result that
moves substantially between seeds is not a result. Every quantum
configuration is therefore retrained from several seeds and reported as a
mean with its spread across seeds, which is a different and more relevant
quantity than the spread across subjects.

The comparisons were tested with a corrected t-test, which assumes the
differences are approximately normal. With nine folds that assumption is
untestable in practice. A paired permutation test assumes only that the
sign of each difference is exchangeable under the null, and with nine folds
the 512 sign assignments can be enumerated, so the p-value is exact rather
than sampled.

Several comparisons are made per dataset, so the Holm step-down correction
is applied within each dataset. It is uniformly more powerful than
Bonferroni and makes no additional assumptions.

Why the classical models are also seeded
-----------------------------------------
An MLP's initialisation matters too. Seeding only the quantum model and
reporting the classical one from a single fixed seed would compare a mean
against a draw, and would flatter whichever happened to start well.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ablations import UnentangledReuploading
from scripts.hybrid import ReuploadingClassifier

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/paper2"
)
RESULTS.mkdir(parents=True, exist_ok=True)

#: Seeds per configuration. Five is the number the review asked for; the
#: quantum runs take about twenty-four hours in total at that setting, and
#: every seed is checkpointed so the sweep can cross sessions.
SEEDS = (0, 1, 2, 3, 4)

WIDTHS = {"bci_iv_2a_lr": 8, "bci_iv_2b": 6}
LAYERS = 9

#: Comparisons tested per dataset, and corrected together within it.
COMPARISONS = (
    ("mlp_24", "hybrid"),
    ("hybrid", "hybrid_no_entangle"),
    ("hybrid", "mlp_8"),
)


def build_mlp(n_hidden: int, n_qubits: int, n_features: int, seed: int):
    """Classical pipeline seeing exactly the features the circuit sees."""
    return make_pipeline(
        StandardScaler(),
        PCA(min(n_qubits, n_features), random_state=0),
        MaxAbsScaler(),
        MLPClassifier(
            hidden_layer_sizes=(n_hidden,),
            max_iter=800,
            early_stopping=True,
            random_state=seed,
        ),
    )


def models(n_qubits: int, n_features: int, seed: int) -> dict:
    """The four configurations, all initialised from the given seed."""
    return {
        "hybrid": ReuploadingClassifier(n_qubits=n_qubits, n_layers=LAYERS, seed=seed),
        "hybrid_no_entangle": UnentangledReuploading(
            n_qubits=n_qubits, n_layers=LAYERS, seed=seed
        ),
        "mlp_24": build_mlp(24, n_qubits, n_features, seed),
        "mlp_8": build_mlp(8, n_qubits, n_features, seed),
    }


def run(dataset: str) -> pd.DataFrame:
    """Evaluate every model from every seed, keeping per-fold scores.

    Each (model, seed) pair is written as it completes, so an interrupted
    session resumes at the pair after the last one finished.
    """
    output = RESULTS / f"seeds_{dataset}.csv"
    records = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {(r["model"], r["seed"]) for r in records}

    data = load_dataset(dataset, cache_dir=CACHE)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)
    n_qubits, n_features = WIDTHS[dataset], tangent.shape[1]
    print(f"  {data}\n    {n_features} features, {n_qubits} qubits", flush=True)

    for seed in SEEDS:
        for name, model in models(n_qubits, n_features, seed).items():
            if (name, seed) in done:
                continue
            started = time.perf_counter()
            result = evaluate(
                model,
                tangent,
                labels,
                groups=data.subjects,
                splitter=LeaveOneSubjectOut(),
                metrics=("kappa",),
            )
            for fold in result.folds:
                records.append(
                    {
                        "dataset": dataset,
                        "model": name,
                        "seed": seed,
                        "subject": fold.test_groups[0],
                        "kappa": fold.scores["kappa"],
                    }
                )
            pd.DataFrame(records).to_csv(output, index=False)
            print(
                f"    seed {seed} {name:20s} kappa "
                f"{result.mean('kappa'):+.3f}  "
                f"({(time.perf_counter() - started) / 60:.1f} min)",
                flush=True,
            )
    return pd.DataFrame(records)


def exact_permutation_p(difference: np.ndarray) -> float:
    """Exact two-sided p-value from sign-flipping the paired differences.

    Under the null the two models are interchangeable, so the sign of each
    paired difference is arbitrary. With nine folds the 512 assignments are
    enumerated, so the result carries no sampling error.

    Args:
        difference: Paired differences, one per fold.

    Returns:
        The exact two-sided p-value.
    """
    n = difference.size
    if n > 20:
        #  Beyond twenty folds enumeration becomes impractical and a large
        #  random sample is used instead.
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(20000, n))
    else:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = (signs * difference).mean(axis=1)
    observed = abs(difference.mean())
    return float((np.abs(null) >= observed - 1e-12).mean())


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, uniformly more powerful than Bonferroni."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compare(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Paired comparisons, averaged over seeds and corrected within dataset."""
    #  Averaging over seeds before pairing asks whether the models differ,
    #  not whether two particular initialisations differ.
    per_subject = (
        frame[frame.dataset == dataset]
        .groupby(["model", "subject"])
        .kappa.mean()
        .unstack(0)
    )

    rows = []
    for first, second in COMPARISONS:
        if first not in per_subject or second not in per_subject:
            continue
        difference = (per_subject[first] - per_subject[second]).to_numpy()
        rng = np.random.default_rng(0)
        draws = rng.choice(
            difference, size=(10000, difference.size), replace=True
        ).mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        spread = difference.std(ddof=1)
        g = (difference.mean() / spread) if spread > 0 else 0.0
        g *= 1.0 - 3.0 / (4.0 * difference.size - 5.0)
        rows.append(
            {
                "dataset": dataset,
                "comparison": f"{first} - {second}",
                "difference": difference.mean(),
                "ci_low": low,
                "ci_high": high,
                "hedges_g": g,
                "p_permutation": exact_permutation_p(difference),
                "n_folds": difference.size,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table.p_permutation.tolist())
    return table


def main() -> None:
    """Run the seed sweep, then the tests and the per-subject table."""
    frames = []
    for dataset in ("bci_iv_2a_lr", "bci_iv_2b"):
        print("=" * 72)
        print(f"{dataset}: {len(SEEDS)} seeds per configuration")
        print("=" * 72)
        frames.append(run(dataset))
    frame = pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 72)
    print("Seed variability: spread of fold-mean kappa across seeds")
    print("=" * 72)
    by_seed = (
        frame.groupby(["dataset", "model", "seed"])
        .kappa.mean()
        .groupby(["dataset", "model"])
        .agg(["mean", "std", "min", "max"])
    )
    print(by_seed.round(4).to_string())
    print("\n  A spread comparable to the difference between models would")
    print("  mean the comparison is not resolvable at this seed count.")

    print("\n" + "=" * 72)
    print("Paired comparisons: exact permutation test, Holm corrected")
    print("=" * 72)
    tables = [compare(frame, d) for d in frame.dataset.unique()]
    tests = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    tests.to_csv(RESULTS / "paired_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("Per-subject table, kappa averaged over seeds")
    print("=" * 72)
    for dataset in frame.dataset.unique():
        table = (
            frame[frame.dataset == dataset]
            .groupby(["subject", "model"])
            .kappa.mean()
            .unstack()
        )
        table.loc["mean"] = table.mean()
        table.to_csv(RESULTS / f"per_subject_{dataset}.csv")
        print(f"\n{dataset}:")
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
