"""Second dataset, and uncertainty on every comparison the paper makes.

What was missing
----------------
The prototype and ablations report point estimates: 0.403 for the circuit,
0.429 for the best classical model, minus 0.003 without entanglement. A
difference of 0.026 between two means whose between-subject standard
deviation is around 0.30 may be nothing at all, and the paper cannot claim
either a loss or a parity without saying which.

Every comparison here is therefore paired by fold. The same subjects are
held out for both models, so the difference is computed per subject and its
uncertainty reflects the variation that matters, rather than the variation
between subjects that both models share.

The corrected test, and why a plain t-test is not enough
---------------------------------------------------------
Leave-one-subject-out folds share training data, so fold scores are
correlated and the usual standard error is too small. The Nadeau and Bengio
correction inflates the variance by one over the fold count plus the ratio
of test to training size, which for nine subjects raises the factor from
0.111 to 0.236. The correction is partial rather than complete, so effect
sizes and intervals are reported alongside and no conclusion rests on a
p-value alone.

Replication on a second dataset
--------------------------------
Dataset 2b provides a different montage and recording protocol, and has one
property that suits this paper in particular. Three electrodes give exactly
six tangent-space features, so at six qubits the circuit receives every
feature and no reduction is applied. The confound this paper is about, a
quantum branch seeing fewer inputs than its classical comparator, is absent
by construction rather than controlled for, which makes the second dataset a
different test rather than a repeat of the first.

PhysioNet MI was attempted first and is not used. Every model evaluated on
it scored at chance, including both classical ones: the circuit reached
-0.002, the classical models 0.034 and 0.000. A dataset on which nothing
works cannot discriminate between methods, so reporting it as a replication
would be misleading in either direction. The cross-dataset analysis in the
companion study found the same weakness, with raw kappa between 0.04 and
0.12 against 0.16 to 0.29 on dataset 2a.

Only the four configurations that carry the paper's claims are repeated, at
the depth established on dataset 2a. Re-tuning on the second dataset would
make it a second experiment rather than a replication, and would allow the
favourable configuration to be selected twice.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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

#: Circuit width per dataset. Dataset 2a has 22 channels and so 253
#: tangent-space features, which must be reduced. Dataset 2b has three
#: channels and six features, so six qubits take all of them and the
#: reduction step is a no-op.
WIDTHS = {"bci_iv_2a_lr": 8, "bci_iv_2b": 6}

#: Depth is held at the value established on dataset 2a, deliberately.
LAYERS = 9


def build_mlp(n_hidden: int, n_qubits: int, n_features: int):
    """Classical pipeline seeing exactly the features the circuit sees.

    Args:
        n_hidden: Width of the hidden layer.
        n_qubits: Circuit width, and the number of components retained.
        n_features: Dimension of the input features.

    Returns:
        The pipeline.
    """
    #  PCA to more components than the data has is an error, and where the
    #  two are equal the step is a rotation that changes nothing. Both are
    #  handled by capping the component count.
    components = min(n_qubits, n_features)
    return make_pipeline(
        StandardScaler(),
        PCA(components, random_state=0),
        MaxAbsScaler(),
        MLPClassifier(
            hidden_layer_sizes=(n_hidden,),
            max_iter=800,
            early_stopping=True,
            random_state=0,
        ),
    )


def models(n_qubits: int, n_features: int) -> dict:
    """The four configurations that carry the paper's claims.

    Args:
        n_qubits: Circuit width for this dataset.
        n_features: Dimension of the tangent-space features.

    Returns:
        Mapping from name to unfitted estimator.
    """
    return {
        "hybrid": ReuploadingClassifier(n_qubits=n_qubits, n_layers=LAYERS),
        "hybrid_no_entangle": UnentangledReuploading(
            n_qubits=n_qubits, n_layers=LAYERS
        ),
        "mlp_24": build_mlp(24, n_qubits, n_features),
        "mlp_8": build_mlp(8, n_qubits, n_features),
    }


def run_dataset(name: str, **kwargs) -> pd.DataFrame:
    """Evaluate every model, keeping the per-fold scores.

    Args:
        name: Dataset identifier.
        **kwargs: Passed to the loader, for the subject subset.

    Returns:
        One row per model and fold.
    """
    output = RESULTS / f"replicate_{name}.csv"
    if output.exists():
        print(f"  {name}: cached", flush=True)
        return pd.read_csv(output)

    data = load_dataset(name, cache_dir=CACHE, **kwargs)
    print(f"  {data}", flush=True)
    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    labels = LabelEncoder().fit_transform(data.labels)

    n_qubits = WIDTHS[name]
    n_features = tangent.shape[1]
    reduced = min(n_qubits, n_features)
    print(
        f"    {n_features} tangent features, {n_qubits} qubits, "
        f"{reduced} components retained"
        f"{' (no reduction)' if reduced == n_features else ''}",
        flush=True,
    )

    rows = []
    for label, model in models(n_qubits, n_features).items():
        started = time.perf_counter()
        result = evaluate(
            model,
            tangent,
            labels,
            groups=data.subjects,
            splitter=LeaveOneSubjectOut(),
            metrics=("kappa",),
        )
        #  Per-fold scores are kept, not just the mean. Every comparison
        #  below is paired by subject, which is only possible with them.
        for fold in result.folds:
            rows.append(
                {
                    "dataset": name,
                    "model": label,
                    "subject": fold.test_groups[0],
                    "kappa": fold.scores["kappa"],
                }
            )
        print(
            f"    {label:20s} kappa {result.mean('kappa'):+.3f} "
            f"± {result.std('kappa'):.3f}  "
            f"({(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def paired_comparison(frame: pd.DataFrame, first: str, second: str) -> dict:
    """Compare two models on the folds they share.

    Args:
        frame: Per-fold scores for one dataset.
        first: Name of the first model.
        second: Name of the second model.

    Returns:
        The mean difference, its interval, effect size and corrected
        p-value.
    """
    wide = frame.pivot(index="subject", columns="model", values="kappa")
    a, b = wide[first].to_numpy(), wide[second].to_numpy()
    difference = a - b
    n = difference.size

    #  Nadeau and Bengio: leave-one-subject-out folds share training data,
    #  so the naive variance understates the uncertainty.
    correction = 1.0 / n + (1.0 / n) / (1.0 - 1.0 / n)
    spread = difference.std(ddof=1)
    error = np.sqrt(correction) * spread if spread > 0 else np.inf
    t_statistic = difference.mean() / error if error else 0.0
    p_value = 2.0 * stats.t.sf(abs(t_statistic), df=n - 1)

    rng = np.random.default_rng(0)
    draws = rng.choice(difference, size=(10000, n), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])

    #  Hedges' g, with the small-sample correction that matters at nine
    #  folds, where it reduces the estimate by roughly nine per cent.
    g = difference.mean() / spread if spread > 0 else 0.0
    g *= 1.0 - 3.0 / (4.0 * n - 5.0)

    return {
        "comparison": f"{first} - {second}",
        "difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "hedges_g": float(g),
        "p_corrected": float(p_value),
        "n_folds": n,
    }


def main() -> None:
    """Replicate on a second dataset and quantify every comparison."""
    print("=" * 72)
    print("Dataset 2a")
    print("=" * 72)
    first = run_dataset("bci_iv_2a_lr")

    print("\n" + "=" * 72)
    print("Dataset 2b: three electrodes, six features, no reduction applied")
    print("=" * 72)
    second = run_dataset("bci_iv_2b")

    comparisons = [
        ("mlp_24", "hybrid"),
        ("hybrid", "hybrid_no_entangle"),
        ("hybrid", "mlp_8"),
    ]
    rows = []
    for name, frame in (("bci_iv_2a_lr", first), ("bci_iv_2b", second)):
        for a, b in comparisons:
            if a in set(frame.model) and b in set(frame.model):
                rows.append({"dataset": name, **paired_comparison(frame, a, b)})

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "paired_comparisons.csv", index=False)

    print("\n" + "=" * 72)
    print("Paired comparisons, corrected for the dependence between folds")
    print("=" * 72)
    print(table.round(4).to_string(index=False))

    print("\n  An interval containing zero means the two models are not")
    print("  distinguishable on this data, which is a different statement")
    print("  from their means being equal.")

    print("\n  means by dataset and model:")
    both = pd.concat([first, second])
    print(
        both.groupby(["dataset", "model"])
        .kappa.agg(["mean", "std", "count"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
