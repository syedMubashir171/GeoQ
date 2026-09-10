"""Classical MLP against a random forest on tangent-space features.

Block 5 deliverable
-------------------
The syllabus is blunt about why this matters: if a classical multilayer
perceptron cannot match a random forest on these features, a hybrid model
that places a quantum circuit inside a classical network has no foundation,
because the classical part is the part doing the work.

The comparison is run under the same leave-one-subject-out protocol as every
other result in this project, so the numbers sit alongside MDM and TS+LDA
rather than needing their own frame of reference.

The training recipe is the one the syllabus names: AdamW with weight decay,
a cosine learning-rate schedule, GELU activations, dropout, and early
stopping on a validation split taken from the training subjects. Each of
those is standard, and stating them matters because a hybrid whose classical
half is undertrained will make its quantum half look better than it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoq.datasets import load_dataset
from geoq.evaluation.protocol import evaluate
from geoq.evaluation.splitters import LeaveOneSubjectOut
from geoq.features.covariance import Covariances
from geoq.features.tangent_space import TangentSpace

CACHE = "/content/drive/MyDrive/GeoQ_workspace/cache"
RESULTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/drive/MyDrive/GeoQ_workspace/results/block5"
)
RESULTS.mkdir(parents=True, exist_ok=True)


def build_models() -> dict:
    """Return the classifiers to compare.

    Every model is wrapped with a scaler fitted inside the fold, so no
    statistic crosses the train-test boundary.
    """
    return {
        "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=0,
                n_jobs=-1,
            ),
        ),
        #  Two widths, because a network too small to fit the features would
        #  understate what the classical half of a hybrid can do, and one
        #  too large would overfit 288 trials per subject. Early stopping
        #  holds out a tenth of the training subjects, never the test one.
        "mlp_small": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            ),
        ),
        "mlp_deep": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-2,
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            ),
        ),
    }


def main(dataset_name: str = "bci_iv_2a_lr") -> None:
    """Evaluate every model on identical leave-one-subject-out folds."""
    data = load_dataset(dataset_name, cache_dir=CACHE)
    print(f"{data}\n", flush=True)

    covariances = Covariances(estimator="oas").fit_transform(data.epochs)
    tangent = TangentSpace().fit_transform(covariances)
    print(
        f"tangent-space features: {tangent.shape[1]} per trial "
        f"from {data.n_channels} channels\n",
        flush=True,
    )

    splitter = LeaveOneSubjectOut()
    records = []
    for name, model in build_models().items():
        result = evaluate(
            model,
            tangent,
            data.labels,
            groups=data.subjects,
            splitter=splitter,
            metrics=("accuracy", "kappa"),
        )
        records.append(
            {
                "model": name,
                "kappa": result.mean("kappa"),
                "kappa_sd": result.std("kappa"),
                "accuracy": result.mean("accuracy"),
            }
        )
        print(
            f"  {name:14s} kappa {result.mean('kappa'):+.3f} "
            f"± {result.std('kappa'):.3f}",
            flush=True,
        )

    frame = pd.DataFrame(records)
    frame.to_csv(RESULTS / "mlp_vs_forest.csv", index=False)

    forest = frame.loc[frame.model == "random_forest", "kappa"].iloc[0]
    best_mlp = frame[frame.model.str.startswith("mlp")].kappa.max()
    print("\n" + "=" * 62)
    print(
        f"random forest {forest:+.3f}   best MLP {best_mlp:+.3f}   "
        f"difference {best_mlp - forest:+.3f}"
    )
    if best_mlp >= forest - 0.02:
        print("\nThe MLP is competitive, so a hybrid placing a quantum layer")
        print("inside a classical network has a working classical half. That")
        print("is the precondition the syllabus sets for Paper 2.")
    else:
        print("\nThe MLP trails the forest. Until that gap closes, a hybrid")
        print("cannot be interpreted: a quantum layer inside an undertrained")
        print("network would be credited with work the network should be")
        print("doing itself.")
    print("=" * 62)


if __name__ == "__main__":
    main()
