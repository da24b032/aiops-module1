"""
AIOps Module 1 - Question 2: MLflow Experiment Comparison
MLP on MNIST, sweeping hidden_layer_size x learning_rate across 6 runs.
"""

import mlflow
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist-mlp")

print("Fetching MNIST (this can take a minute the first time)...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X, y = mnist.data, mnist.target.astype(int)

rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=10000, replace=False)
X, y = X[idx], y[idx]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

classes = np.unique(y_train)

N_EPOCHS = 20

hidden_layer_options = [(50,), (100,), (100, 50)]
learning_rate_options = [0.001, 0.01]

run_id_summary = []

for hidden_layers in hidden_layer_options:
    for lr in learning_rate_options:

        run_name = f"mlp-h{hidden_layers}-lr{lr}"

        with mlflow.start_run(run_name=run_name) as run:

            mlflow.log_params({
                "model_type": "MLPClassifier",
                "hidden_layer_sizes": str(hidden_layers),
                "learning_rate_init": lr,
                "activation": "relu",
                "solver": "adam",
                "n_epochs": N_EPOCHS,
                "dataset": "MNIST",
                "n_train_samples": X_train.shape[0],
            })

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=lr,
                activation="relu",
                solver="adam",
                max_iter=1,
                warm_start=True,
                random_state=42,
            )

            for epoch in range(N_EPOCHS):

                model.partial_fit(
                    X_train,
                    y_train,
                    classes=classes
                )

                train_loss = model.loss_
                val_acc = model.score(X_val, y_val)

                mlflow.log_metric(
                    "train_loss",
                    train_loss,
                    step=epoch
                )

                mlflow.log_metric(
                    "val_accuracy",
                    val_acc,
                    step=epoch
                )

            test_acc = model.score(X_test, y_test)

            mlflow.log_metrics({
                "final_train_loss": model.loss_,
                "final_val_accuracy": val_acc,
                "final_test_accuracy": test_acc,
            })

            mlflow.set_tag("team", "aiops-module1")

            print(
                f"{run_name}: "
                f"val_acc={val_acc:.4f} "
                f"test_acc={test_acc:.4f}"
            )

            run_id_summary.append(
                (run_name, run.info.run_id, val_acc, test_acc)
            )

print("\n--- Run summary ---")

for name, rid, val_acc, test_acc in run_id_summary:
    print(
        f"{name:30s} "
        f"run_id={rid} "
        f"val_acc={val_acc:.4f} "
        f"test_acc={test_acc:.4f}"
    )

runs_df = mlflow.search_runs(
    experiment_names=["mnist-mlp"]
)

best = runs_df.sort_values(
    "metrics.final_val_accuracy",
    ascending=False
).iloc[0]

print(
    f"\nBest run: {best['tags.mlflow.runName']} "
    f"(val_acc={best['metrics.final_val_accuracy']:.4f})"
)
