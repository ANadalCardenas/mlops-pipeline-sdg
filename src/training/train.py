"""
Main training module. Orchestrates a single end-to-end training run:

1. Loads the dataset for the specified data version via load_data.py
2. Splits features and target (drops Target and the excluded columns from X)
3. Performs a stratified train/test split (80/20, random_state=42)
4. Builds and fits the sklearn Pipeline from build_features.py
5. Evaluates the model and saves train_metrics.json via evaluate.py and io.py
6. Generates evaluation PNG plots (confusion matrix, ROC, precision-recall)
7. Logs params, metrics, artifacts, and the model artifact to MLflow
8. Saves the model locally under models/<data_version>/model_latest/
9. Writes model_info.json (run ID, git SHA, model URI, timestamp)

Exposes a callable `run_training(...)` function (used by orchestration.py)
and a CLI entry point via argparse for direct invocation.

CLI args: --data-version, --output-dir, --experiment-name, --run-name
"""

import argparse          # standard library: parses command-line arguments
import os                # standard library: file system operations (makedirs, path joins, getenv)
import shutil            # standard library: used to delete the local model directory before overwriting it
from datetime import datetime, timezone  # standard library: used to stamp the training timestamp in model_info.json

import mlflow            # MLflow client: tracks params, metrics, and artifacts to the remote server
import mlflow.sklearn    # MLflow's sklearn integration: log_model and save_model for sklearn pipelines
from mlflow.models import infer_signature  # infers input/output schema from real data to document the model API
from sklearn.model_selection import train_test_split  # splits data into train and test sets

# Internal modules — each one handles a single responsibility
from src.data.load_data import load_dataset                          # reads the versioned CSV
from src.evaluation.evaluate import evaluate_model, generate_evaluation_plots  # metrics + plots
from src.features.build_features import EXCLUDED_COLUMNS, TARGET, build_training_pipeline  # builds the untrained pipeline
from src.utils.io import save_json                                   # writes a dict to a JSON file
from src.utils.versioning import get_git_sha                         # returns the current git commit SHA


def run_training(data_version: str, output_dir: str, experiment_name: str, run_name: str) -> None:
    # Tell MLflow where to send tracking data; falls back to a local ./mlruns/ folder if the
    # environment variable is not set (useful for running the pipeline without a remote server)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    # Create (or reuse) the named experiment on the tracking server; all runs go under this bucket
    mlflow.set_experiment(experiment_name)

    # Load the raw CSV for the given version and return it as a DataFrame
    df = load_dataset(data_version)

    # Target is the variable to predict, so it belongs in y, not X; the excluded columns
    # (leakage, mostly-missing or no-signal columns) are dropped so the model and its
    # signature never depend on them. errors="ignore" tolerates datasets without some of them
    X = df.drop(columns=[TARGET, *EXCLUDED_COLUMNS], errors="ignore")
    y = df[TARGET]

    # Split into 80% training data and 20% test data;
    # stratify=y ensures both splits have the same class ratio (important: class D is only ~5% of rows);
    # random_state=42 makes the split reproducible across runs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build the untrained sklearn Pipeline (preprocessor + LogisticRegression);
    # we pass X_train so the pipeline can detect column types (numeric vs categorical)
    pipeline = build_training_pipeline(X_train)

    # fit() is a built-in sklearn Pipeline method — it runs fit_transform on every preprocessing
    # step and then fit on the final LogisticRegression estimator, all in one call
    pipeline.fit(X_train, y_train)

    # Compute accuracy, precision, recall, f1, and roc_auc on the held-out test set
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Create the output directory so the files below have somewhere to land
    os.makedirs(output_dir, exist_ok=True)

    # Subdirectory for the three PNG evaluation plots
    figures_dir = os.path.join(output_dir, "figures")

    # Full path for the metrics JSON file
    metrics_path = os.path.join(output_dir, "train_metrics.json")

    # Write the metrics dict to disk so CI can read it back and post it as a PR comment
    save_json(metrics, metrics_path)

    # Save confusion matrix, ROC curve, and precision-recall curve as PNG files
    generate_evaluation_plots(pipeline, X_test, y_test, output_dir=figures_dir)

    # Open a new MLflow run; everything inside the `with` block is associated with this run
    with mlflow.start_run(run_name=run_name) as run:
        # Log the training configuration so we can reproduce this run later
        mlflow.log_params({
            "data_version": data_version,       # which dataset was used (e.g. "v1")
            "model_type": "LogisticRegression", # algorithm name for filtering in the UI
            "test_size": 0.2,                   # fraction of data held out for evaluation
            "random_state": 42,                 # seed used for the train/test split
        })

        # Log all five evaluation metrics; they appear as columns in the MLflow experiment view
        mlflow.log_metrics(metrics)

        # Upload the metrics JSON file as an artifact so it is stored alongside the run
        mlflow.log_artifact(metrics_path)

        # Upload every file in figures_dir (all three PNGs) as artifacts for this run
        mlflow.log_artifacts(figures_dir)

        # infer_signature records the input column names/types and the output shape,
        # so anyone loading this model knows exactly what data it expects
        signature = infer_signature(X_train, pipeline.predict(X_train))

        # Serialize and upload the fitted sklearn pipeline so it can be loaded later with mlflow.sklearn.load_model
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature, input_example=X_train.iloc[:5])

    # Remove any previous local copy before saving — MLflow refuses to overwrite a non-empty directory
    local_model_path = f"models/{data_version}/model_latest"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)

    # Save a local copy of the model so it can be loaded without hitting the remote server
    mlflow.sklearn.save_model(pipeline, path=local_model_path)

    # Collect metadata about this training run into a single JSON file for traceability
    model_info = {
        "data_version": data_version,           # which dataset version was used
        "git_commit_sha": get_git_sha(),        # exact code revision that produced this model
        "model_name": "sdg-model",              # logical name of the model
        "mlflow_run_id": run.info.run_id,       # links back to the MLflow run for full details
        "model_uri": f"runs:/{run.info.run_id}/model",  # URI to load the model from MLflow
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),  # when training finished
    }

    # Write model_info.json next to train_metrics.json in the output directory
    save_json(model_info, os.path.join(output_dir, "model_info.json"))

    # Return the run ID so orchestration.py can use it for model registration
    return run.info.run_id


if __name__ == "__main__":
    # This block only runs when the script is executed directly (not when imported by orchestration.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", required=True)    # e.g. v1 or v2
    parser.add_argument("--output-dir", required=True)      # where to write JSON files and plots
    parser.add_argument("--experiment-name", default="sdg-experiment")  # MLflow experiment bucket
    parser.add_argument("--run-name", default="run")        # label for this specific run in MLflow
    args = parser.parse_args()

    # Delegate to run_training so the logic is reusable when imported from orchestration.py
    run_training(
        data_version=args.data_version,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
