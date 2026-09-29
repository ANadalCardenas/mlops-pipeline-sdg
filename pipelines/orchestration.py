"""
Single CLI entrypoint for the full ML pipeline. Imports and calls src/ modules
in the correct order:

1.  Parses CLI arguments (--data-version, --experiment-name, --run-name,
   --output-dir, --register-model, --model-stage)
2. Creates the output directory tree (output_dir/ and output_dir/figures/)
3. Delegates to train.run_training(...) to execute the complete pipeline
4. Optionally registers the model in the MLflow Model Registry and transitions
   it to the requested stage (Staging or Production)

This is the script that CI (pr_validation.yml) calls on every PR.

CLI args:
    --data-version      (default: v1)
    --experiment-name   (default: sdg-experiment)
    --run-name          (default: orchestration-run)
    --output-dir        (default: reports/run)
    --register-model    (flag, default: False)
    --model-stage       (default: None, choices: None | Staging | Production)
"""

import argparse
import os
import sys

import mlflow

# Make sure src/ is importable when the script is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train import run_training
from src.utils.io import load_json, save_json

MODEL_NAME = "sdg-model"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--experiment-name", default="sdg-experiment")
    parser.add_argument("--run-name", default="orchestration-run")
    parser.add_argument("--output-dir", default="reports/run")

    # store_true means the flag is False by default and becomes True when passed
    parser.add_argument("--register-model", action="store_true", default=False)

    # "None" as a string choice lets CI pass it explicitly without triggering a stage transition
    parser.add_argument("--model-stage", default=None, choices=["None", "Staging", "Production"])
    args = parser.parse_args()

    # Create the output directory tree before training so sub-modules can write freely
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    run_id = run_training(
        data_version=args.data_version,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    # Register the model in the MLflow Model Registry if requested, and update model_info.json with registry metadata
    if args.register_model:
        # Register the run's model artifact under the logical name MODEL_NAME;
        # the registry auto-increments the version number on each call
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=MODEL_NAME,
        )

        if args.model_stage and args.model_stage != "None":
            client = mlflow.MlflowClient()
            # Aliases replace the deprecated stages API (removed in MLflow 3.0).
            # An alias is a named pointer to a specific version — setting it on a new version
            # automatically moves the pointer away from the previous one, achieving the same
            # "only one active version per stage" behaviour without the deprecation warning.
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias=args.model_stage,   # e.g. "Staging" or "Production"
                version=result.version,
            )

        # Append registry metadata to the model_info.json already written by run_training()
        model_info_path = os.path.join(args.output_dir, "model_info.json")
        model_info = load_json(model_info_path)
        model_info.update({
            "registry_model_name": MODEL_NAME,
            "registry_model_version": result.version,           # integer assigned by the registry
            "registry_stage": args.model_stage or "None",
            "registry_model_uri": f"models:/{MODEL_NAME}/{result.version}",
        })
        # Save the updated model info back to the same path, overwriting the previous version without registry metadata
        save_json(model_info, model_info_path)
