"""
Inference entrypoint. Loads the current Production model from the MLflow Model
Registry, predicts on a single input record, and inserts the result into a
`predictions` table of a SQL database.

1. Resolves the `Production` alias of `sdg-model` to a concrete registry version
2. Loads exactly that version
3. Generates a prediction (class label) for the JSON input record
4. Inserts prediction_id, timestamp, model_version, input_data, and prediction
   as a new row in the `predictions` table (created on the first insert)

The database comes from --db-url / the DATABASE_URL environment variable, so the
same code works with a local SQLite file or a managed PostgreSQL: only the URL changes.
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone

import mlflow
import pandas as pd
from sqlalchemy import create_engine

MODEL_NAME = "sdg-model"
MODEL_ALIAS = "Production"
DEFAULT_DB_URL = "sqlite:///predictions/predictions.db"


def run_inference(input_data: dict, db_url: str) -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    # Resolve the alias first and load by explicit version, so the alias can't move
    # between loading the model and recording which version produced the prediction
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version}")

    prediction = model.predict(pd.DataFrame([input_data]))[0]

    record = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "input_data": json.dumps(input_data),
        "prediction": str(prediction),  # class label, e.g. "A"
    }

    # Insert the prediction as a new row; if_exists="append" keeps every past prediction
    # and pandas creates the table on the first insert
    engine = create_engine(db_url)
    pd.DataFrame([record]).to_sql("predictions", engine, if_exists="append", index=False)
    engine.dispose()

    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)  # JSON object with one record's features
    # Database connection string; defaults to a local SQLite file
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", DEFAULT_DB_URL))
    args = parser.parse_args()

    # SQLite needs the folder to exist before creating the .db file
    os.makedirs("predictions", exist_ok=True)
    print(json.dumps(run_inference(json.loads(args.input), args.db_url)))
