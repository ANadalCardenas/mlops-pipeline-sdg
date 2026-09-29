import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.features.build_features import EXCLUDED_COLUMNS, TARGET, build_training_pipeline
from src.inference import predict


def _register_version(df):
    X = df.drop(columns=[TARGET, *EXCLUDED_COLUMNS], errors="ignore")
    pipeline = build_training_pipeline(X).fit(X, df[TARGET])

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=infer_signature(X, pipeline.predict(X)))

    return mlflow.register_model(f"runs:/{run.info.run_id}/model", predict.MODEL_NAME).version


def test_run_inference_uses_production_version_and_appends_to_database(tmp_path, monkeypatch, sdg_dataframe):
    # Point MLflow at a local file store so the test needs no remote registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")

    # Two versions exist, but only the first one carries the Production alias
    production_version = _register_version(sdg_dataframe)
    _register_version(sdg_dataframe)
    mlflow.MlflowClient().set_registered_model_alias(predict.MODEL_NAME, predict.MODEL_ALIAS, production_version)

    input_data = sdg_dataframe.drop(columns=[TARGET, *EXCLUDED_COLUMNS], errors="ignore").iloc[0].to_dict()
    # A temporary SQLite file keeps the test free of external services
    db_url = f"sqlite:///{tmp_path / 'predictions.db'}"

    first = predict.run_inference(input_data, db_url)
    second = predict.run_inference(input_data, db_url)

    rows = pd.read_sql("SELECT * FROM predictions", db_url, dtype={"model_version": str})
    assert list(rows.columns) == ["prediction_id", "timestamp", "model_version", "input_data", "prediction"]
    assert len(rows) == 2
    assert list(rows["prediction_id"]) == [first["prediction_id"], second["prediction_id"]]
    assert set(rows["model_version"]) == {str(production_version)}
    assert json.loads(rows["input_data"][0]) == input_data
    assert set(rows["prediction"]) <= {"A", "B", "C", "D"}
