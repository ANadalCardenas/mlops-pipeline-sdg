import json

import numpy as np
import pandas as pd

from src.training import train


def _fake_dataset():
    rng = np.random.RandomState(1)
    n = 30

    target = np.array(["A", "B", "C"] * (n // 3))
    rng.shuffle(target)

    return pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "F1": rng.choice(["success", "warning", "error"], size=n),
        "F22": rng.uniform(0, 50, size=n),
        "Target": target,
    })


def test_run_training_produces_metrics_model_and_mlflow_run(tmp_path, monkeypatch):
    # Both the mlflow fallback tracking URI ("file:./mlruns") and the local model save path
    # ("models/<version>/model_latest") are relative to cwd, so run everything inside tmp_path
    # and force a local file-store instead of any real tracking server.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(train, "load_dataset", lambda data_version: _fake_dataset())
    # tmp_path isn't a git repo, so `git rev-parse` would fail after the chdir above
    monkeypatch.setattr(train, "get_git_sha", lambda: "testsha1")

    output_dir = tmp_path / "out"

    run_id = train.run_training(
        data_version="unit-test",
        output_dir=str(output_dir),
        experiment_name="unit-test-experiment",
        run_name="unit-test-run",
    )

    assert isinstance(run_id, str) and run_id

    metrics = json.loads((output_dir / "train_metrics.json").read_text())
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "recall_D"}

    model_info = json.loads((output_dir / "model_info.json").read_text())
    assert model_info["mlflow_run_id"] == run_id
    assert model_info["data_version"] == "unit-test"
    assert model_info["model_uri"] == f"runs:/{run_id}/model"
    assert model_info["git_commit_sha"] == "testsha1"

    for filename in ("confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"):
        assert (output_dir / "figures" / filename).exists()

    assert (tmp_path / "models" / "unit-test" / "model_latest").exists()
