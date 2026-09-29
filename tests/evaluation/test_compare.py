import json

from src.evaluation import compare


def _write_metrics(path, **overrides):
    metrics = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8, "roc_auc": 0.8, "recall_D": 0.5}
    metrics.update(overrides)
    path.write_text(json.dumps(metrics))
    return metrics


def test_compare_metrics_reports_improved_and_better_verdict(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_metrics(baseline_path, f1=0.70, roc_auc=0.75)
    _write_metrics(candidate_path, f1=0.80, roc_auc=0.85)

    output_dir = tmp_path / "out"
    result = compare.compare_metrics(str(candidate_path), str(baseline_path), str(output_dir))

    assert result["verdict"] == "BETTER"
    assert result["metrics"]["f1"]["status"] == "improved"
    assert (output_dir / "comparison.json").exists()

    comment = (output_dir / "pr_comment.md").read_text()
    assert "BETTER" in comment
    assert "View experiment on DagsHub" not in comment


def test_compare_metrics_reports_degraded_and_worse_verdict_with_dagshub_link(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_metrics(baseline_path, f1=0.80, roc_auc=0.85)
    _write_metrics(candidate_path, f1=0.70, roc_auc=0.75)

    output_dir = tmp_path / "out"
    result = compare.compare_metrics(
        str(candidate_path), str(baseline_path), str(output_dir),
        dagshub_url="https://dagshub.example/exp",
        candidate_version="v1",
    )

    assert result["verdict"] == "WORSE OR EQUAL"
    assert result["metrics"]["f1"]["status"] == "degraded"

    comment = (output_dir / "pr_comment.md").read_text()
    assert "https://dagshub.example/exp" in comment
    assert "Candidate dataset: v1" in comment


def test_compare_metrics_status_unchanged_within_tolerance(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_metrics(baseline_path, accuracy=0.9)
    _write_metrics(candidate_path, accuracy=0.9 + 0.00005)

    result = compare.compare_metrics(str(candidate_path), str(baseline_path), str(tmp_path / "out"))

    assert result["metrics"]["accuracy"]["status"] == "unchanged"


def test_compare_metrics_reports_recall_d_without_affecting_verdict(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_metrics(baseline_path, recall_D=0.6)
    _write_metrics(candidate_path, recall_D=0.0)

    output_dir = tmp_path / "out"
    result = compare.compare_metrics(str(candidate_path), str(baseline_path), str(output_dir))

    assert result["metrics"]["recall_D"]["status"] == "degraded"
    assert result["verdict"] == "BETTER"  # decided by F1 and ROC AUC only
    assert "| Recall D" in (output_dir / "pr_comment.md").read_text()


def test_compare_metrics_skips_metric_missing_from_baseline(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_metrics(candidate_path)
    # A baseline trained with code older than recall_D doesn't have that key
    baseline = _write_metrics(baseline_path)
    del baseline["recall_D"]
    baseline_path.write_text(json.dumps(baseline))

    output_dir = tmp_path / "out"
    result = compare.compare_metrics(str(candidate_path), str(baseline_path), str(output_dir))

    assert "recall_D" not in result["metrics"]
    assert "Recall D" not in (output_dir / "pr_comment.md").read_text()
