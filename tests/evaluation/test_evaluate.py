import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation.evaluate import evaluate_model, generate_evaluation_plots


def _fit_simple_model():
    # Three classes separated along the first feature, like F22 in the real data
    rng = np.random.RandomState(0)
    X = rng.normal(size=(90, 3))
    y = np.digitize(X[:, 0] + rng.normal(scale=0.1, size=90), [-0.5, 0.5])
    y = np.array(["A", "B", "C"])[y]
    model = LogisticRegression().fit(X, y)
    return model, X, y


def test_evaluate_model_returns_expected_metric_keys_and_ranges():
    model, X, y = _fit_simple_model()

    metrics = evaluate_model(model, X, y)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_generate_evaluation_plots_writes_three_png_files(tmp_path):
    model, X, y = _fit_simple_model()

    generate_evaluation_plots(model, X, y, output_dir=str(tmp_path))

    for filename in ("confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
