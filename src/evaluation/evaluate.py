"""
Computes multiclass classification metrics and generates evaluation plots for
a fitted sklearn model.

`evaluate_model` returns a dict with keys: accuracy, precision, recall, f1,
roc_auc, recall_D, computed via sklearn.metrics on the held-out test set.
Precision, recall and F1 are macro-averaged (every class weighs the same, so the
minority class counts as much as the others) and ROC AUC is one-vs-rest,
macro-averaged. recall_D is the recall of the minority class D alone, reported
separately because the macro average can hide that the model never predicts it.

`generate_evaluation_plots` saves three PNG files to the specified output
directory:
  - confusion_matrix.png  (ConfusionMatrixDisplay.from_estimator)
  - roc_curve.png         (one-vs-rest RocCurveDisplay, one curve per class)
  - precision_recall_curve.png (one-vs-rest PrecisionRecallDisplay, one curve per class)

Public API:
    evaluate_model(model, X_test, y_test) -> dict
    generate_evaluation_plots(model, X_test, y_test, output_dir: str) -> None
"""

import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Minority class (~5% of rows), whose recall is reported on its own
MINORITY_CLASS = "D"


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # one probability column per class, in model.classes_ order

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro", labels=model.classes_),
        # Share of the real D rows that the model predicted as D
        f"recall_{MINORITY_CLASS}": recall_score(
            y_test, y_pred, labels=[MINORITY_CLASS], average="macro", zero_division=0
        ),
    }


def generate_evaluation_plots(model, X_test, y_test, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix: shows absolute counts per (true class, predicted class)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    y_prob = model.predict_proba(X_test)

    # ROC and precision-recall curves are binary by definition, so draw one
    # "this class vs the rest" curve per class on the same axes
    for display, filename in (
        (RocCurveDisplay, "roc_curve.png"),
        (PrecisionRecallDisplay, "precision_recall_curve.png"),
    ):
        _, ax = plt.subplots()
        for i, cls in enumerate(model.classes_):
            display.from_predictions(y_test == cls, y_prob[:, i], name=f"class {cls}", ax=ax)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
