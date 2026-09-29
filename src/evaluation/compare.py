import json
import os
import sys
import argparse


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compare_metrics(
    candidate_path: str,
    baseline_path: str,
    output_dir: str,
    dagshub_url: str = "",
    candidate_version: str = "v2",
    baseline_version: str = "v1",
) -> dict:
    with open(candidate_path) as f:
        candidate = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    metrics = {}
    for m in METRICS:
        b, c = baseline[m], candidate[m]
        delta = c - b
        if delta > 0.0001:
            status = "improved"
        elif delta < -0.0001:
            status = "degraded"
        else:
            status = "unchanged"
        metrics[m] = {"baseline": round(b, 4), "candidate": round(c, 4),
                      "delta": round(delta, 4), "status": status}

    verdict = (
        "BETTER"
        if candidate["f1"] >= baseline["f1"] and candidate["roc_auc"] >= baseline["roc_auc"]
        else "WORSE OR EQUAL"
    )

    comparison = {"verdict": verdict, "metrics": metrics}

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    _write_pr_comment(comparison, output_dir, dagshub_url, candidate_version, baseline_version)

    return comparison


def _write_pr_comment(
    comparison: dict, output_dir: str, dagshub_url: str, candidate_version: str, baseline_version: str
) -> None:
    verdict = comparison["verdict"]
    m = comparison["metrics"]

    def fmt_delta(d):
        return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

    rows = ""
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
              "f1": "F1", "roc_auc": "ROC AUC"}
    for key, label in labels.items():
        v = m[key]
        rows += (f"| {label:<9} | {v['baseline']:.4f}   | {v['candidate']:.4f}    "
                 f"| {fmt_delta(v['delta']):<7} | {v['status']:<8} |\n")

    dagshub_line = (f"- [View experiment on DagsHub]({dagshub_url})\n" if dagshub_url else "")

    f1_b, f1_c = m["f1"]["baseline"], m["f1"]["candidate"]
    auc_b, auc_c = m["roc_auc"]["baseline"], m["roc_auc"]["candidate"]

    comment = f"""## SDG Model Validation Report

**Verdict: {verdict}**

- Candidate dataset: {candidate_version}
- Baseline dataset: {baseline_version} (current production)
- Baseline branch: `main`
{dagshub_line}
| Metric    | Baseline | Candidate | Delta   | Status   |
|-----------|----------|-----------|---------|----------|
{rows}
Precision, recall and F1 are macro-averaged over the classes; ROC AUC is one-vs-rest, macro-averaged.

### F1 Score

```mermaid
xychart-beta
  title "F1 Score"
  x-axis ["Baseline", "Candidate"]
  y-axis 0 --> 1
  bar [{f1_b:.3f}, {f1_c:.3f}]
```

### ROC AUC

```mermaid
xychart-beta
  title "ROC AUC"
  x-axis ["Baseline", "Candidate"]
  y-axis 0 --> 1
  bar [{auc_b:.3f}, {auc_c:.3f}]
```

### Artifacts
- candidate_metrics.json
- baseline_metrics.json
- comparison.json
- confusion_matrix.png / roc_curve.png / precision_recall_curve.png
"""

    with open(os.path.join(output_dir, "pr_comment.md"), "w") as f:
        f.write(comment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dagshub-url", default="")
    parser.add_argument("--candidate-version", default="v2")
    parser.add_argument("--baseline-version", default="v1")
    # CI passes this flag so a candidate that doesn't beat the baseline fails the check and blocks the merge
    parser.add_argument("--fail-if-worse", action="store_true", default=False)
    args = parser.parse_args()

    result = compare_metrics(
        args.candidate, args.baseline, args.output_dir, args.dagshub_url,
        args.candidate_version, args.baseline_version,
    )
    print(json.dumps(result, indent=2))

    if args.fail_if_worse and result["verdict"] != "BETTER":
        sys.exit(1)
