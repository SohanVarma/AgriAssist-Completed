from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

REPORT_PATH = RESULTS_DIR / "classification_report.txt"
OUTPUT_PATH = RESULTS_DIR / "precision_recall_by_class.png"


def parse_classification_report(report_path: Path):
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found.\n"
            "Run evaluation first:\n"
            "python3 src/07_model_evaluation.py"
        )

    text = report_path.read_text(encoding="utf-8")

    labels = []
    precision_values = []
    recall_values = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("precision"):
            continue

        if line.startswith("accuracy"):
            continue

        if line.startswith("macro avg"):
            continue

        if line.startswith("weighted avg"):
            continue

        parts = re.split(r"\s+", line)

        if len(parts) < 5:
            continue

        try:
            support = int(float(parts[-1]))
            f1_score = float(parts[-2])
            recall = float(parts[-3])
            precision = float(parts[-4])
            label = " ".join(parts[:-4])
        except ValueError:
            continue

        if not label:
            continue

        labels.append(label)
        precision_values.append(precision)
        recall_values.append(recall)

    if not labels:
        raise ValueError(
            "Could not find class-level precision and recall values in classification_report.txt"
        )

    return labels, precision_values, recall_values


def plot_precision_recall(labels, precision_values, recall_values):
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(18, 8))

    plt.bar(x - width / 2, precision_values, width, label="Precision")
    plt.bar(x + width / 2, recall_values, width, label="Recall")

    plt.xlabel("Dataset / Class")
    plt.ylabel("Score")
    plt.title("Precision and Recall for Individual Classes")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=220)
    plt.close()

    print(f"Saved precision-recall graph to: {OUTPUT_PATH}")


def main():
    labels, precision_values, recall_values = parse_classification_report(REPORT_PATH)

    print("Classes found:")
    for label, precision, recall in zip(labels, precision_values, recall_values):
        print(f"{label}: precision={precision:.2f}, recall={recall:.2f}")

    plot_precision_recall(labels, precision_values, recall_values)


if __name__ == "__main__":
    main()
