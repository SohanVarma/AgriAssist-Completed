from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

REPORT_PATH = RESULTS_DIR / "classification_report.txt"

OUTPUT_PRECISION_RECALL = RESULTS_DIR / "precision_recall_by_class.png"
OUTPUT_F1 = RESULTS_DIR / "f1_score_by_class.png"
OUTPUT_SUPPORT = RESULTS_DIR / "class_support_by_class.png"
OUTPUT_COMBINED = RESULTS_DIR / "precision_recall_f1_by_class.png"


def parse_classification_report(report_path: Path):
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found.\n"
            "Run this first:\n"
            "python3 src/07_model_evaluation.py"
        )

    text = report_path.read_text(encoding="utf-8", errors="ignore")

    labels = []
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []

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

        labels.append(label)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        support_values.append(support)

    if not labels:
        raise ValueError("No class metrics found in classification_report.txt")

    return labels, precision_values, recall_values, f1_values, support_values


def save_precision_recall_graph(labels, precision_values, recall_values):
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(18, 8))
    plt.bar(x - width / 2, precision_values, width, label="Precision")
    plt.bar(x + width / 2, recall_values, width, label="Recall")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision and Recall by Class")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PRECISION_RECALL, dpi=220)
    plt.close()

    print(f"Saved: {OUTPUT_PRECISION_RECALL}")


def save_f1_graph(labels, f1_values):
    x = np.arange(len(labels))

    plt.figure(figsize=(18, 8))
    plt.bar(x, f1_values)

    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.title("F1-score by Class")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_F1, dpi=220)
    plt.close()

    print(f"Saved: {OUTPUT_F1}")


def save_support_graph(labels, support_values):
    x = np.arange(len(labels))

    plt.figure(figsize=(18, 8))
    plt.bar(x, support_values)

    plt.xlabel("Class")
    plt.ylabel("Number of Test Samples")
    plt.title("Class Support by Class")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_SUPPORT, dpi=220)
    plt.close()

    print(f"Saved: {OUTPUT_SUPPORT}")


def save_combined_graph(labels, precision_values, recall_values, f1_values):
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(20, 8))
    plt.bar(x - width, precision_values, width, label="Precision")
    plt.bar(x, recall_values, width, label="Recall")
    plt.bar(x + width, f1_values, width, label="F1-score")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-score by Class")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_COMBINED, dpi=220)
    plt.close()

    print(f"Saved: {OUTPUT_COMBINED}")


def main():
    labels, precision_values, recall_values, f1_values, support_values = parse_classification_report(REPORT_PATH)

    print("\nParsed class metrics:")
    for label, precision, recall, f1, support in zip(
        labels, precision_values, recall_values, f1_values, support_values
    ):
        print(
            f"{label}: precision={precision:.2f}, "
            f"recall={recall:.2f}, f1={f1:.2f}, support={support}"
        )

    save_precision_recall_graph(labels, precision_values, recall_values)
    save_f1_graph(labels, f1_values)
    save_support_graph(labels, support_values)
    save_combined_graph(labels, precision_values, recall_values, f1_values)

    print("\nAll graphs generated successfully.")


if __name__ == "__main__":
    main()
