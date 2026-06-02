from pathlib import Path
import json
import re
import csv
from importlib import import_module

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

QUESTIONS_PATH = DATA_DIR / "ragas_eval_questions.json"

OUTPUT_JSON = RESULTS_DIR / "custom_rag_evaluation_results.json"
OUTPUT_CSV = RESULTS_DIR / "custom_rag_evaluation_results.csv"
OUTPUT_GRAPH = RESULTS_DIR / "custom_rag_evaluation_metrics.png"


rag_module = import_module("09_full_rag_pipeline")
AgriculturalRAG = rag_module.AgriculturalRAG
KB_PATH = rag_module.KB_PATH


STOPWORDS = {
    "the", "is", "are", "a", "an", "and", "or", "of", "to", "in", "for", "on",
    "with", "by", "from", "as", "at", "it", "this", "that", "be", "before",
    "after", "should", "could", "would", "can", "may", "must", "into", "about",
    "using", "use", "used", "also", "not", "if", "then", "than", "their", "they",
    "them", "his", "her", "your", "you", "we", "our", "he", "she", "i"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def safe_divide(num, den):
    return num / den if den else 0.0


def load_eval_questions():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{QUESTIONS_PATH} not found. Create it first."
        )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def document_to_text(doc):
    return " ".join(
        [
            f"Crop: {doc.get('crop', 'unknown')}.",
            f"Problem Type: {doc.get('problem_type', 'unknown')}.",
            f"Problem Name: {doc.get('problem_name', 'unknown')}.",
            f"Region: {doc.get('region', 'unknown')}.",
            f"Season: {doc.get('season', 'unknown')}.",
            f"Urgency: {doc.get('urgency', 'unknown')}.",
            f"Product: {doc.get('product_recommended', 'unknown')}.",
            f"Content: {doc.get('content', '')}",
        ]
    )


def compute_metrics(question, answer, contexts, ground_truth):
    question_tokens = tokenize(question)
    answer_tokens = tokenize(answer)
    ground_truth_tokens = tokenize(ground_truth)

    context_text = " ".join(contexts)
    context_tokens = tokenize(context_text)

    individual_context_tokens = [tokenize(ctx) for ctx in contexts]

    answer_relevancy = safe_divide(
        len(answer_tokens.intersection(question_tokens.union(ground_truth_tokens))),
        len(question_tokens.union(ground_truth_tokens)),
    )

    context_relevancy = safe_divide(
        len(context_tokens.intersection(question_tokens)),
        len(question_tokens),
    )

    relevant_contexts = 0
    for ctx_tokens in individual_context_tokens:
        overlap = len(ctx_tokens.intersection(question_tokens))
        if overlap > 0:
            relevant_contexts += 1

    context_precision = safe_divide(relevant_contexts, len(individual_context_tokens))

    faithfulness_approx = safe_divide(
        len(answer_tokens.intersection(context_tokens)),
        len(answer_tokens),
    )

    ground_truth_overlap = safe_divide(
        len(answer_tokens.intersection(ground_truth_tokens)),
        len(ground_truth_tokens),
    )

    overall_score = (
        answer_relevancy
        + context_relevancy
        + context_precision
        + faithfulness_approx
        + ground_truth_overlap
    ) / 5

    return {
        "answer_relevancy": round(answer_relevancy, 4),
        "context_relevancy": round(context_relevancy, 4),
        "context_precision": round(context_precision, 4),
        "faithfulness_approx": round(faithfulness_approx, 4),
        "ground_truth_overlap": round(ground_truth_overlap, 4),
        "overall_score": round(overall_score, 4),
    }


def run_evaluation():
    rag = AgriculturalRAG(KB_PATH)
    eval_questions = load_eval_questions()

    rows = []

    for idx, item in enumerate(eval_questions, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\nEvaluating {idx}: {question}")

        result = rag.generate_advisory(question, top_k=5)

        retrieved_docs = result["retrieved_documents"]
        contexts = [document_to_text(doc_item["document"]) for doc_item in retrieved_docs]

        answer = result["generated_advisory"]

        metrics = compute_metrics(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        row = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            **metrics,
        }

        rows.append(row)

        print(metrics)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_columns = [
        "question",
        "answer_relevancy",
        "context_relevancy",
        "context_precision",
        "faithfulness_approx",
        "ground_truth_overlap",
        "overall_score",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row[col] for col in csv_columns})

    print("\nSaved:")
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_CSV}")

    return rows


def plot_results(rows):
    metric_names = [
        "answer_relevancy",
        "context_relevancy",
        "context_precision",
        "faithfulness_approx",
        "ground_truth_overlap",
        "overall_score",
    ]

    averages = {}

    for metric in metric_names:
        averages[metric] = sum(row[metric] for row in rows) / len(rows)

    plt.figure(figsize=(12, 6))
    plt.bar(list(averages.keys()), list(averages.values()))

    plt.xlabel("RAG Evaluation Metric")
    plt.ylabel("Average Score")
    plt.title("Custom RAG Evaluation Metrics for AgriAssist")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(OUTPUT_GRAPH, dpi=220)
    plt.close()

    print(f"- {OUTPUT_GRAPH}")

    print("\nAverage metrics:")
    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")


def main():
    rows = run_evaluation()
    plot_results(rows)


if __name__ == "__main__":
    main()
