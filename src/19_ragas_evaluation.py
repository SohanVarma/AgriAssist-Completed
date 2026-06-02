from pathlib import Path
import json
import os

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from importlib import import_module


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

QUESTIONS_PATH = DATA_DIR / "ragas_eval_questions.json"
OUTPUT_JSON = RESULTS_DIR / "ragas_evaluation_results.json"
OUTPUT_CSV = RESULTS_DIR / "ragas_evaluation_results.csv"

load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")

rag_module = import_module("09_full_rag_pipeline")

AgriculturalRAG = rag_module.AgriculturalRAG
KB_PATH = rag_module.KB_PATH


def load_eval_questions():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{QUESTIONS_PATH} not found. Create data/ragas_eval_questions.json first."
        )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ragas_dataset():
    rag = AgriculturalRAG(KB_PATH)
    eval_questions = load_eval_questions()

    rows = []

    for item in eval_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        result = rag.generate_advisory(question, top_k=5)

        contexts = []

        for retrieved in result["retrieved_documents"]:
            doc = retrieved["document"]

            context_text = " ".join(
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

            contexts.append(context_text)

        rows.append(
            {
                "question": question,
                "answer": result["generated_advisory"],
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

        print(f"Prepared RAGAS sample for: {question}")

    return Dataset.from_list(rows)


def run_ragas_evaluation():
    dataset = build_ragas_dataset()

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()

    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print("\nRAGAS Evaluation Completed")
    print(result)

    print("\nSaved outputs:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {OUTPUT_JSON}")


if __name__ == "__main__":
    run_ragas_evaluation()
