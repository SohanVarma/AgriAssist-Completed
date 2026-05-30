from pathlib import Path
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

KB_PATH = Path("data/advisory_knowledge_base_large.json")

load_dotenv(dotenv_path=Path(".env"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)


def tokenize(text):
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))


def generate_ai_advisory(user_query, region, retrieved_docs):
    if not retrieved_docs:
        return (
            "Agricultural Advisory Report\n\n"
            "No relevant documents were retrieved. Please provide a more specific query including crop name, "
            "problem type, symptoms, location, or season.\n\n"
            "Safety Note:\n"
            "No pesticide, fungicide, fertilizer, irrigation equipment, or other product should be used based only "
            "on AI output. Consult a local agricultural officer or certified agronomist before taking action."
        )

    evidence_text = "\n\n".join(
        [
            f"Document {i + 1}\n"
            f"Retrieval Score: {item['score']:.3f}\n"
            f"TF-IDF Score: {item.get('tfidf_score', 0):.3f}\n"
            f"Keyword Overlap: {item.get('keyword_overlap', 0)}\n"
            f"Crop: {item['document'].get('crop', 'unknown')}\n"
            f"Problem Type: {item['document'].get('problem_type', 'unknown')}\n"
            f"Region/Location: {item['document'].get('region', 'unknown')}\n"
            f"Season: {item['document'].get('season', 'unknown')}\n"
            f"Urgency: {item['document'].get('urgency', 'unknown')}\n"
            f"Recommended Product: {item['document'].get('product_recommended', 'unknown')}\n"
            f"Content: {item['document'].get('content', '')}"
            for i, item in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""
You are AgriAssist, an agricultural decision-support assistant.

User Query:
{user_query}

Region:
{region}

Retrieved Documents:
{evidence_text}

Your task:
Analyze the retrieved documents and generate an in-depth agricultural advisory report.

The report must include these sections:
1. Query Understanding
2. Retrieved Evidence Analysis
3. Crop and Problem Diagnosis
4. Recommended Action Plan
5. Product or Treatment Guidance
6. Limitations
7. Safety Note

Rules:
- Base the report only on retrieved documents.
- Do not invent exact disease names if the documents only say "Disease".
- Do not invent pesticide names, dosage, or chemical instructions.
- For equipment problems, focus on equipment troubleshooting.
- For disease problems, focus on symptom inspection and expert verification.
- Clearly mention uncertainty if the retrieved evidence is generic, noisy, or multilingual.
- Always advise consulting a local agricultural expert before using chemicals or buying equipment.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful agricultural advisory assistant. You provide practical, evidence-based, safety-conscious guidance.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


class AgriculturalRAG:
    def __init__(self, kb_path):
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {kb_path}. Run: python3 src/10_build_large_kb.py"
            )

        with open(kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.texts = [
            " ".join(
                [
                    str(doc.get("crop", "")),
                    str(doc.get("problem_type", "")),
                    str(doc.get("region", "")),
                    str(doc.get("season", "")),
                    str(doc.get("urgency", "")),
                    str(doc.get("product_recommended", "")),
                    str(doc.get("content", "")),
                ]
            )
            for doc in self.documents
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )

        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.doc_vectors)[0]

        query_tokens = tokenize(query)

        ranked = []
        for idx, doc in enumerate(self.documents):
            doc_text = self.texts[idx]
            doc_tokens = tokenize(doc_text)

            keyword_overlap = len(query_tokens.intersection(doc_tokens))

            final_score = float(tfidf_scores[idx]) + (0.05 * keyword_overlap)

            ranked.append(
                (
                    idx,
                    final_score,
                    float(tfidf_scores[idx]),
                    keyword_overlap,
                )
            )

        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        results = []
        for idx, final_score, tfidf_score, overlap in ranked:
            results.append(
                {
                    "score": final_score,
                    "tfidf_score": tfidf_score,
                    "keyword_overlap": overlap,
                    "document": self.documents[idx],
                }
            )

        return results

    def generate_advisory(self, user_query: str, region: str = "India", top_k: int = 5):
        search_query = f"{user_query} {region} farmer advisory"
        retrieved_docs = self.retrieve(search_query, top_k=top_k)

        final_advisory = generate_ai_advisory(
            user_query=user_query,
            region=region,
            retrieved_docs=retrieved_docs,
        )

        return {
            "query": search_query,
            "retrieved_documents": retrieved_docs,
            "generated_advisory": final_advisory,
            "safety_note": "AI-generated agricultural recommendations should be verified with local agricultural experts.",
        }


def main():
    rag = AgriculturalRAG(KB_PATH)

    query = input("\nEnter agricultural query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    result = rag.generate_advisory(query, top_k=5)

    output_lines = [
        "# AgriAssist Full RAG Output",
        "",
        f"Query used for retrieval: {result['query']}",
        "",
        "## Retrieved Documents",
        "",
    ]

    for i, item in enumerate(result["retrieved_documents"], start=1):
        doc = item["document"]

        output_lines.append(f"### Document {i}")
        output_lines.append(f"Final Score: {item['score']:.3f}")
        output_lines.append(f"TF-IDF Score: {item['tfidf_score']:.3f}")
        output_lines.append(f"Keyword Overlap: {item['keyword_overlap']}")
        output_lines.append(f"Crop: {doc.get('crop', 'unknown')}")
        output_lines.append(f"Problem Type: {doc.get('problem_type', 'unknown')}")
        output_lines.append(doc.get("content", ""))
        output_lines.append("")

    output_lines.append("## AI Generated Agricultural Advisory")
    output_lines.append(result["generated_advisory"])
    output_lines.append("")
    output_lines.append("## Safety Note")
    output_lines.append(result["safety_note"])

    final_text = "\n".join(output_lines)

    output_path = RESULTS_DIR / "full_rag_output.md"
    output_path.write_text(final_text, encoding="utf-8")

    print(final_text)
    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
