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
            "# Agricultural Advisory Report\n\n"
            "## 1. Query Understanding\n"
            "No relevant documents were retrieved for the farmer's query. This means the system does not have enough evidence to generate a reliable advisory.\n\n"
            "## 2. Recommendation\n"
            "Please provide a more specific query including crop name, visible symptoms, problem type, location, season, or uploaded image evidence.\n\n"
            "## 3. Safety Note\n"
            "Do not apply pesticides, fungicides, fertilizers, irrigation equipment, or any other agricultural product based only on AI output. Consult a local agricultural officer or certified agronomist before taking action."
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
You are AgriAssist, an expert-level agricultural decision-support assistant.

Your job is to analyze retrieved agricultural evidence and generate a very detailed, structured, farmer-friendly advisory report.

User Query:
{user_query}

Region:
{region}

Retrieved Evidence:
{evidence_text}

Generate a LONG, IN-DEPTH agricultural advisory report.

The report must be detailed enough for an academic project demonstration. It should not be short. It should be structured, explanatory, and evidence-grounded.

Use the following exact section headings:

# Agricultural Advisory Report

## 1. Query Understanding
Explain what the user/farmer is asking.
Identify the likely crop, problem type, and intent.
Mention whether the query came from direct text or model prediction if inferable.
Explain why this query matters agriculturally.

## 2. Retrieved Evidence Summary
Summarize the retrieved documents in detail.
Mention the crops, problem types, regions, seasons, urgency levels, products, and repeated patterns found in the retrieved evidence.
Explain whether the retrieved documents are consistent or conflicting.
Mention if some retrieved records are noisy, multilingual, generic, or incomplete.

## 3. Evidence-Based Crop and Problem Diagnosis
Identify the most likely crop involved.
Identify the most likely problem category, such as disease, pest, irrigation, equipment, nutrient issue, or general advisory.
If retrieved documents only say "Disease", do not invent a specific disease name.
If the query mentions a specific disease but retrieved documents are generic, clearly explain that the evidence supports only a general disease advisory.
Explain what symptoms or field observations the farmer should verify.

## 4. Severity and Urgency Assessment
Use the urgency levels, crop type, and problem type from the retrieved documents.
Classify severity as Low, Medium, or High.
Explain why.
Mention what could happen if the issue is ignored.
Mention what information is still needed to judge severity more accurately.

## 5. Detailed Recommended Action Plan
Give a step-by-step action plan.
Separate actions into:
- Immediate actions
- Field inspection actions
- Preventive actions
- Follow-up actions

For disease-related issues, include:
- inspect leaves/stems/fruits
- check spread pattern
- isolate/remove heavily affected material only when appropriate
- improve spacing and airflow
- avoid excess leaf wetness
- maintain field sanitation
- consult local experts before chemical use

For pest-related issues, include:
- inspect for insects, eggs, larvae, leaf damage, sticky residue
- use traps or monitoring where relevant
- follow integrated pest management
- avoid unnecessary broad pesticide use

For equipment-related issues, include:
- inspect pumps, sprayers, pipes, nozzles, drip lines
- check blockage, leakage, pressure, calibration, installation
- repair before buying new equipment
- verify equipment suitability with technician or agricultural officer

For irrigation-related issues, include:
- check soil moisture
- check pump function
- check pipe/drip blockages
- avoid overwatering
- ensure drainage

## 6. Product or Treatment Guidance
Only mention products that appear in the retrieved documents.
Do not invent pesticides, fungicides, fertilizers, dosage, timing, or chemical names.
If a product is unclear, missing, or says nan/unknown, say that no reliable product recommendation was found.
Explain that any product must be verified locally before use.
Mention that dosage depends on crop stage, pest/disease confirmation, weather, soil, and local agricultural guidelines.

## 7. Region and Season Considerations
Discuss how region and season may affect the recommendation.
If retrieved documents mention different states or seasons, explain that recommendations should be localized.
Mention weather, humidity, rainfall, irrigation, and seasonal disease/pest pressure when relevant.
Do not invent exact regional rules.

## 8. Practical Farmer Checklist
Provide a checklist the farmer can follow in the field.
Use clear bullet points.
Include observation, diagnosis confirmation, advisory verification, product safety, and follow-up monitoring.

## 9. Limitations of the Retrieved Evidence
Clearly explain limitations.
Mention if records are short, multilingual, generic, noisy, or not disease-specific.
Mention if image prediction confidence should be considered carefully.
Mention that AI advisory depends on retrieved evidence quality.
Mention missing information such as exact crop variety, field age, soil condition, weather, symptom images, and local pest pressure.

## 10. Final Advisory
Give a clear final recommendation paragraph.
It should summarize what the farmer should do next.
It should be useful, safe, and not overconfident.

## 11. Safety Note
Give a strong safety note.
Warn that AI cannot replace certified agricultural experts.
Warn not to apply chemicals, pesticides, fungicides, fertilizers, or buy equipment solely based on AI.
Advise consulting local agricultural officers, agronomists, or extension workers.
Mention safe handling, protective equipment, correct dosage verification, and environmental precautions.

Rules:
- Make the answer detailed and long.
- Use the retrieved evidence as the basis.
- Do not hallucinate specific diseases if not supported.
- Do not invent pesticide names or dosage.
- Do not give unsafe chemical instructions.
- For equipment problems, do not give only disease advice.
- For disease problems, include inspection and disease-management guidance.
- Keep the language professional but easy to understand.
- Write in a way suitable for an academic AI project demo.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful agricultural advisory assistant. "
                    "You generate long, structured, evidence-grounded, safety-conscious agricultural reports."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.25,
        max_tokens=3500,
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
                    str(doc.get("problem_name", "")),
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
        output_lines.append(f"Region: {doc.get('region', 'unknown')}")
        output_lines.append(f"Season: {doc.get('season', 'unknown')}")
        output_lines.append(f"Urgency: {doc.get('urgency', 'unknown')}")
        output_lines.append(f"Recommended Product: {doc.get('product_recommended', 'unknown')}")
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
