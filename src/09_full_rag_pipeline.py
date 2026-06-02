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


def normalize_product(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in ["", "nan", "none", "null", "unknown"]:
        return ""

    return value


def generate_ai_advisory(user_query, region, retrieved_docs):
    if not retrieved_docs:
        return (
            "# Agricultural Advisory Report\n\n"
            "## 1. Query Understanding\n"
            "No relevant documents were retrieved for this query. The system does not have enough evidence to provide a confident crop-specific advisory.\n\n"
            "## 2. General Guidance\n"
            "Please provide more details such as crop name, visible symptoms, field condition, region, season, irrigation pattern, pest presence, and uploaded image evidence.\n\n"
            "## 3. Safety Note\n"
            "Consult a local agricultural officer or certified agronomist before applying any treatment."
        )

    evidence_text = "\n\n".join(
        [
            f"Document {i + 1}\n"
            f"Retrieval Score: {item['score']:.3f}\n"
            f"TF-IDF Score: {item.get('tfidf_score', 0):.3f}\n"
            f"Keyword Overlap: {item.get('keyword_overlap', 0)}\n"
            f"Crop: {item['document'].get('crop', 'unknown')}\n"
            f"Problem Type: {item['document'].get('problem_type', 'unknown')}\n"
            f"Problem Name: {item['document'].get('problem_name', 'unknown')}\n"
            f"Region/Location: {item['document'].get('region', 'unknown')}\n"
            f"Season: {item['document'].get('season', 'unknown')}\n"
            f"Urgency: {item['document'].get('urgency', 'unknown')}\n"
            f"Recommended Product: {normalize_product(item['document'].get('product_recommended', '')) or 'No clear product found'}\n"
            f"Content: {item['document'].get('content', '')}"
            for i, item in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""
You are AgriAssist, an agricultural decision-support assistant.

Generate a LONG and DETAILED agricultural advisory report of approximately 1 to 2 pages.

User Query:
{user_query}

Region:
{region}

Retrieved Evidence:
{evidence_text}

Important instruction:
Use the retrieved documents as the primary evidence. You may add general agricultural best-practice guidance when the retrieved evidence is limited, but do not present unsupported details as confirmed facts.

Use these exact section headings:

# Agricultural Advisory Report

## 1. Query Understanding
Explain what the farmer is asking.
Identify the likely crop, problem type, and intent.
Mention whether the issue appears to be disease-related, pest-related, irrigation-related, equipment-related, nutrient-related, or general advisory-related.
Explain why this issue matters for crop health, productivity, yield, and farmer decision-making.

## 2. Retrieved Evidence Summary
Summarize the retrieved documents in detail.
Mention crop, problem type, problem name if available, region, season, urgency, recommended product if available, and repeated patterns.
Explain whether the documents are consistent or conflicting.
Mention if the retrieved evidence is generic, noisy, multilingual, incomplete, or not specific enough.
Mention whether the retrieval scores look strong, moderate, or weak based on the evidence provided.

## 3. Evidence-Based Diagnosis
Identify the most likely crop and problem category based on retrieved documents.
If the documents only say “Disease,” do not claim a specific disease with certainty.
If the user query mentions a specific disease but the retrieved evidence is generic, say that the evidence supports only a general advisory for that disease/problem type.
Mention what symptoms or field signs the farmer should verify before taking action.
For image-based queries, explain that the model prediction and visual observation should be verified with field inspection.

## 4. Severity and Urgency Assessment
Classify severity as Low, Medium, or High based on available evidence.
Explain the reason for the severity level.
Mention what may happen if the issue is ignored.
Mention what extra information is needed to judge severity more accurately, such as crop age, field spread, weather, irrigation history, symptom duration, and pest visibility.

## 5. Detailed Action Plan
Provide a practical step-by-step plan.

Include:
- Immediate actions
- Field inspection actions
- Preventive actions
- Follow-up monitoring actions

For disease-related issues, include:
- Inspect leaves, stems, fruits, and roots if relevant
- Check whether symptoms are spreading
- Compare symptoms across multiple plants instead of relying on one plant
- Remove heavily affected material only when appropriate
- Improve field sanitation
- Improve spacing and airflow
- Avoid excessive leaf wetness
- Avoid unnecessary repeated spraying
- Consult local agricultural experts before treatment decisions

For pest-related issues, include:
- Inspect for insects, eggs, larvae, holes, sticky residue, webbing, and leaf damage
- Use monitoring traps if relevant
- Follow Integrated Pest Management
- Avoid unnecessary broad pesticide use
- Monitor pest population before and after intervention

For equipment-related issues, include:
- Inspect pumps, sprayers, pipes, nozzles, drip lines, filters, valves, and pressure
- Check blockage, leakage, incorrect installation, poor calibration, broken parts, and uneven flow
- Repair before buying new equipment
- Verify equipment suitability with a technician or agricultural officer

For irrigation-related issues, include:
- Check soil moisture
- Check drainage
- Check pump and pipe function
- Check drip emitter clogging and uneven distribution
- Avoid both overwatering and water stress
- Monitor crop response after irrigation changes

For nutrient-related issues, include:
- Observe leaf color patterns and plant growth
- Consider soil testing
- Avoid random fertilizer application
- Verify fertilizer choice locally

## 6. Product or Treatment Guidance
Mention only products or treatment categories that appear in the retrieved documents.
If the retrieved documents show missing, unclear, nan, or unknown product data, say that no reliable product recommendation was found.
Explain what the mentioned product category is generally used for if it appears in the retrieved documents.
Do not provide exact dosage, spray interval, mixing ratio, or guaranteed application method.
Explain that product suitability depends on crop stage, weather, pest or disease confirmation, soil condition, irrigation, local regulations, and label instructions.
Recommend expert verification before using any agricultural input.
If a product appears unrelated to the problem type, mention that it should be verified before use.

## 7. General Agricultural Best Practices
Provide general safe best-practice guidance relevant to the problem type.
Include crop monitoring, sanitation, irrigation control, soil health, balanced nutrition, resistant varieties where relevant, crop rotation where relevant, weed management, equipment maintenance, and record keeping.
Make this section practical and farmer-friendly.

## 8. Precautions
Give a detailed precautions section.

Include:
- Do not apply any agricultural input without expert verification
- Do not mix products without guidance
- Wear protective equipment if any treatment is applied under expert advice
- Keep children and animals away from treated areas
- Avoid spraying during strong wind, rain, or extreme heat
- Do not exceed label recommendations
- Prevent contamination of wells, ponds, streams, and irrigation channels
- Store agricultural inputs safely
- Wash hands and equipment after field operations
- Keep records of symptoms, dates, products used, and crop response
- Follow local agricultural department guidance

## 9. Farmer Field Checklist
Create a clear checklist the farmer can follow.
Use bullet points.
Include observation, photo collection, symptom tracking, affected-area marking, expert consultation, product verification, equipment inspection if relevant, treatment monitoring, and follow-up.

## 10. Limitations of This Advisory
Explain limitations clearly.
Mention that the system depends on retrieved documents and uploaded image/model prediction if used.
Mention that exact diagnosis requires local field inspection.
Mention missing information such as crop variety, plant age, soil condition, exact symptoms, weather, irrigation history, pest visibility, disease spread pattern, and previous treatment history.
Mention that the retrieved farmer records may be noisy or multilingual.

## 11. Final Advisory
Give a detailed final recommendation paragraph.
It should summarize what the farmer should do next.
It should be useful and practical but not overconfident.
It should tell the farmer how to move from observation to expert-verified action.

## 12. Safety Note
Explain that farmers should consult certified agricultural officers, agronomists, or local extension workers before applying pesticides, fungicides, fertilizers, or purchasing equipment.
Mention safe handling, protective equipment, correct local verification, label checking, and environmental precautions.

Rules:
- Make the report long, detailed, and suitable for a 1 to 2 page academic project demo.
- Use retrieved evidence first.
- General best-practice guidance is allowed.
- Do not guarantee diagnosis.
- Do not provide exact pesticide dosage, spray interval, or mixing ratio.
- Do not invent a specific chemical prescription.
- Include detailed precautions.
- Keep the tone professional and farmer-friendly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate long, structured, evidence-grounded agricultural advisory reports. "
                    "You may include general best-practice guidance, but you must avoid exact pesticide dosage, unsafe chemical instructions, or guaranteed diagnosis."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.35,
        max_tokens=4500,
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

            crop = str(doc.get("crop", "")).lower()
            problem_type = str(doc.get("problem_type", "")).lower()
            problem_name = str(doc.get("problem_name", "")).lower()
            product = str(doc.get("product_recommended", "")).lower()
            season = str(doc.get("season", "")).lower()
            region = str(doc.get("region", "")).lower()

            query_lower = query.lower()

            metadata_boost = 0.0

            if crop and crop != "general" and crop in query_lower:
                metadata_boost += 0.15

            if problem_type and problem_type != "general" and problem_type in query_lower:
                metadata_boost += 0.15

            if problem_name and problem_name != "unknown" and problem_name in query_lower:
                metadata_boost += 0.20

            if product and product not in ["nan", "none", "unknown", ""] and product in query_lower:
                metadata_boost += 0.05

            if season and season != "unknown" and season in query_lower:
                metadata_boost += 0.04

            if region and region != "india" and region in query_lower:
                metadata_boost += 0.04

            final_score = (
                float(tfidf_scores[idx])
                + (0.05 * keyword_overlap)
                + metadata_boost
            )

            ranked.append(
                (
                    idx,
                    final_score,
                    float(tfidf_scores[idx]),
                    keyword_overlap,
                    metadata_boost,
                )
            )

        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        results = []

        for idx, final_score, tfidf_score, overlap, metadata_boost in ranked:
            results.append(
                {
                    "score": final_score,
                    "tfidf_score": tfidf_score,
                    "keyword_overlap": overlap,
                    "metadata_boost": metadata_boost,
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
        output_lines.append(f"Metadata Boost: {item.get('metadata_boost', 0):.3f}")
        output_lines.append(f"Crop: {doc.get('crop', 'unknown')}")
        output_lines.append(f"Problem Type: {doc.get('problem_type', 'unknown')}")
        output_lines.append(f"Problem Name: {doc.get('problem_name', 'unknown')}")
        output_lines.append(f"Region: {doc.get('region', 'unknown')}")
        output_lines.append(f"Season: {doc.get('season', 'unknown')}")
        output_lines.append(f"Urgency: {doc.get('urgency', 'unknown')}")
        output_lines.append(
            f"Recommended Product: {normalize_product(doc.get('product_recommended', '')) or 'No clear product found'}"
        )
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
