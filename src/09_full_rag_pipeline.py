from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

KB_PATH = Path('data/advisory_knowledge_base_large.json')


class AgriculturalRAG:
    def __init__(self, kb_path):
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {kb_path}. Run: python3 src/10_build_large_kb.py"
            )

        with open(kb_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        self.texts = [doc['content'] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                'score': float(similarities[idx]),
                'document': self.documents[idx]
            })
        return results

    def generate_advisory(self, disease_name, region='India', top_k=5):
        query = f'{disease_name} crop disease treatment for farmers in {region}'
        retrieved_docs = self.retrieve(query, top_k=top_k)

        advisory_sections = []
        for item in retrieved_docs:
            advisory_sections.append(item['document']['content'])

        combined = '\n\n'.join(advisory_sections)

        final_advisory = f'''
Agricultural Advisory Report

Disease Query: {disease_name}
Region: {region}

The following recommendations were generated using a Retrieval-Augmented Generation pipeline. The system searched an external farmer advisory knowledge base, ranked documents using cosine similarity, and selected the top {top_k} most relevant records.

Retrieved Evidence Summary:
{combined}

Detailed Recommendation:
Farmers should begin by inspecting plants regularly for visible symptoms such as yellowing, spots, fungal growth, curling, drying, wilting, or unusual discoloration. If symptoms are detected early, affected leaves or plant material should be removed carefully to reduce disease spread. Field sanitation is important because infected debris can continue to carry pathogens and increase future outbreaks.

Irrigation should be managed carefully. Excess water, poor drainage, and frequent leaf wetness can increase the risk of fungal and bacterial disease. Drip irrigation or root-zone watering is preferred when possible because it reduces moisture on leaves. Proper spacing between plants improves airflow and reduces humidity around the crop canopy.

Nutrient management should also be balanced. Overuse of fertilizer, especially nitrogen, may increase vulnerability to some diseases, while poor soil nutrition can reduce plant immunity. Farmers should consider soil testing, crop rotation, resistant varieties, and seed treatment as preventive practices.

Integrated Pest Management should be followed before excessive pesticide usage. This includes cultural practices, biological control, field monitoring, and chemical control only when necessary. Local agricultural officers, certified agronomists, or regional extension workers should be consulted before applying fungicides or pesticides.

This advisory is generated from retrieved agricultural documents and should be treated as decision-support guidance rather than a replacement for professional agricultural consultation.
'''

        return {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'generated_advisory': final_advisory,
            'safety_note': 'AI-generated agricultural recommendations should be verified with local agricultural experts.'
        }


def main():
    rag = AgriculturalRAG(KB_PATH)
    query = input("\nEnter agricultural query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    result = rag.generate_advisory(query, top_k=5)

    output_lines = []
    output_lines.append('# AgriAssist Full RAG Output')
    output_lines.append('')
    output_lines.append(f"Query: {result['query']}")
    output_lines.append('')
    output_lines.append('## Retrieved Documents')
    output_lines.append('')

    for i, item in enumerate(result['retrieved_documents'], start=1):
        output_lines.append(f"### Document {i}")
        output_lines.append(f"Similarity Score: {item['score']:.3f}")
        output_lines.append(item['document']['content'])
        output_lines.append('')

    output_lines.append('## Generated Advisory')
    output_lines.append(result['generated_advisory'])
    output_lines.append('')
    output_lines.append('## Safety Note')
    output_lines.append(result['safety_note'])

    final_text = '\n'.join(output_lines)
    output_path = RESULTS_DIR / 'full_rag_output.md'
    output_path.write_text(final_text, encoding='utf-8')

    print(final_text)
    print(f'\nSaved output to: {output_path}')


if __name__ == '__main__':
    main()
