from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

KB_PATH = Path('data/advisory_knowledge_base_large.json')


class AgriculturalRAG:
    def __init__(self, kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        self.texts = [doc['content'] for doc in self.documents]

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=2):
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.doc_vectors
        )[0]

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for idx in ranked_indices:
            results.append({
                'score': float(similarities[idx]),
                'document': self.documents[idx]
            })

        return results

    def generate_advisory(self, disease_name, region='India'):
        query = f'{disease_name} crop disease treatment for farmers in {region}'

        retrieved_docs = self.retrieve(query)

        advisory = []

        for item in retrieved_docs:
            advisory.append(item['document']['content'])

        return {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'generated_advisory': '\n\n'.join(advisory),
            'safety_note': (
                'AI-generated agricultural recommendations should '
                'be verified with local agricultural experts.'
            )
        }


def main():
    rag = AgriculturalRAG(KB_PATH)

    result = rag.generate_advisory('leaf_spot')

    output_lines = []

    output_lines.append('# AgriAssist Full RAG Output')
    output_lines.append('')

    output_lines.append(f"Query: {result['query']}")
    output_lines.append('')

    output_lines.append('## Retrieved Documents')
    output_lines.append('')

    for item in result['retrieved_documents']:
        output_lines.append(
            f"Similarity Score: {item['score']:.3f}"
        )
        output_lines.append(
            item['document']['content']
        )
        output_lines.append('')

    output_lines.append('## Generated Advisory')
    output_lines.append(result['generated_advisory'])
    output_lines.append('')

    output_lines.append('## Safety Note')
    output_lines.append(result['safety_note'])

    final_text = '\n'.join(output_lines)

    output_path = RESULTS_DIR / 'full_rag_output.md'

    output_path.write_text(
        final_text,
        encoding='utf-8'
    )

    print(final_text)
    print(f'\nSaved output to: {output_path}')


if __name__ == '__main__':
    main()