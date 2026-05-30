from pathlib import Path
import json
import re

from pypdf import PdfReader

PDF_DIR = Path("data/rag_documents")
OUTPUT_PATH = Path("data/processed_rag/document_knowledge_base.json")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if len(chunk.strip()) > 100:
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)

        if text:
            pages.append((page_num, text))

    return pages


def main():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {PDF_DIR}")

    records = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        pages = extract_pdf_text(pdf_path)

        for page_num, page_text in pages:
            chunks = chunk_text(page_text)

            for chunk_idx, chunk in enumerate(chunks):
                records.append({
                    "id": f"{pdf_path.stem}_page_{page_num}_chunk_{chunk_idx}",
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "crop": "general",
                    "problem_type": "general",
                    "region": "India",
                    "season": "unknown",
                    "urgency": "unknown",
                    "product_recommended": "",
                    "content": chunk
                })

    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nCreated document KB with {len(records)} chunks")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
