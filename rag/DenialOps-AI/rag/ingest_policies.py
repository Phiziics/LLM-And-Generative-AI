# Build a simple local RAG index using TF-IDF retrieval

from pathlib import Path
import joblib
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


DOCUMENTS_DIR = Path("rag/documents")
OUTPUTS_DIR = Path("rag/outputs")
INDEX_PATH = OUTPUTS_DIR / "policy_tfidf_index.joblib"


def load_txt_file(file_path):
    # Load text from a TXT file

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf_file(file_path):
    # Load text from a PDF file page by page

    reader = PdfReader(str(file_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "text": text,
                "page": page_number
            })

    return pages


def chunk_text(text, chunk_size=1000, overlap=150):
    # Split long text into overlapping chunks

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_documents():
    # Load TXT and PDF documents from rag/documents

    records = []

    for txt_path in DOCUMENTS_DIR.glob("*.txt"):
        print(f"Loading TXT: {txt_path}", flush=True)

        text = load_txt_file(txt_path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks, start=1):
            records.append({
                "id": f"{txt_path.stem}_chunk_{i}",
                "text": chunk,
                "source": str(txt_path),
                "page": "N/A"
            })

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        print(f"Loading PDF: {pdf_path}", flush=True)

        pages = load_pdf_file(pdf_path)

        for page in pages:
            chunks = chunk_text(page["text"])

            for i, chunk in enumerate(chunks, start=1):
                records.append({
                    "id": f"{pdf_path.stem}_page_{page['page']}_chunk_{i}",
                    "text": chunk,
                    "source": str(pdf_path),
                    "page": page["page"]
                })

    return records


def build_tfidf_index(records):
    # Build TF-IDF vectors for all document chunks

    texts = [record["text"] for record in records]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)

    index = {
        "records": records,
        "vectorizer": vectorizer,
        "matrix": matrix
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(index, INDEX_PATH)

    return index


def main():
    # Run the local RAG ingestion workflow

    print("Starting TF-IDF policy ingestion...", flush=True)

    records = load_documents()

    if not records:
        print("No documents found. Add TXT or PDF files to rag/documents.", flush=True)
        return

    build_tfidf_index(records)

    print(f"Created chunks: {len(records)}", flush=True)
    print(f"Saved TF-IDF index to: {INDEX_PATH}", flush=True)


if __name__ == "__main__":
    main()