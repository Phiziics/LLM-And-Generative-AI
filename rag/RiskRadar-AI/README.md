# RiskRadar AI

RiskRadar AI is a Retrieval-Augmented Generation system that analyzes real SEC 10-K filings and answers company risk questions with grounded citations.

The project uses public SEC filing data, document processing, vector search, hybrid retrieval, reranking, local LLM generation, and evaluation to build a practical financial-risk research assistant.

---

## Project Goal

Public company filings contain valuable information about business risks, competition, regulation, cybersecurity, supply chain exposure, financial pressure, and operating uncertainty.

The problem is that SEC filings are long and difficult to analyze manually.

RiskRadar AI solves this by allowing users to ask natural language questions such as:

```text
What AI and competition risks does NVIDIA mention?
What cybersecurity risks does Microsoft describe?
What supply chain risks does Tesla mention?
What competition risks does Apple disclose?
What semiconductor competition risks does AMD mention?
```

The system retrieves relevant SEC filing evidence first, then generates a grounded answer with citations.

---

## System Workflow

```text
SEC 10-K filings
→ raw HTML download
→ clean text extraction
→ section extraction
→ chunking
→ embeddings
→ Chroma vector database
→ baseline retrieval
→ hybrid search
→ reranking
→ grounded answer generation
→ citation validation
→ RAG evaluation
```

---

## Architecture

```text
User Question
     ↓
Query Embedding
     ↓
Vector Search + BM25 Keyword Search
     ↓
Reciprocal Rank Fusion
     ↓
Cross-Encoder Reranking
     ↓
Top Evidence Chunks
     ↓
Grounded Prompt
     ↓
Local Ollama LLM
     ↓
Cited Answer
     ↓
Evaluation Layer
```

---

## Data Source

This project uses public SEC EDGAR filing data.

Initial company universe:

```text
NVDA
MSFT
AAPL
TSLA
AMD
```

The first version focuses on annual 10-K filings because they contain detailed business, risk, and financial discussion sections.

---

## Key Features

* Downloads real SEC filing metadata and 10-K documents
* Cleans raw SEC HTML into readable text
* Extracts important 10-K sections:

  * Item 1 Business
  * Item 1A Risk Factors
  * Item 7 Management Discussion and Analysis
  * Item 7A Market Risk
* Splits long sections into overlapping RAG chunks
* Creates local embeddings using Sentence Transformers
* Stores vectors in a persistent Chroma database
* Supports semantic vector search
* Supports BM25 keyword search
* Combines retrieval methods with reciprocal rank fusion
* Uses cross-encoder reranking for stronger evidence ranking
* Generates grounded answers with local Ollama
* Validates citation usage and answer-evidence overlap
* Documents failure cases and limitations

---

## Tools Used

| Tool                  | Purpose                                |
| --------------------- | -------------------------------------- |
| Python                | Main programming language              |
| pandas                | Data processing and analysis           |
| requests              | SEC API and document downloads         |
| BeautifulSoup         | HTML parsing and text extraction       |
| matplotlib            | EDA charts                             |
| sentence-transformers | Embeddings and reranking               |
| ChromaDB              | Local vector database                  |
| rank-bm25             | Keyword search                         |
| Ollama                | Local LLM answer generation            |
| DuckDB                | Optional lightweight analytics storage |
| Jupyter               | Educational notebook workflow          |

---

## Repository Structure

```text
RiskRadar-AI/
│
├── notebooks/
│   ├── 00_project_intro.ipynb
│   ├── 01_data_ingestion_sec_filings.ipynb
│   ├── 02_data_understanding_and_eda.ipynb
│   ├── 03_section_extraction.ipynb
│   ├── 04_chunking_experiments.ipynb
│   ├── 05_embeddings_and_vector_store.ipynb
│   ├── 06_baseline_retrieval.ipynb
│   ├── 07_hybrid_search_and_reranking.ipynb
│   ├── 08_rag_answer_generation.ipynb
│   └── 09_rag_evaluation.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
│
├── reports/
│   ├── rag_answer_generation_demo.md
│   └── rag_evaluation_report.md
│
├── src/
├── app/
├── tests/
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

Note: the `data/` folders are generated locally and should not be committed to GitHub.

---

## Notebook Pipeline

### `00_project_intro.ipynb`

Introduces the project, business problem, company universe, SEC CIK lookup, and project outputs.

### `01_data_ingestion_sec_filings.ipynb`

Downloads SEC company filing metadata and raw 10-K HTML files.

Output examples:

```text
data/raw/sec_metadata/starter_company_all_filings.csv
data/raw/sec_metadata/starter_company_latest_10k.csv
data/raw/sec_filings/
```

### `02_data_understanding_and_eda.ipynb`

Cleans SEC HTML into readable text and explores document size, filing dates, word counts, and keyword patterns.

Output examples:

```text
data/processed/clean_sec_filing_metadata.csv
data/processed/sec_10k_keyword_counts.csv
data/processed/clean_sec_text/
```

### `03_section_extraction.ipynb`

Extracts structured 10-K sections and creates a RAG-ready section dataset.

Output examples:

```text
data/processed/sec_10k_sections_metadata.csv
data/processed/sec_10k_rag_sections.csv
data/processed/sec_sections/
```

### `04_chunking_experiments.ipynb`

Tests chunking strategies and creates the final RAG chunk dataset.

Final chunking strategy:

```text
350 words per chunk
75 words overlap
```

Output:

```text
data/processed/sec_10k_rag_chunks.csv
```

### `05_embeddings_and_vector_store.ipynb`

Creates embeddings for SEC chunks and stores them in Chroma.

Output examples:

```text
data/vectorstore/chroma_sec_10k/
data/processed/sec_10k_embedding_summary.csv
data/processed/sec_10k_baseline_retrieval_tests.csv
```

### `06_baseline_retrieval.ipynb`

Tests semantic retrieval with ticker and section filters.

Output examples:

```text
data/processed/sec_10k_filtered_retrieval_eval.csv
data/processed/sec_10k_unfiltered_retrieval_eval.csv
data/processed/sec_10k_retrieval_eval_summary.csv
```

### `07_hybrid_search_and_reranking.ipynb`

Improves retrieval using:

```text
semantic vector search
+ BM25 keyword search
+ reciprocal rank fusion
+ cross-encoder reranking
```

Output examples:

```text
data/processed/sec_10k_sample_hybrid_results.csv
data/processed/sec_10k_sample_reranked_results.csv
data/processed/sec_10k_retrieval_method_comparison.csv
```

### `08_rag_answer_generation.ipynb`

Generates grounded answers using retrieved SEC evidence and local Ollama.

Output examples:

```text
data/processed/sec_10k_rag_demo_answers.csv
data/processed/sec_10k_rag_answer_validation.csv
reports/rag_answer_generation_demo.md
```

### `09_rag_evaluation.ipynb`

Evaluates citation usage, citation validity, answer-evidence overlap, retrieval quality, and known failure cases.

Output examples:

```text
data/processed/sec_10k_citation_evaluation.csv
data/processed/sec_10k_groundedness_proxy_eval.csv
data/processed/sec_10k_rag_quality_flags.csv
data/processed/sec_10k_rag_evaluation_summary.csv
reports/rag_evaluation_report.md
```

---

## Setup

Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

Install requirements.

```powershell
pip install -r requirements.txt
```

Register the Jupyter kernel.

```powershell
python -m ipykernel install --user --name riskradar-ai --display-name "Python (RiskRadar AI)"
```

---

## Requirements

Main packages:

```text
pandas
numpy
requests
beautifulsoup4
lxml
tqdm
python-dotenv
jupyter
ipykernel
matplotlib
scikit-learn
duckdb
sentence-transformers
chromadb
rank-bm25
```

The exact local environment is stored in:

```text
requirements-lock.txt
```

---

## Local LLM Setup

This project uses Ollama for local answer generation.

Start Ollama:

```powershell
ollama serve
```

Install a local model:

```powershell
ollama pull llama3.2:3b
```

or:

```powershell
ollama pull llama3.1:8b
```

Recommended faster model:

```text
llama3.2:3b
```

Higher quality but slower model:

```text
llama3.1:8b
```

In notebook `08_rag_answer_generation.ipynb`, set:

```python
OLLAMA_MODEL = "llama3.2:3b"
```

---

## Example Questions

```text
What AI and competition risks does NVIDIA mention?
What cybersecurity risks does Microsoft mention?
What supply chain risks does Tesla mention?
What competition risks does Apple describe?
What semiconductor competition risks does AMD mention?
```

Example output format:

```text
Answer:
NVIDIA identifies competition and AI-related risk around rapid innovation, product demand, customer concentration, and execution pressure. The filing evidence indicates that the company operates in highly competitive markets where new technologies and changing customer requirements may affect results [Source 1], [Source 2].

Evidence Used:
- NVDA 2025 10-K, item_1a_risk_factors, chunk 3
- NVDA 2025 10-K, item_1a_risk_factors, chunk 7
```

---

## Evaluation Approach

The project includes a first-version RAG evaluation layer.

It checks:

```text
citation marker presence
retrieved citation count
citation validity
answer-evidence keyword overlap
retrieval hit rates
quality flags
known failure cases
manual review template
```

This is not a perfect faithfulness evaluation, but it provides a practical baseline for responsible RAG development.

---

## Known Limitations

* The system only answers from filings that were downloaded and embedded.
* It does not answer live market news questions.
* Local LLM generation can be slow depending on hardware.
* SEC section extraction may need improvement for edge-case filing formats.
* Keyword overlap is only a proxy for groundedness.
* Human review is still needed for high-stakes financial decisions.

---

## Future Improvements

Planned extensions:

```text
FastAPI backend
Streamlit or Gradio interface
structured SEC XBRL financial metrics
query routing by SEC section
stronger faithfulness evaluation
PDF or Markdown report export
Docker deployment
scheduled filing refresh
multi-company comparison reports
```

---

## Business Use Cases

RiskRadar AI can support:

* financial risk research
* competitor analysis
* investment memo preparation
* company due diligence
* regulatory risk review
* business strategy research
* analyst workflow automation

---

## Main Output

The most important generated files are:

```text
data/processed/sec_10k_rag_chunks.csv
data/vectorstore/chroma_sec_10k/
data/processed/sec_10k_rag_demo_answers.csv
reports/rag_answer_generation_demo.md
reports/rag_evaluation_report.md
```

---

## Final Summary

RiskRadar AI is a local-first RAG system for analyzing real SEC filings.

It demonstrates an end-to-end pipeline:

```text
data ingestion
→ text processing
→ section extraction
→ chunking
→ embeddings
→ vector database
→ hybrid retrieval
→ reranking
→ grounded answer generation
→ evaluation
```

The result is a practical, auditable RAG system that answers financial-risk questions using real company filing evidence.
