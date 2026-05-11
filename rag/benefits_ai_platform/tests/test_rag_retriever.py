from pathlib import Path

from src.rag.retrieve import BenefitsRetriever


def test_rag_retriever_loads_index():
    project_root = Path.cwd()
    retriever = BenefitsRetriever(project_root=project_root)

    assert retriever.index is not None
    assert len(retriever.metadata) > 0


def test_rag_retriever_returns_results():
    project_root = Path.cwd()
    retriever = BenefitsRetriever(project_root=project_root)

    results = retriever.retrieve(
        query="What is the out-of-pocket maximum?",
        top_k=3,
    )

    assert isinstance(results, list)
    assert len(results) == 3

    first_result = results[0]

    assert "document_name" in first_result
    assert "chunk_id" in first_result
    assert "text" in first_result
    assert "score" in first_result