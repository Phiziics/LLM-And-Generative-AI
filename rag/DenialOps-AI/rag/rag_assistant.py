# Retrieve relevant policy chunks from a local TF-IDF index and return a grounded answer

from pathlib import Path
import joblib
from sklearn.metrics.pairwise import cosine_similarity


INDEX_PATH = Path("rag/outputs/policy_tfidf_index.joblib")


def load_index():
    # Load saved TF-IDF index

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "RAG index not found. Run: python rag/ingest_policies.py"
        )

    return joblib.load(INDEX_PATH)


def retrieve_policy_context(query, k=5):
    # Retrieve top matching policy chunks for a user question

    index = load_index()

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    records = index["records"]

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, matrix).flatten()

    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        record = records[idx]

        results.append({
            "rank": rank,
            "score": float(similarities[idx]),
            "source": record["source"],
            "page": record["page"],
            "content": record["text"]
        })

    return results


def answer_policy_question(query, k=5):
    # Return a retrieval-grounded answer from the most relevant policy chunks

    sources = retrieve_policy_context(query, k=k)

    if not sources:
        return {
            "answer": "No relevant policy context was found in the current RAG document store.",
            "sources": []
        }

    answer = (
        "Based on the retrieved policy context, review the source excerpts below. "
        "This is not a final coverage decision. It is policy-grounded support for billing review.\n\n"
    )

    for source in sources:
        answer += f"Source {source['rank']}\n"
        answer += f"File: {source['source']}\n"
        answer += f"Page: {source['page']}\n"
        answer += f"Similarity score: {source['score']:.4f}\n"
        answer += f"Excerpt: {source['content'][:900]}...\n\n"

    return {
        "answer": answer,
        "sources": sources
    }


def build_claim_policy_query(claim_row):
    # Build a policy search query from a claim-like row

    hcpcs_code = claim_row.get("hcpcs_code", "")
    provider_type = claim_row.get("provider_type", "")
    place_of_service = claim_row.get("place_of_service", "")
    is_drug = claim_row.get("is_drug", "")
    provider_state = claim_row.get("provider_state", "")

    query = (
        f"Medicare coverage documentation medical necessity billing coding "
        f"HCPCS {hcpcs_code} provider type {provider_type} "
        f"place of service {place_of_service} drug indicator {is_drug} "
        f"state {provider_state}"
    )

    return query


if __name__ == "__main__":
    test_query = "What documentation is needed for medical necessity and billing review?"

    result = answer_policy_question(test_query, k=3)

    print(result["answer"])