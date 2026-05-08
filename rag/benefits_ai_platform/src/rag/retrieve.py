from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class BenefitsRetriever:
    def __init__(
        self,
        project_root: str | Path | None = None,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()

        self.index_path = (
            Path(index_path)
            if index_path
            else self.project_root / "vector_db" / "cms_plan_rag.index"
        )

        self.metadata_path = (
            Path(metadata_path)
            if metadata_path
            else self.project_root / "vector_db" / "cms_plan_chunks_metadata.pkl"
        )

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}")

        self.embedding_model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as file:
            self.metadata = pickle.load(file)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.array(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            record = self.metadata[idx].copy()
            record["score"] = float(score)
            results.append(record)

        return results


if __name__ == "__main__":
    retriever = BenefitsRetriever()

    results = retriever.retrieve(
        "What is the out-of-pocket maximum for this plan?",
        top_k=3,
    )

    for result in results:
        print("\n---")
        print("Score:", round(result["score"], 4))
        print("Document:", result["document_name"])
        print(result["text"])