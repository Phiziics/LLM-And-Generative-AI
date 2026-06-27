# Import tools for working with file paths
from pathlib import Path

# Import regular expressions for text cleaning and tokenization
import re

# Import pandas for working with chunk tables
import pandas as pd

# Import numpy for JSON-safe numeric conversion
import numpy as np

# Import ChromaDB for vector database search
import chromadb

# Import SentenceTransformer for query embeddings
from sentence_transformers import SentenceTransformer

# Import CrossEncoder for reranking retrieved chunks
from sentence_transformers import CrossEncoder

# Import BM25 for keyword search
from rank_bm25 import BM25Okapi


class RiskRadarRetriever:
    """
    Retrieval engine for RiskRadar AI.

    This class loads:
    - SEC RAG chunks
    - embedding model
    - Chroma vector database
    - BM25 keyword search
    - cross-encoder reranker

    It returns citation-ready SEC evidence.
    """

    def __init__(self, project_root=None, load_reranker=True):
        """
        Initialize the retriever.
        """

        # Resolve project root automatically if not provided
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)

        # Store whether reranker should be loaded
        self.load_reranker = load_reranker

        # Define important folders
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.vectorstore_dir = self.data_dir / "vectorstore"
        self.chroma_dir = self.vectorstore_dir / "chroma_sec_10k"

        # Define required files
        self.rag_chunks_file = self.processed_dir / "sec_10k_rag_chunks.csv"
        self.embedding_summary_file = self.processed_dir / "sec_10k_embedding_summary.csv"

        # Load all required project assets
        self._load_data()
        self._load_models()
        self._load_chroma_collection()

        # Build BM25 corpus once for unfiltered search
        self._build_bm25_index()

    def _load_data(self):
        """
        Load RAG chunks and embedding settings.
        """

        # Stop if the RAG chunk file is missing
        if not self.rag_chunks_file.exists():
            raise FileNotFoundError(
                f"Missing RAG chunks file: {self.rag_chunks_file}"
            )

        # Stop if the embedding summary file is missing
        if not self.embedding_summary_file.exists():
            raise FileNotFoundError(
                f"Missing embedding summary file: {self.embedding_summary_file}"
            )

        # Load RAG chunks
        self.rag_chunks_df = pd.read_csv(self.rag_chunks_file).fillna("")

        # Load embedding summary
        embedding_summary = pd.read_csv(self.embedding_summary_file)

        # Convert embedding summary into a dictionary
        self.embedding_settings = dict(
            zip(
                embedding_summary["setting"],
                embedding_summary["value"]
            )
        )

        # Store embedding model name
        self.embedding_model_name = self.embedding_settings["embedding_model"]

        # Store Chroma collection name
        self.collection_name = self.embedding_settings["collection_name"]

    def _load_models(self):
        """
        Load embedding model and optional reranker model.
        """

        # Load embedding model used for Chroma query embeddings
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Set reranker model name
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Load reranker only if requested
        if self.load_reranker:
            self.reranker_model = CrossEncoder(self.reranker_model_name)
        else:
            self.reranker_model = None

    def _load_chroma_collection(self):
        """
        Load existing Chroma vector collection.
        """

        # Stop if Chroma folder is missing
        if not self.chroma_dir.exists():
            raise FileNotFoundError(
                f"Missing Chroma vectorstore folder: {self.chroma_dir}"
            )

        # Create persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir)
        )

        # Load existing Chroma collection
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name
        )

        # Validate collection count
        if self.collection.count() != len(self.rag_chunks_df):
            raise ValueError(
                "Chroma collection count does not match RAG chunk dataset count."
            )

    def _build_bm25_index(self):
        """
        Build BM25 index over all chunks.

        For filtered search, we build a smaller temporary BM25 index.
        """

        # Tokenize every chunk once
        self.tokenized_corpus = [
            self.tokenize_text(text)
            for text in self.rag_chunks_df["chunk_text"].tolist()
        ]

        # Build BM25 index for all chunks
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _standardize_ticker(self, ticker):
        """
        Standardize ticker input.
        """

        # Return None if no ticker was provided
        if ticker is None:
            return None

        # Convert ticker to uppercase string
        return str(ticker).upper().strip()

    def _convert_record_to_json_safe(self, record):
        """
        Convert pandas/numpy values into JSON-safe Python values.
        """

        # Create empty dictionary
        clean_record = {}

        # Loop through record items
        for key, value in record.items():

            # Convert numpy integer to Python integer
            if isinstance(value, np.integer):
                clean_record[key] = int(value)

            # Convert numpy float to Python float
            elif isinstance(value, np.floating):
                clean_record[key] = float(value)

            # Convert missing values to None
            elif pd.isna(value):
                clean_record[key] = None

            # Keep normal values as-is
            else:
                clean_record[key] = value

        # Return clean record
        return clean_record

    def get_available_companies(self):
        """
        Return companies currently available in the RAG system.
        """

        # Select company metadata
        companies_df = (
            self.rag_chunks_df[
                ["ticker", "company_name"]
            ]
            .drop_duplicates()
            .sort_values("ticker")
            .reset_index(drop=True)
        )

        # Convert to JSON-safe records
        records = [
            self._convert_record_to_json_safe(record)
            for record in companies_df.to_dict(orient="records")
        ]

        # Return records
        return records

    def get_available_sections(self):
        """
        Return SEC sections currently available in the RAG system.
        """

        # Get unique section names
        sections = sorted(
            self.rag_chunks_df["section_name"].dropna().unique().tolist()
        )

        # Return sections
        return sections

    def tokenize_text(self, text):
        """
        Convert text into lowercase tokens for BM25 keyword search.
        """

        # Convert to lowercase string
        text = str(text).lower()

        # Keep only letters, numbers, and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Collapse repeated spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Split text into tokens
        tokens = text.split()

        # Return tokens
        return tokens

    def build_chroma_where_filter(self, ticker=None, section_name=None):
        """
        Build a Chroma-compatible metadata filter.

        Chroma requires multiple filters to be wrapped in $and.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Create empty list for filter conditions
        filter_conditions = []

        # Add ticker filter if provided
        if ticker is not None:
            filter_conditions.append({"ticker": ticker})

        # Add section filter if provided
        if section_name is not None:
            filter_conditions.append({"section_name": section_name})

        # Return no filter if no conditions exist
        if len(filter_conditions) == 0:
            return None

        # Return single filter directly
        if len(filter_conditions) == 1:
            return filter_conditions[0]

        # Combine multiple filters with Chroma $and
        return {"$and": filter_conditions}

    def vector_search(self, query, top_k=10, ticker=None, section_name=None):
        """
        Run semantic vector search against Chroma.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Embed the user query
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # Build optional Chroma metadata filter
        where_filter = self.build_chroma_where_filter(
            ticker=ticker,
            section_name=section_name
        )

        # Query Chroma without filter
        if where_filter is None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        # Query Chroma with filter
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

        # Create list for result records
        result_records = []

        # Loop through returned results
        for rank, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):

            # Store one result
            result_records.append({
                "chunk_id": metadata.get("chunk_id", ""),
                "vector_rank": rank,
                "vector_distance": distance,
                "ticker": metadata.get("ticker", ""),
                "company_name": metadata.get("company_name", ""),
                "filing_date": metadata.get("filing_date", ""),
                "section_name": metadata.get("section_name", ""),
                "citation_label": metadata.get("citation_label", ""),
                "filing_url": metadata.get("filing_url", ""),
                "chunk_text": doc
            })

        # Return results as DataFrame
        return pd.DataFrame(result_records)

    def bm25_search(self, query, top_k=10, ticker=None, section_name=None):
        """
        Run BM25 keyword search over the chunk dataset.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Create filtered search DataFrame
        search_df = self.rag_chunks_df.copy()

        # Apply ticker filter if provided
        if ticker is not None:
            search_df = search_df[search_df["ticker"] == ticker].copy()

        # Apply section filter if provided
        if section_name is not None:
            search_df = search_df[search_df["section_name"] == section_name].copy()

        # Reset index after filtering
        search_df = search_df.reset_index(drop=True)

        # Return empty DataFrame if no records match
        if search_df.empty:
            return pd.DataFrame()

        # Tokenize filtered corpus
        tokenized_corpus = [
            self.tokenize_text(text)
            for text in search_df["chunk_text"].tolist()
        ]

        # Build BM25 index for filtered corpus
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize query
        tokenized_query = self.tokenize_text(query)

        # Score filtered chunks
        scores = bm25.get_scores(tokenized_query)

        # Add BM25 scores
        search_df["bm25_score"] = scores

        # Sort by BM25 score
        search_df = search_df.sort_values(
            "bm25_score",
            ascending=False
        ).head(top_k).copy()

        # Add BM25 rank
        search_df["bm25_rank"] = range(1, len(search_df) + 1)

        # Return only useful columns
        output_df = search_df[
            [
                "chunk_id",
                "bm25_rank",
                "bm25_score",
                "ticker",
                "company_name",
                "filing_date",
                "section_name",
                "citation_label",
                "filing_url",
                "chunk_text"
            ]
        ].copy()

        # Return BM25 output
        return output_df

    def hybrid_search(
        self,
        query,
        top_k=5,
        candidate_k=20,
        ticker=None,
        section_name=None,
        rrf_k=60
    ):
        """
        Combine vector search and BM25 search using reciprocal rank fusion.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Run vector search
        vector_df = self.vector_search(
            query=query,
            top_k=candidate_k,
            ticker=ticker,
            section_name=section_name
        )

        # Run BM25 search
        bm25_df = self.bm25_search(
            query=query,
            top_k=candidate_k,
            ticker=ticker,
            section_name=section_name
        )

        # Create dictionary to merge results by chunk ID
        merged_records = {}

        # Add vector results
        for _, row in vector_df.iterrows():

            # Get chunk ID
            chunk_id = row["chunk_id"]

            # Initialize merged record if needed
            if chunk_id not in merged_records:
                merged_records[chunk_id] = {
                    "chunk_id": chunk_id,
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "filing_date": row["filing_date"],
                    "section_name": row["section_name"],
                    "citation_label": row["citation_label"],
                    "filing_url": row["filing_url"],
                    "chunk_text": row["chunk_text"],
                    "vector_rank": None,
                    "vector_distance": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0
                }

            # Store vector metadata
            merged_records[chunk_id]["vector_rank"] = row["vector_rank"]
            merged_records[chunk_id]["vector_distance"] = row["vector_distance"]

            # Add reciprocal rank fusion score from vector rank
            merged_records[chunk_id]["rrf_score"] += 1 / (rrf_k + row["vector_rank"])

        # Add BM25 results
        for _, row in bm25_df.iterrows():

            # Get chunk ID
            chunk_id = row["chunk_id"]

            # Initialize merged record if needed
            if chunk_id not in merged_records:
                merged_records[chunk_id] = {
                    "chunk_id": chunk_id,
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "filing_date": row["filing_date"],
                    "section_name": row["section_name"],
                    "citation_label": row["citation_label"],
                    "filing_url": row["filing_url"],
                    "chunk_text": row["chunk_text"],
                    "vector_rank": None,
                    "vector_distance": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0
                }

            # Store BM25 metadata
            merged_records[chunk_id]["bm25_rank"] = row["bm25_rank"]
            merged_records[chunk_id]["bm25_score"] = row["bm25_score"]

            # Add reciprocal rank fusion score from BM25 rank
            merged_records[chunk_id]["rrf_score"] += 1 / (rrf_k + row["bm25_rank"])

        # Convert merged records to DataFrame
        hybrid_df = pd.DataFrame(list(merged_records.values()))

        # Return empty DataFrame safely
        if hybrid_df.empty:
            return pd.DataFrame()

        # Sort by RRF score
        hybrid_df = hybrid_df.sort_values(
            "rrf_score",
            ascending=False
        ).head(top_k).reset_index(drop=True)

        # Add hybrid rank
        hybrid_df["hybrid_rank"] = range(1, len(hybrid_df) + 1)

        # Return hybrid results
        return hybrid_df

    def rerank_results(self, query, candidate_df, top_k=5):
        """
        Rerank candidate chunks using the cross-encoder reranker.
        """

        # Return empty DataFrame if no candidates exist
        if candidate_df.empty:
            return pd.DataFrame()

        # If reranker was not loaded, return top hybrid results
        if self.reranker_model is None:
            output_df = candidate_df.head(top_k).copy()
            output_df["rerank_score"] = None
            output_df["rerank_rank"] = range(1, len(output_df) + 1)
            return output_df

        # Create query-document pairs
        pairs = [
            [query, chunk_text]
            for chunk_text in candidate_df["chunk_text"].tolist()
        ]

        # Predict reranker relevance scores
        rerank_scores = self.reranker_model.predict(pairs)

        # Copy candidate DataFrame
        reranked_df = candidate_df.copy()

        # Add reranker scores
        reranked_df["rerank_score"] = rerank_scores

        # Sort by reranker score
        reranked_df = reranked_df.sort_values(
            "rerank_score",
            ascending=False
        ).head(top_k).reset_index(drop=True)

        # Add reranker rank
        reranked_df["rerank_rank"] = range(1, len(reranked_df) + 1)

        # Return reranked results
        return reranked_df

    def retrieve_evidence(
        self,
        query,
        top_k=5,
        candidate_k=30,
        ticker=None,
        section_name=None,
        use_reranker=True
    ):
        """
        Final evidence retrieval function.

        Steps:
        1. Hybrid search.
        2. Optional reranking.
        3. Return citation-ready evidence.
        """

        # Standardize ticker
        ticker = self._standardize_ticker(ticker)

        # Run hybrid search to collect candidates
        hybrid_candidates = self.hybrid_search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            ticker=ticker,
            section_name=section_name
        )

        # Return empty DataFrame if no candidates exist
        if hybrid_candidates.empty:
            return pd.DataFrame()

        # Rerank if requested
        if use_reranker:
            final_results = self.rerank_results(
                query=query,
                candidate_df=hybrid_candidates,
                top_k=top_k
            )

        # Otherwise use hybrid order
        else:
            final_results = hybrid_candidates.head(top_k).copy()

            # Add columns for consistent output
            final_results["rerank_score"] = None
            final_results["rerank_rank"] = range(1, len(final_results) + 1)

        # Fill missing values
        final_results = final_results.fillna("")

        # Return final evidence DataFrame
        return final_results

    def retrieve_evidence_records(
        self,
        query,
        top_k=5,
        candidate_k=30,
        ticker=None,
        section_name=None,
        use_reranker=True
    ):
        """
        Retrieve evidence and return JSON-safe records.
        """

        # Retrieve evidence DataFrame
        evidence_df = self.retrieve_evidence(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            ticker=ticker,
            section_name=section_name,
            use_reranker=use_reranker
        )

        # Return empty list if no evidence exists
        if evidence_df.empty:
            return []

        # Convert records to JSON-safe dictionaries
        records = [
            self._convert_record_to_json_safe(record)
            for record in evidence_df.to_dict(orient="records")
        ]

        # Return records
        return records

    def health_check(self):
        """
        Return retriever health status.
        """

        # Return status dictionary
        return {
            "rag_chunks_file_exists": self.rag_chunks_file.exists(),
            "embedding_summary_file_exists": self.embedding_summary_file.exists(),
            "chroma_dir_exists": self.chroma_dir.exists(),
            "chunk_rows": len(self.rag_chunks_df),
            "chroma_records": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection_name,
            "reranker_loaded": self.reranker_model is not None,
            "available_companies": self.get_available_companies(),
            "available_sections": self.get_available_sections(),
        }