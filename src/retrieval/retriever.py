import os
import logging
from typing import List, Dict, Any

from src.database.sqlite_manager import SQLiteManager
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class Retriever:
    """
    Unified Retriever managing Vector (FAISS), BM25 (Lexical), and Hybrid (Reciprocal Rank Fusion) search strategies.
    """

    INDEX_PATH = "data/faiss_index/rag.index"

    def __init__(self) -> None:
        self.db = SQLiteManager()
        self.embedder = EmbeddingService()
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        self.hybrid = HybridRetriever()

        if os.path.exists(self.INDEX_PATH):
            self.vector.load_index(self.INDEX_PATH)

    def retrieve(self, query: str, top_k: int = 8, search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant document chunks for a query using the specified search mode.

        Args:
            query (str): Search query text.
            top_k (int): Number of top results to return.
            search_type (str): Search engine mode ("hybrid", "vector", "bm25").

        Returns:
            List[Dict[str, Any]]: List of retrieved chunk dictionaries.
        """
        rows = self.db.get_all_chunks()
        if not rows:
            return []

        chunk_records = [
            {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "page": row[3]
            }
            for row in rows
        ]
        chunks_text = [record["text"] for record in chunk_records]

        if os.path.exists(self.INDEX_PATH):
            self.vector.load_index(self.INDEX_PATH)

        # Build BM25 Index
        self.bm25.build_index(chunks_text)

        # Query Embedding & Vector Search
        query_embedding = self.embedder.generate_embeddings([query])
        distances, vector_indices_raw = self.vector.search(query_embedding, top_k * 3)
        vector_indices = [
            idx for idx in vector_indices_raw[0].tolist()
            if 0 <= idx < len(chunk_records)
        ]

        # BM25 Search
        bm25_indices = self.bm25.search(query, top_k * 3).tolist()

        # Select Strategy
        stype = (search_type or "hybrid").lower()
        if "vector" in stype:
            final_indices = vector_indices
        elif "bm25" in stype:
            final_indices = bm25_indices
        else:
            final_indices = self.hybrid.retrieve(vector_indices, bm25_indices)

        # Build deduplicated result set
        retrieved_chunks: List[Dict[str, Any]] = []
        seen_texts = set()

        for idx in final_indices:
            if idx < len(chunk_records):
                chunk_record = chunk_records[idx]
                text_key = " ".join(chunk_record["text"].lower().split())
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    retrieved_chunks.append(chunk_record)

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

