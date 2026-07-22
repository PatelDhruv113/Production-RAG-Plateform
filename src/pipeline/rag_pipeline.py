import time
import logging
from typing import Dict, Any, List

from src.query_processing.query_rewriter import QueryRewriter
from src.query_processing.multi_query_generator import MultiQueryGenerator
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_compressor import ContextCompressor
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_service import CitationService
from src.generation.confidence_service import ConfidenceScorer
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.observability.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Enterprise-grade Retrieval-Augmented Generation (RAG) Pipeline.
    Integrates query re-writing, multi-query expansion, hybrid retrieval,
    re-ranking, context compression, citation scoring, and automated evaluation.
    """

    def __init__(self) -> None:
        self.rewriter = QueryRewriter()
        self.multi_query = MultiQueryGenerator()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.compressor = ContextCompressor()
        self.prompt_builder = PromptBuilder()
        self.generator = AnswerGenerator()
        self.citations = CitationService()
        self.confidence = ConfidenceScorer()
        self.evaluator = RagasEvaluator()
        self.logger = MetricsLogger()

    def run(self, query: str, search_type: str = "hybrid", top_k: int = 8) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline for a given user query.

        Args:
            query (str): The user input question.
            search_type (str): Retrieval mode ("hybrid", "vector", "bm25").
            top_k (int): Number of top document chunks to retrieve.

        Returns:
            Dict[str, Any]: Structured dictionary containing answer, citations,
                            retrieved chunks, confidence score, evaluation, and latency metrics.
        """
        start_total = time.perf_counter()

        # Step 1: Query Rewriting
        with self.logger.measure("query_rewrite"):
            try:
                rewritten_query = self.rewriter.rewrite(query)
            except Exception as e:
                logger.warning(f"Query rewriting fallback due to error: {e}")
                rewritten_query = query

        # Step 2: Multi-Query Expansion
        with self.logger.measure("multi_query"):
            try:
                expanded_queries = self.multi_query.generate(rewritten_query)
            except Exception as e:
                logger.warning(f"Multi-query expansion fallback due to error: {e}")
                expanded_queries = []

        all_queries = [query, rewritten_query, *expanded_queries]

        # Step 3: Retrieval (Vector / BM25 / Hybrid)
        with self.logger.measure("retrieval"):
            retrieved_pool: List[Dict[str, Any]] = []
            for q in all_queries:
                chunks = self.retriever.retrieve(query=q, top_k=top_k, search_type=search_type)
                retrieved_pool.extend(chunks)

            # Deduplicate by Chunk ID and Text Key
            deduped_chunks: List[Dict[str, Any]] = []
            seen_chunk_ids = set()
            for chunk in retrieved_pool:
                chunk_id = chunk["id"] if isinstance(chunk, dict) else chunk
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    deduped_chunks.append(chunk)

            unique_chunks: List[Dict[str, Any]] = []
            seen_chunk_texts = set()
            for chunk in deduped_chunks:
                chunk_text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
                text_key = " ".join(chunk_text.lower().split())
                if text_key not in seen_chunk_texts:
                    seen_chunk_texts.add(text_key)
                    unique_chunks.append(chunk)

            candidate_chunks = unique_chunks

        # Step 4: Re-ranking
        with self.logger.measure("reranking"):
            reranked_chunks = self.reranker.rerank(query=query, chunks=candidate_chunks, top_k=top_k)

        # Step 5: Context Compression
        with self.logger.measure("context_compression"):
            compressed_chunks = self.compressor.compress(query=query, chunks=reranked_chunks)
            context = "\n\n".join(
                chunk["text"] if isinstance(chunk, dict) else str(chunk)
                for chunk in compressed_chunks
            )

        # Step 6 & 7: Prompt Construction & LLM Answer Generation
        with self.logger.measure("generation"):
            prompt = self.prompt_builder.build(query=query, context=context)
            answer = self.generator.generate(prompt)

        # Step 8: Automated Evaluation (LLM-as-Judge & Faithfulness)
        with self.logger.measure("evaluation"):
            evaluation = self.evaluator.evaluate(query=query, answer=answer, context=context)
            try:
                self.retriever.db.save_evaluation(
                    query=query,
                    answer=answer,
                    faithfulness=evaluation["faithfulness"],
                    answer_relevancy=evaluation["answer_relevancy"],
                    judge_score=evaluation["judge_score"],
                    hallucinated=evaluation["hallucinated"]
                )
            except Exception as db_err:
                logger.error(f"Failed to persist evaluation record to DB: {db_err}")

        # Step 9: Retrieval Confidence & Citation Generation
        confidence = self.confidence.score(reranked_chunks, query=query, answer=answer)
        sources = self.citations.generate(reranked_chunks) if confidence > 25 else []

        total_latency_ms = round((time.perf_counter() - start_total) * 1000, 2)
        metrics = self.logger.get_metrics()
        metrics["total_latency_ms"] = total_latency_ms

        return {
            "answer": answer,
            "sources": sources,
            "chunks": reranked_chunks,
            "confidence": confidence,
            "evaluation": evaluation,
            "metrics": metrics
        }


