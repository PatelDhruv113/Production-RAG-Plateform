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


class RAGPipeline:

        def __init__(self):

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
        
        def run( self, query: str,     category=None):

            # Step 1
            rewritten_query = self.rewriter.rewrite(
                query
            )

            # Step 2
            queries = self.multi_query.generate(
                rewritten_query
            )

            # Step 3
            all_chunks = []

            for q in queries:

                chunks = self.retriever.retrieve(
                    q,
                    category=category
                )

                all_chunks.extend(
                    chunks
                )
            deduped_chunks = []

            seen_chunk_ids = set()

            for chunk in all_chunks:

                chunk_id = chunk["id"] if isinstance(chunk, dict) else chunk

                if chunk_id not in seen_chunk_ids:

                    seen_chunk_ids.add(chunk_id)

                    deduped_chunks.append(chunk)

            all_chunks = deduped_chunks
            # Step 4
            reranked_chunks = self.reranker.rerank(
                query=query,
                chunks=all_chunks
            )

            # Step 5
            # compressed_chunks = self.compressor.compress(
            #     query=query,
            #     chunks=reranked_chunks
            # )

            # # Step 6
            # context = "\n".join(
            #     compressed_chunks
            # )
            context = "\n".join(
                chunk["text"] if isinstance(chunk, dict) else chunk
                for chunk in reranked_chunks
            )
            # print("\n===== CONTEXT =====")
            # print(context[:2000])
            # print("===================\n")
            # Step 7
            prompt = self.prompt_builder.build(
                query=query,
                context=context
            )

            # Step 8
            answer = self.generator.generate(
                prompt
            )

            evaluation = (
             self.evaluator.evaluate(
                 query=query,
                 answer=answer,
                 context=context
             )
            )

            self.retriever.db.save_evaluation(
               query=query,
               faithfulness=evaluation["faithfulness"],
               answer_relevancy=evaluation["answer_relevancy"],
               judge_score=evaluation["judge_score"],
               hallucinated=evaluation["hallucinated"]
            )

            confidence = self.confidence.score(
                reranked_chunks,
                query=query,
                answer=answer
            )

            if confidence > 25:

                sources = self.citations.generate(reranked_chunks)

            else:

                sources = []

            return {
                "answer": answer,
                "sources": sources,
                "chunks": reranked_chunks,
                "confidence": confidence,
                "evaluation": evaluation
            }
