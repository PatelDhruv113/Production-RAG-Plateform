from src.query_processing.query_rewriter import QueryRewriter
from src.query_processing.multi_query_generator import MultiQueryGenerator

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_compressor import ContextCompressor

from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator


class RAGPipeline:

    def __init__(self):

        self.rewriter = QueryRewriter()

        self.multi_query = MultiQueryGenerator()

        self.retriever = Retriever()

        self.reranker = Reranker()

        self.compressor = ContextCompressor()

        self.prompt_builder = PromptBuilder()

        self.generator = AnswerGenerator()

    def run(
        self,
        query: str
    ):

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
                q
            )

            all_chunks.extend(
                chunks
            )

        # Step 4
        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=all_chunks
        )

        # Step 5
        compressed_chunks = self.compressor.compress(
            query=query,
            chunks=reranked_chunks
        )

        # Step 6
        context = "\n".join(
            compressed_chunks
        )

        # Step 7
        prompt = self.prompt_builder.build(
            query=query,
            context=context
        )

        # Step 8
        answer = self.generator.generate(
            prompt
        )

        return answer