import logging
from typing import Dict, Any
from src.evaluation.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    RAG Evaluation Engine providing automated Faithfulness, Relevancy,
    Hallucination Detection, and LLM-as-Judge metrics with robust fallback heuristics.
    """

    def __init__(self) -> None:
        self.judge_engine = LLMJudge()

    def evaluate(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluates answer quality against context and query.

        Args:
            query (str): User question.
            answer (str): Generated response.
            context (str): Retrieved context.

        Returns:
            Dict[str, Any]: Evaluation dictionary containing scores, hallucination flag, and rationale.
        """
        # Attempt evaluation via LLM Judge
        judge_res = self.judge_engine.judge(question=query, answer=answer, context=context)
        if judge_res:
            return judge_res

        # Fallback to word-overlap heuristic if LLM Judge is offline/unavailable
        faithfulness_score = self._faithfulness(answer, context)
        relevancy_score = self._relevancy(query, answer)
        judge_score = round((faithfulness_score + relevancy_score) / 2.0, 2)
        hallucinated = 1 if faithfulness_score < 50.0 else 0

        return {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "judge_score": judge_score,
            "hallucinated": hallucinated,
            "reasoning": "Evaluated using heuristic word-overlap model (LLM Judge offline)."
        }

    def _faithfulness(self, answer: str, context: str) -> float:
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return 0.0
        overlap = len(answer_words & context_words)
        return round((overlap / len(answer_words)) * 100.0, 2)

    def _relevancy(self, query: str, answer: str) -> float:
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & answer_words)
        return round((overlap / len(query_words)) * 100.0, 2)
