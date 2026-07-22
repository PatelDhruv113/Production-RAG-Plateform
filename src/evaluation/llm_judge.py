import json
import logging
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    LLM-as-a-Judge evaluator for automated RAG answer quality and hallucination detection.
    Enforces structured JSON output schema for Faithfulness, Relevancy, Hallucination, and Reasoning.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
        try:
            self.llm = ChatGroq(
                model_name=model_name,
                temperature=0
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq for LLMJudge: {e}")
            self.llm = None

    def judge(self, question: str, answer: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates a RAG response against retrieved context and original question.

        Args:
            question (str): User query.
            answer (str): Generated answer.
            context (str): Retrieved context passages.

        Returns:
            Optional[Dict[str, Any]]: Evaluation scores and reasoning, or None if evaluation fails.
        """
        if not self.llm:
            return None

        prompt = f"""You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Analyze the provided Question, Context, and Answer.

Question:
{question}

Context:
{context}

Answer:
{answer}

Evaluate the following metrics on a scale of 0 to 100:
1. Faithfulness: Is the Answer fully supported by the Context without making up facts?
2. Relevancy: Does the Answer directly address the Question?
3. Judge Score: Overall quality score (0-100).
4. Hallucinated: 1 if the answer contains information NOT supported by Context, else 0.
5. Reasoning: A 1-2 sentence explanation of your decision.

Respond ONLY with a valid JSON object matching this schema:
{{
  "faithfulness": <number 0-100>,
  "answer_relevancy": <number 0-100>,
  "judge_score": <number 0-100>,
  "hallucinated": <0 or 1>,
  "reasoning": "<explanation string>"
}}
"""
        try:
            response = self.llm.invoke(prompt)
            content = str(response.content).strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return {
                "faithfulness": float(data.get("faithfulness", 0.0)),
                "answer_relevancy": float(data.get("answer_relevancy", 0.0)),
                "judge_score": float(data.get("judge_score", 0.0)),
                "hallucinated": int(data.get("hallucinated", 0)),
                "reasoning": str(data.get("reasoning", "Evaluated by LLM-as-Judge"))
            }
        except Exception as e:
            logger.error(f"LLM Judge execution or parsing error: {e}")
            return None
