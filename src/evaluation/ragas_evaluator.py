from datasets import DataSet
from ragas import evaluate
from ragas.metrics import(
    faithfulness,
    answer_relevancy
)

class RagasEvaluator:

    def evaluate(self, questions, answers, contexts, ground_truths):

        dataset = DataSet.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths

            }
        )

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ]
        )

        return result