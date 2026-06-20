# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import(
#     faithfulness,
#     answer_relevancy
# )

# class RagasEvaluator:

#     def evaluate(self, query, answer, context):

#         faithfulness = self._faithfulness(answer, context)

#         relevancy = self._relevancy(query, answer)

#         hallucinated = (1 if faithfulness < 50 else 0)

#         return {
#             "faithfulness": faithfulness,
#             "answer_relevancy": relevancy,
#             "hallucinated": hallucinated
#         }
    
#     def _faithfulness(self, answer, context):

#         answer_words = set(answer.lower().split())

#         context_words = set(context.lower().split())

#         overlap = len(answer_words & context_words)

#         total = len(answer_words)

#         if total == 0:
#             return 0
        
#         return round(
#             overlap/total *100 , 2
#         )
    
#     def _relevancy(self, query, answer):

#         query_words = set(query.lower().split())

#         answer_words = set(answer.lower().split())

#         overlap = len(query_words & answer_words)

#         total = len(query_words)

#         if total == 0:
#             return 0
        
#         return round(
#             overlap / total * 100, 2
#         )



class RagasEvaluator:

    def evaluate(
        self,
        query,
        answer,
        context
    ):

        faithfulness_score = self._faithfulness(
            answer,
            context
        )

        relevancy_score = self._relevancy(
            query,
            answer
        )

        judge_score = round(
         (
             faithfulness_score +
             relevancy_score
         ) / 2,
         2
        )

        hallucinated = (
            1
            if faithfulness_score < 50
            else 0
        )

        return {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "judge_score": judge_score,
            "hallucinated": hallucinated
        }

    def _faithfulness(
        self,
        answer,
        context
    ):

        answer_words = set(
            answer.lower().split()
        )

        context_words = set(
            context.lower().split()
        )

        overlap = len(
            answer_words & context_words
        )

        total = len(
            answer_words
        )

        if total == 0:
            return 0

        return round(
            (overlap / total) * 100,
            2
        )

    def _relevancy(
        self,
        query,
        answer
    ):

        query_words = set(
            query.lower().split()
        )

        answer_words = set(
            answer.lower().split()
        )

        overlap = len(
            query_words & answer_words
        )

        total = len(
            query_words
        )

        if total == 0:
            return 0

        return round(
            (overlap / total) * 100,
            2
        )