import re


class ConfidenceScorer:

    # def score(self, distances):

    #     avg_distance = (
    #         distances[0].mean()
    #     )

    #     confidence = max(0, 1 - avg_distance)

    #     return round(confidence*100, 2 )

    def _content_words(self, text):

        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "in",
            "is",
            "of",
            "or",
            "the",
            "to",
            "what",
            "which",
            "who",
            "why",
            "how"
        }

        return {
            word
            for word in re.findall(r"\w+", text.lower())
            if word not in stop_words and len(word) > 2
        }

    def is_not_found(self, answer):

        if not answer:

            return True

        return "could not find the answer" in answer.lower()

    def score(self, retrieved_chunks, query="", answer=""):

        if not retrieved_chunks or self.is_not_found(answer):

            return 0

        query_words = self._content_words(query)

        answer_words = self._content_words(answer)

        context_text = " ".join(
            chunk["text"] if isinstance(chunk, dict) else chunk
            for chunk in retrieved_chunks
        )

        context_words = self._content_words(context_text)

        if not query_words or not context_words:

            return 0

        query_coverage = len(
            query_words.intersection(context_words)
        ) / len(query_words)

        if answer_words:

            answer_grounding = len(
                answer_words.intersection(context_words)
            ) / len(answer_words)

        else:

            answer_grounding = 0

        confidence = (
            query_coverage * 70
            + answer_grounding * 30
        )

        if query_coverage < 0.25:

            confidence = min(confidence, 25)

        return round(confidence)
