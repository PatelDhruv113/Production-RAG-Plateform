import re


class Reranker:

    def __init__(self):

        self.stop_words = {
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
            "how",
            "under",
            "india",
            "board",
            "regulations",
            "regulation",
            "securities",
            "exchange"
        }

    def _terms(self, text):

        return {
            word
            for word in re.findall(r"\w+", text.lower())
            if word not in self.stop_words and len(word) > 2
        }

    def rerank(self, query, chunks, top_k=5 ):

        query_terms = self._terms(query)
        query_phrases = [
            "employee stock option scheme",
            "employee stock purchase scheme",
            "stock appreciation rights",
            "general employee benefits scheme",
            "retirement benefit scheme",
            "sweat equity"
        ]

        def score(chunk):
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            chunk_terms = self._terms(text)
            phrase_score = sum(
                3
                for phrase in query_phrases
                if phrase in text.lower()
            )
            return len(query_terms.intersection(chunk_terms)) + phrase_score

        ranked = sorted(
            chunks,
            key=score,
            reverse=True
        )
 
        return ranked[:top_k]
