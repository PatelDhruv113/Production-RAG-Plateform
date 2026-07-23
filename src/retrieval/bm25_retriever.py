from rank_bm25 import BM25Okapi
import re

class BM25Retriever:

    def __init__(self):

        self.bm25 = None
        self.chunks = None
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

    def tokenize(self, text):

        return [
            word
            for word in re.findall(r"\w+", text.lower())
            if word not in self.stop_words and len(word) > 2
        ]

    def build_index(self, chunks):

        if not chunks:
            raise ValueError(
                "No chunks found in database"
            )
        
        self.chunks = chunks

        tokenized_chunks = [
            self.tokenize(chunk)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        rank_indices = (scores.argsort()[::-1][:top_k])

        return rank_indices
