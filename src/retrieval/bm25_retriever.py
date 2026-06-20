from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self):

        self.bm25 = None
        self.chunks = None

    def build_index(self, chunks):

        if not chunks:
            raise ValueError(
                "No chunks found in database"
            )
        
        self.chunks = chunks

        tokenized_chunks = [
            chunk.split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        rank_indices = (scores.argsort()[::-1][:top_k])

        return rank_indices