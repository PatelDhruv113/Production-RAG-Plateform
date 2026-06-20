class Reranker:

    def __init__(self):
        pass

    def rerank(self, query, chunks, top_k=5 ):

        query_terms = set(query.lower().split())

        def score(chunk):
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            chunk_terms = set(text.lower().split())
            return len(query_terms.intersection(chunk_terms))

        ranked = sorted(
            chunks,
            key=score,
            reverse=True
        )
 
        return ranked[:top_k]
