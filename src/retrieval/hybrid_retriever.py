class HybridRetriever:

    def reciprocal_rank_fusion( self, vector_indices, bm25_indices, k=60):

        scores = {}

        for rank, idx in enumerate(vector_indices):

            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1) # RPF formula

        for rank, idx in enumerate(bm25_indices):

            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            idx
            for idx, _
            in sorted_results
        ]

    def retrieve(self, vector_indices, bm25_indices):

        return self.reciprocal_rank_fusion(vector_indices, bm25_indices)