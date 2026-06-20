class MultiQueryRetriever:

    def merger_results(self, all_results):

        merged = []

        seen = set()

        for result in all_results:

            for chunk in result:

                chunk_id = chunk["id"] if isinstance(chunk, dict) else chunk

                if chunk_id not in seen:

                    seen.add(chunk_id)

                    merged.append(chunk)

        return merged
