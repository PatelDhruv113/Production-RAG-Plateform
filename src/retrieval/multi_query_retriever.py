class MultiQueryRetriever:

    def merger_results(self, all_results):

        merged = []

        seen = set()

        for result in all_results:

            for chunk in result:

                if chunk not in seen:

                    seen.add(chunk)

                    merged.append(chunk)

        return merged