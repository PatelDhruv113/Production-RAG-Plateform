class ConfidenceScorer:

    def score(self, distances):

        avg_distance = (
            distances[0].mean()
        )

        confidence = max(0, 1 - avg_distance)

        return round(confidence*100, 2 )