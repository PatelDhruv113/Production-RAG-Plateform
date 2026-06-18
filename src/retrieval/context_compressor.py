import re


class ContextCompressor:

    def compress(self, query, chunks):

        query_words = set(
            re.findall(
                r"\w+",
                query.lower()
            )
        )

        stop_words = {
            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "in",
            "to"
        }

        query_words = query_words - stop_words

        compressed_chunks = []

        for chunk in chunks:

            sentences = chunk.split(".")

            relevant = []

            for sentence in sentences:

                sentence_words = set(
                    re.findall(
                        r"\w+",
                        sentence.lower()
                    )
                )

                overlap = query_words.intersection(
                    sentence_words
                )

                if len(overlap) > 0:
                    relevant.append(
                        sentence.strip()
                    )

            compressed_chunks.append(
                ". ".join(relevant)
            )

        return compressed_chunks