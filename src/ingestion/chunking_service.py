class ChunkingService:

    def __init__(self, chunk_size=1000, chunk_overlap=200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, text:str):

        if not text:

            return []

        chunks = []
        start = 0
        text = text.strip()

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text)
            )

            split_at = text.rfind(
                "\n",
                start,
                end
            )

            if split_at <= start:

                split_at = text.rfind(
                    " ",
                    start,
                    end
                )

            if split_at <= start or end == len(text):

                split_at = end

            chunk = text[start:split_at].strip()

            if chunk:

                chunks.append(chunk)

            if split_at >= len(text):

                break

            start = max(
                split_at - self.chunk_overlap,
                0
            )

        return chunks
