class CitationService:

    def generate(self, chunks):

        citations=[]

        for chunk in chunks:

            if not isinstance(chunk, dict):

                continue
                
            source = chunk["source"]

            page = chunk["page"]

            citations.append(
                f"{source} (page {page})"
            )

        return list(
            dict.fromkeys(citations)
        )
