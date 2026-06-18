class CitationService:

    def generate(self, sources):

        unique_sources = list(
            set(sources)
        )

        citation_text = "\n".join(
            [
                f"{i+1}. {source}"
                for i,source
                in enumerate(unique_sources)
            ]
        )
    
        return citation_text