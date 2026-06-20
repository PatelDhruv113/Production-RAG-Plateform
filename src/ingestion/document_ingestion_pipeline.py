from pathlib import Path
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_service import ChunkingService
from src.ingestion.embedding_service import EmbeddingService

from src.database.sqlite_manager import SQLiteManager

from src.retrieval.vector_retriever import VectorRetriever


class DocumentIngestionPipeline:

    def __init__(self):

        self.loader = DocumentLoader()

        self.chunker = ChunkingService()

        self.embedder = EmbeddingService()

        self.db = SQLiteManager()

        self.vector = VectorRetriever()

    def ingest(
        self,
        pdf_path,
        category
    ):


        pdf_path = Path(pdf_path)

        documents = self.loader.load_folder(
            str(pdf_path.parent)
        )

        for document in documents:

            if document["filepath"] != str(pdf_path):

                continue

            document_id = self.db.save_document(
                filename=document["filename"],
                category=category
            )

            all_embeddings = []

            for page in document["pages"]:

                page_number = page["page_number"]

                chunks = self.chunker.chunk_documents(
                    page["text"]
                )

                if not chunks:
                    continue

                embeddings = (
                    self.embedder.generate_embeddings(
                        chunks
                    )
                )

                all_embeddings.extend(
                    embeddings
                )

                for chunk in chunks:

                    self.db.save_chunk(
                        document_id=document_id,
                        chunk_text=chunk,
                        page_number=page_number,
                        category=category,
                        source_file=document["filename"]
                    )

            if all_embeddings:
                import os

                index_path = (
                    "data/faiss_index/rag.index"
                )
                
                if os.path.exists(index_path):
                
                    self.vector.load_index(
                        index_path
                    )


                self.vector.add_embeddings(
                    all_embeddings
                )

                self.vector.save_index(
                    "data/faiss_index/rag.index"
                )