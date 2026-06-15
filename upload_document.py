from src.database.sqlite_manager import SQLiteManager
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_service import ChunkingService

db = SQLiteManager()

loader = DocumentLoader()

chunker = ChunkingService()


documents = loader.load_folder(
    "data/documents"
)

for doc in documents:

    document_id = db.save_document(
        filename=doc["filename"],
        category="Unknown"
    )

    chunks = chunker.chunk_documents(
        doc["text"]
    )

    for chunk in chunks:

        db.save_chunk(
            document_id=document_id,
            chunk_text=chunk,
            page_number=0,
            category="Unknown"
        )
print("Ingestion Complete")

