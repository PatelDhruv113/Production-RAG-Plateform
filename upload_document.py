from src.database.sqlite_manager import SQLiteManager
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_service import ChunkingService
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_retriever import VectorRetriever

db = SQLiteManager()
db.create_tables()
db.reset_document_store()

loader = DocumentLoader()
chunker = ChunkingService()
embedder = EmbeddingService()
retriever = VectorRetriever()

documents = loader.load_folder("data/documents")
print("Documents Found:", len(documents), flush=True)

all_chunks = []
chunk_mapping = {}

for doc in documents:
    print(f"Indexing {doc['filename']}...", flush=True)

    document_id = db.save_document(
        filename=doc["filename"],
        category="Unknown"
    )

    pages = doc.get(
        "pages",
        [
            {
                "page_number": 1,
                "text": doc["text"]
            }
        ]
    )

    for page in pages:
        chunks = chunker.chunk_documents(page["text"])

        for chunk in chunks:
            chunk_id = db.save_chunk(
                document_id=document_id,
                chunk_text=chunk,
                page_number=page["page_number"],
                category="Unknown",
                source_file=doc["filename"]
            )

            all_chunks.append(chunk)

            chunk_mapping[len(all_chunks) - 1] = chunk_id

if not all_chunks:
    raise RuntimeError("No chunks were created. Check data/documents for readable PDFs.")

print("Chunks Created:", len(all_chunks), flush=True)

# Generate embeddings after all chunks are collected
embeddings = embedder.generate_embeddings(all_chunks)

retriever.create_index(embeddings)

retriever.save_index("data/faiss_index/rag.index")

print("Done", flush=True)
