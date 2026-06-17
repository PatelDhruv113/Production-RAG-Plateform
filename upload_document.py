from src.database.sqlite_manager import SQLiteManager
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_service import ChunkingService
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.chunk_mapper import ChunkMapper
from src.retrieval.bm25_retriever import BM25Retriever

# db = SQLiteManager()

# loader = DocumentLoader()

# chunker = ChunkingService()

# embedder = EmbeddingService()

# retriever = VectorRetriever()

# mapper = ChunkMapper()

# documents = loader.load_folder(
#     "data/documents"
# )

# all_chunks = []
# chunk_mapping = {}

# for doc in documents:

#     document_id = db.save_document(
#         filename=doc["filename"],
#         category="Unknown"
#     )

#     chunks = chunker.chunk_documents(
#         doc["text"]
#     )

#     for chunk in chunks:

#         db.save_chunk(
#             document_id=document_id,
#             chunk_text=chunk,
#             page_number=0,
#             category="Unknown"
#         )

#         all_chunks.append(chunk)
        
#         chunk_mapping[len(all_chunks) - 1] = chunk_id
# embeddings = embedder.generate_embeddings(
#     all_chunks
# )

# retriever.create_index(
#     embeddings
# )


# retriever.save_index(
#     "data/faiss_index/rag.index"
# )

# print("Done")






db = SQLiteManager()

rows = db.get_all_chunks()

chunks = [
    row[1]
    for row in rows
]

bm25 = BM25Retriever()

bm25.build_index(
    chunks
)

results = bm25.search(
    "derivative rules"
)

print(results)