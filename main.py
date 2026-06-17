# from src.database.sqlite_manager import SQLiteManager
# from src.ingestion.document_loader import DocumentLoader
# from src.ingestion.chunking_service import ChunkingService


# loader = DocumentLoader()

# db = SQLiteManager()


# chunker = ChunkingService()
# # db.create_tables()

# # document_id = db.save_document(
# #     filename="calculas.pdf",
# #     category = "Maths"
# # )


# # print(document_id)



# # documents = db.get_all_documents()

# # for doc in documents:
# #     print(doc)


# documents = loader.load_folder("data/documents")


# chunks = chunker.chunk_documents(documents[0]["text"])

# print("Chunks:", len(chunks))

# if len(chunks) > 0:
#     print(chunks[0])











from src.database.sqlite_manager import SQLiteManager
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

hybrid = HybridRetriever()

# db = SQLiteManager()

# embedder = EmbeddingService()

# retriever = VectorRetriever()

# retriever.load_index(
#     "data/faiss_index/rag.index"
# )

# query = "What is derivative?"

# query_embedding = embedder.generate_embeddings(
#     [query]
# )

# distances, indices = retriever.search(
#     query_embedding,
#     top_k=5
# )

# print(indices)




vector_results = [5, 8, 10, 2]

bm25_results = [10, 5, 1, 8]


results = hybrid.retrieve(
    vector_results,
    bm25_results
)

print(results)












# # print("Step 1: Importing DocumentLoader...")
# # import sys
# # import os

# # # Disable model auto-downloads from HuggingFace
# # os.environ['HF_HUB_OFFLINE'] = '1'
# # os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# # try:
# #     from src.ingestion.document_loader import DocumentLoader
# #     print("Step 2: DocumentLoader imported successfully")
# # except Exception as e:
# #     print(f"ERROR importing DocumentLoader: {e}")
# #     sys.exit(1)

# # try:
# #     print("Step 3: Importing ChunkingService...")
# #     from src.ingestion.chunking_service import ChunkingService
# #     print("Step 4: ChunkingService imported successfully")
# # except Exception as e:
# #     print(f"ERROR importing ChunkingService: {e}")
# #     sys.exit(1)

# # try:
# #     print("Step 5: Creating loader instance...")
# #     loader = DocumentLoader()
# #     print("Step 6: Loader created")

# #     print("Step 7: Loading documents...")
# #     documents = loader.load_folder("data/documents")
# #     print(f"Step 8: Loaded {len(documents)} documents")

# #     if len(documents) == 0:
# #         print("ERROR: No documents found")
# #         sys.exit(1)

# #     print("Step 9: Creating chunker instance...")
# #     chunker = ChunkingService()
# #     print("Step 10: Chunker created")

# #     print("Step 11: Chunking documents...")
# #     chunks = chunker.chunk_documents(documents[0]["text"])
# #     print(f"Step 12: Chunked into {len(chunks)} chunks")

# #     print("\nFirst Chunk:\n")
# #     print(chunks[0])

# # except Exception as e:
# #     print(f"ERROR: {e}")
# #     import traceback
# #     traceback.print_exc()
# #     sys.exit(1)