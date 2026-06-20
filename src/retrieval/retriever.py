from src.database.sqlite_manager import SQLiteManager

from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever

from src.ingestion.embedding_service import EmbeddingService



class Retriever:

    def __init__(self):

         self.db = SQLiteManager()

         self.embedder = EmbeddingService()
 
         self.vector = VectorRetriever()
 
         self.bm25 = BM25Retriever()
 
         self.hybrid = HybridRetriever()

         self.vector.load_index(
            "data/faiss_index/rag.index"
         )

    def retrieve(self,query,category=None,top_k=5):


      if category:
      
          rows = self.db.get_chunks_by_category(
              category
          )
              
      
      else:
      
          rows = self.db.get_all_chunks()
      
      chunk_records = [
          {
              "id": row[0],
              "text": row[1],
              "source": row[2],
              "page": row[3]
          }
          for row in rows
      ]
      
      chunks = [
          chunk["text"]
          for chunk in chunk_records
      ]
      
      if len(chunks) == 0:
          return []
      
      # BM25
      self.bm25.build_index(chunks)
      
      # Query Embedding
      query_embedding = (
          self.embedder.generate_embeddings(
              [query]
          )
      )
      
      # Vector Search
      distances, vector_indices = (
          self.vector.search(
              query_embedding,
              top_k
          )
      )
      
      vector_indices = (
          vector_indices[0].tolist()
      )
      
      # BM25 Search
      bm25_indices = (
          self.bm25.search(
              query,
              top_k
          ).tolist()
      )
      
      # Hybrid Search
      final_indices = (
          self.hybrid.retrieve(
              vector_indices,
              bm25_indices
          )
      )
      
      retrieved_chunks = []
      
      for idx in final_indices[:top_k]:
      
          if idx < len(chunk_records):
      
              retrieved_chunks.append(
                  chunk_records[idx]
              )
      
      return retrieved_chunks