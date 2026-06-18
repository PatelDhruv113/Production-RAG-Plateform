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

    def retrieve(self,query, top_k=5):

       rows = self.db.get_all_chunks()

       chunks = [
           row[1]
           for row in rows
       ]

       #Bm25 
       self.bm25.build_index(chunks)

       #query embedding
       query_embedding = (
           self.embedder.generate_embeddings(
               [query]
           )
       )

       #vector search
       distances, vector_indices = (
           self.vector_search(
               query_embedding,
               top_k
           )
       )

       vector_indices = (
           vector_indices[0].tolist()
       )
       
       #Bm25 Search
       bm25_indices = (
           self.bm25_search(
              query,
              top_k
           ).tolist()
       )

       #Hybrid Fusion
       final_indices = (
           self.hybrid.retrieve(
               vector_indices,
               bm25_indices
           )
       )

       retrieved_chunks = []

       for idx in final_indices[:top_k]:
           
           if idx < len(chunks):
               
               retrieved_chunks.append(chunks[idx])

       return retrieved_chunks
