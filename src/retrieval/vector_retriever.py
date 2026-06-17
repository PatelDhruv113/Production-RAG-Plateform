import faiss
import numpy as np

class VectorRetriever:

    def __init__(self):

        self.index = None

    def create_index(self,embeddings):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

    def save_index(self, path):

        faiss.write_index(self.index, path)

    def load_index(self, path):

        self.index = faiss.read_index(path)
        
    def search(self, query_embeddings, top_k=5):

        distance, indices = self.index.search(
            query_embeddings,
            top_k
        )

        return distance, indices