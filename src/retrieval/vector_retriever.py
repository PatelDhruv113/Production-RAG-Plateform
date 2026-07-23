import faiss
import numpy as np
import os

class VectorRetriever:

    def __init__(self):

        self.index = None

    def create_index(self,embeddings):

        dimension = embeddings.shape[1] #(100, 384)  shape[1] 384

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

    def add_embeddings(self, embeddings):

        embeddings = np.array(
            embeddings,
            dtype=np.float32  # if not this dtype use so they took by default falt64 why falt32 for fast retrievation.
        )

        if self.index is None:

            self.create_index(
                embeddings
            )

        else:

            self.index.add(
                embeddings
            )
            
    def save_index(self, path):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        faiss.write_index(self.index, path)

    def load_index(self, path):

        if not os.path.exists(path):

            self.index = None

            return False

        self.index = faiss.read_index(path)

        return True
        
    def search(self, query_embeddings, top_k=5):

        if self.index is None:

            return (
                np.array([[]], dtype=np.float32),
                np.array([[]], dtype=np.int64)
            )

        distance, indices = self.index.search(
            query_embeddings,
            top_k
        )

        return distance, indices
