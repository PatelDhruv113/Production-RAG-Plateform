import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class EmbeddingService:

    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def generate_embeddings(self, texts):

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = self.model(**encoded)

        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        summed_embeddings = masked_embeddings.sum(dim=1)
        token_counts = attention_mask.sum(dim=1).clamp(min=1)

        embeddings = summed_embeddings / token_counts

        return embeddings.cpu().numpy().astype(np.float32)
