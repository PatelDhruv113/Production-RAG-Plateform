import time
import pandas as pd
import numpy as np
from src.retrieval.retriever import Retriever

class RetrievalBenchmarker:

    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()

    def generate_sample_benchmark_dataset(self, num_samples=5):
        """
        Derives sample evaluation queries and expected keywords from stored DB chunks.
        """
        db_chunks = self.retriever.db.get_all_chunks()
        if not db_chunks:
            return [
                {
                    "query": "What are the rules under SEBI board regulations?",
                    "relevant_keywords": ["sebi", "regulations", "board"]
                },
                {
                    "query": "What is employee stock option scheme?",
                    "relevant_keywords": ["employee", "stock", "option", "scheme"]
                }
            ]

        benchmark_set = []
        # Sample chunks across the DB
        indices = np.linspace(0, len(db_chunks) - 1, min(num_samples, len(db_chunks)), dtype=int)
        for idx in indices:
            row = db_chunks[idx]
            chunk_text = row[1]
            words = [w for w in chunk_text.split() if len(w) > 4][:5]
            if not words:
                continue
            query = f"Information about {' '.join(words[:3])}"
            benchmark_set.append({
                "query": query,
                "relevant_keywords": [w.lower() for w in words]
            })

        return benchmark_set

    def run_benchmark(self, benchmark_set=None, top_k=5):
        """
        Runs benchmarking across 'vector', 'bm25', and 'hybrid' strategies.
        """
        if not benchmark_set:
            benchmark_set = self.generate_sample_benchmark_dataset()

        strategies = ["vector", "bm25", "hybrid"]
        results = []

        for strategy in strategies:
            total_latency = 0
            precisions = []
            recalls = []
            mrrs = []

            for item in benchmark_set:
                query = item["query"]
                keywords = set(item["relevant_keywords"])

                start = time.perf_counter()
                retrieved = self.retriever.retrieve(query=query, top_k=top_k, search_type=strategy)
                elapsed_ms = (time.perf_counter() - start) * 1000
                total_latency += elapsed_ms

                # Evaluate keyword coverage in top_k retrieved chunks
                hit_ranks = []
                retrieved_keywords_found = set()

                for rank, chunk in enumerate(retrieved, 1):
                    text = (chunk["text"] if isinstance(chunk, dict) else str(chunk)).lower()
                    found = {kw for kw in keywords if kw in text}
                    if found:
                        retrieved_keywords_found.update(found)
                        hit_ranks.append(rank)

                precision = len(retrieved_keywords_found) / max(1, len(retrieved))
                recall = len(retrieved_keywords_found) / max(1, len(keywords))
                mrr = (1.0 / hit_ranks[0]) if hit_ranks else 0.0

                precisions.append(precision)
                recalls.append(recall)
                mrrs.append(mrr)

            num_q = max(1, len(benchmark_set))
            results.append({
                "Strategy": strategy.upper(),
                "Avg Latency (ms)": round(total_latency / num_q, 2),
                "Precision@K": round(np.mean(precisions) * 100, 2),
                "Recall@K": round(np.mean(recalls) * 100, 2),
                "MRR": round(np.mean(mrrs), 3)
            })

        return pd.DataFrame(results)
