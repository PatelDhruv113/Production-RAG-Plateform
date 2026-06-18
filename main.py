from src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

query = input("Enter Query: ")

answer = pipeline.run(query)

print("\n")

print(answer)