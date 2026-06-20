from src.pipeline.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()

query = input("Enter Query: ")

# answer = pipeline.run(query)

# print("\n")
# print(answer)


result = pipeline.run(query)

print()

print("Answer:")
print(result["answer"])

print()

print("Confidence:")
print(result["confidence"])

print()

print("Sources:")

for source in result["sources"]:

    print(source)

print()

print("Evaluation:")
print(
    result["evaluation"]
)