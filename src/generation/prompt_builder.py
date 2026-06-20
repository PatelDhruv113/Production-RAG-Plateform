class PromptBuilder:

    def build(self,query,context):

        # print("context:", context)
        prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

If answer is not found,
say:
"I could not find the answer in the documents."

Context:

{context}

Question:

{query}

Answer:
"""

        return prompt