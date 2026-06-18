from langchain_groq import ChatGroq

class QueryRewriter:

    def __init__(self):

        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

    def rewrite(self, query):

           prompt = f"""
Rewrite the following query
for better document retrieval.

Query:
{query}

Return only rewritten query.
"""
           
           response = self.llm.invoke(prompt)

           return response.content.strip()