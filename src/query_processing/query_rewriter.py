import os
from langchain_groq import ChatGroq

class QueryRewriter:

    def __init__(self):
        pass

    @property
    def llm(self) -> ChatGroq:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API Key is missing. Please set it in the Streamlit sidebar.")
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=api_key
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