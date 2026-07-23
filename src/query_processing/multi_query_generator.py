import os
from langchain_groq import ChatGroq

class MultiQueryGenerator:

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

    def generate(self, query):

         prompt = f"""
Generate 3 alternative search queries.

Original Query:
{query}

Return one query per line.
"""
         response = self.llm.invoke(prompt)

         queries = [
             q.strip()
             for q in response.content.split("\n")
             if q.strip()
         ]

         return queries