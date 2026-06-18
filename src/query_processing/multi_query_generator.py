from langchain_groq import ChatGroq

class MultiQueryGenerator:

    def __init__(self):

        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0
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