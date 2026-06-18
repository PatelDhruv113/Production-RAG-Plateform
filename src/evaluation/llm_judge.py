from langchain_groq import ChatGroq

class LLMJudge:

    def __init__(self):
        
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

    def judge(self, question, answer, context):

         prompt = f"""
You are evaluating a RAG answer.

Question:
{question}

Context:
{context}

Answer:
{answer}

Score from 1-10.

Return only score.
"""
         response = self.llm.invoke(prompt)

         return response.content