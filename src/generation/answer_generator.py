import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


class AnswerGenerator:

    def __init__(self):
        pass

    @property
    def llm(self) -> ChatGroq:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API Key is missing. Please set it in the Streamlit sidebar.")
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key
        )

    def generate(self, prompt):

        response = self.llm.invoke(prompt)

        return response.content