from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


class AnswerGenerator:

    def __init__(self):

        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    def generate(self, prompt):

        response = self.llm.invoke(prompt)

        return response.content