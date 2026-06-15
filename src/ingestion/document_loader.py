from pathlib import Path
from pypdf import PdfReader

class DocumentLoader:

    def load_pdf(self, file_path: str) -> str:

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text
    
    def load_folder(self, folder_path: str):

        documents = []

        for pdf_file in Path(folder_path).rglob("*.pdf"):

            text = self.load_pdf(
                str(pdf_file)
            )

            documents.append(
                {
                    "filename": pdf_file.name,
                    "filepath": str(pdf_file),
                    "text": text
                }
            )
        return documents