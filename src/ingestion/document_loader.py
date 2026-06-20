from pathlib import Path
from pypdf import PdfReader

class DocumentLoader:

    def load_pdf(self, file_path: str) -> str:

        pages = self.load_pdf_pages(file_path)

        return "\n".join(
            page["text"]
            for page in pages
        )

    def load_pdf_pages(self, file_path: str):

        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if page_text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": page_text
                    }
                )

        return pages
    
    def load_folder(self, folder_path: str):

        documents = []

        for pdf_file in Path(folder_path).rglob("*.pdf"):

            pages = self.load_pdf_pages(
                str(pdf_file)
            )

            documents.append(
                {
                    "filename": pdf_file.name,
                    "filepath": str(pdf_file),
                    "text": "\n".join(
                        page["text"]
                        for page in pages
                    ),
                    "pages": pages
                }
            )
        return documents
