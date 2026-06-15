import sqlite3

class SQLiteManager:

    def __init__(self, db_path="rag.db"):
        self.db_path = db_path

    def connect(self):
        return sqlite3.connect(self.db_path)
    
    def create_tables(self):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          filename TEXT NOT NULL,
          category TEXT NOT NULL,
          upload_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           document_id INTEGER,
           chunk_text TEXT,
           page_number INTEGER,
           category TEXT,
           FOREIGN KEY(document_id)
           REFERENCES documents(id)
        )
        """)

        conn.commit()
        conn.close()

        print("Tables created successfully")


    def save_document(self, filename: str, category: str):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO documents
            (filename, category)
            VALUES (?, ?)
            """,
            (filename, category)
        )

        conn.commit()

        document_id = cursor.lastrowid

        conn.close()

        return document_id
    
    
    def get_all_documents(self):

        conn = self.connect()
        cursor = conn.cursor()
    
        cursor.execute(
            "SELECT * FROM documents"
        )
    
        rows = cursor.fetchall()
    
        conn.close()
    
        return rows
    
    def save_chunk(self, document_id, chunk_text, page_number, category):

        conn = self.connect()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chunks
            (
                document_id,
                chunk_text,
                page_number,
                category
            )
            VALUES(?, ?, ? ,?)
            """,
            (
                document_id,
                chunk_text,
                page_number,
                category
            )
        )

        conn.commit()
        conn.close()