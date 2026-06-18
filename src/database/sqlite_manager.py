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
           source_file TEXT,
           FOREIGN KEY(document_id)
           REFERENCES documents(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           query TEXT,
           faithfulness REAL,
           answer_relevancy REAL,
           judge_score REAL,
           hallucinated INTEGER,
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    def save_chunk(self, document_id, chunk_text, page_number, category, source_file):

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
                category,
                source_file
            )
        )

        conn.commit()

        chunk_id = cursor.lastrowid

        conn.close()

        return  chunk_id
    
    def get_all_chunks(self):

        conn = self.connect()
        cursor = conn.cursor()


        cursor.execute(
            """
            SELECT 
                id, 
                chunk_text 
            FROM chunks
           """
        )

        rows = cursor.fetchall()
        conn.close()

        return rows
    
    def get_chunk_by_id(self, chunk_id):

        conn = self.connect()

        cursor = conn.connect()

        cursor.execute(
            """
            SELECT chunk_text
            FROM chunks 
            WHERE id=?
            """,
            (chunk_id)
        )

        row = cursor.fetchone()

        conn.close()

        return row
    
    def get_chunk_text(self, chunk_id):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_text
            FROM chunks
            WHERE id=?
            """,
            (chunk_id)
        )

        row = cursor.fetchone()

        conn.close()

        if row:
            return row[0]
        
        return None
    
    def get_chunk_with_source(self, chunk_id):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                chunk_text,
                source_file
            FROM chunks
            WHERE id=?
            """,
            (chunk_id)
        )

        row = cursor.fetchone()

        conn.close()

        return row
    
    def save_evaluation(self, query, faithfulness,  answer_relevancy, judge_score, hallucinated):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
        """
        INSERT INTO evaluations(
            query,
            faithfulness,
            answer_relevancy,
            judge_score,
            hallucinated
        )
        VALUES (?, ?, ?, ?, ?)
        """,
          (
              query,
              faithfulness,
              answer_relevancy,
              judge_score,
              hallucinated
          )
        )
     
        conn.commit()
        conn.close()
        