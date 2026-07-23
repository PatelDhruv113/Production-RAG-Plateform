import sqlite3
import logging
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class SQLiteManager:
    """
    SQLite Relational Storage Manager for indexing document metadata,
    text chunks, and automated evaluation metrics.
    """

    def __init__(self, db_path: str = "rag.db") -> None:
        self.db_path = db_path
        self.create_tables()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def create_tables(self) -> None:
        """Creates required relational schema if tables do not exist."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            page_number INTEGER,
            source_file TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            faithfulness REAL,
            answer_relevancy REAL,
            judge_score REAL,
            hallucinated INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()

    def save_document(self, filename: str) -> Optional[int]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO documents (filename) VALUES (?)", (filename,))
            conn.commit()
            document_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            document_id = row[0] if row else None
        finally:
            conn.close()
        return document_id

    def get_document_by_filename(self, filename: str) -> Optional[int]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE filename = ? LIMIT 1", (filename,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_all_documents(self) -> List[Tuple[Any, ...]]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def save_chunk(self, document_id: int, chunk_text: str, page_number: int, source_file: str) -> int:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (document_id, chunk_text, page_number, source_file)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, chunk_text, page_number, source_file)
        )
        conn.commit()
        chunk_id = cursor.lastrowid
        conn.close()
        return chunk_id

    def get_all_chunks(self) -> List[Tuple[int, str, str, int]]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT id, chunk_text, source_file, page_number
                FROM chunks
                ORDER BY id
            """)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Database operational warning: {e}")
            self.create_tables()
            rows = []
        finally:
            conn.close()
        return rows

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Tuple[str]]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_text FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    def get_chunk_text(self, chunk_id: int) -> Optional[str]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_text FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def save_evaluation(
        self,
        query: str,
        answer: str,
        faithfulness: float,
        answer_relevancy: float,
        judge_score: float,
        hallucinated: int = 0
    ) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO evaluations (query, answer, faithfulness, answer_relevancy, judge_score, hallucinated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (query, answer, faithfulness, answer_relevancy, judge_score, hallucinated)
        )
        conn.commit()
        conn.close()

    def get_evaluations(self) -> List[Tuple[Any, ...]]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query, faithfulness, answer_relevancy, judge_score, hallucinated, created_at
            FROM evaluations
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def reset_document_store(self) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM documents")
        conn.commit()
        conn.close()


