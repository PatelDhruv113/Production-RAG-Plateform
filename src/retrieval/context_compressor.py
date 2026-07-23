import re
import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Context Compressor for pruning non-relevant sentences from retrieved document chunks,
    reducing token noise and cost for LLM generation.
    """

    def __init__(self) -> None:
        self.stop_words = {"what", "is", "are", "the", "a", "an", "of", "in", "to", "and", "or", "for", "with", "on", "at"}

    def compress(
        self,
        query: str,
        chunks: List[Union[Dict[str, Any], str]]
    ) -> List[Union[Dict[str, Any], str]]:
        """
        Compresses chunks by selecting sentences containing query keywords.

        Args:
            query (str): Input query text.
            chunks (List[Union[Dict[str, Any], str]]): Candidate document chunks.

        Returns:
            List[Union[Dict[str, Any], str]]: Compressed chunks with pruned sentences.
        """
        query_words = set(re.findall(r"\w+", query.lower())) - self.stop_words
        if not query_words:
            return chunks

        compressed_chunks: List[Union[Dict[str, Any], str]] = []

        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
            sentences = [s.strip() for s in text.split(".") if s.strip()]

            relevant_sentences = []
            for sentence in sentences:
                sentence_words = set(re.findall(r"\w+", sentence.lower()))
                if query_words.intersection(sentence_words):
                    relevant_sentences.append(sentence)

            compressed_text = ". ".join(relevant_sentences) + "." if relevant_sentences else text

            if isinstance(chunk, dict):
                chunk_copy = chunk.copy()
                chunk_copy["text"] = compressed_text
                compressed_chunks.append(chunk_copy)
            else:
                compressed_chunks.append(compressed_text)

        return compressed_chunks
