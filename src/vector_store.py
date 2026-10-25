from typing import List, Tuple
import numpy as np
from src.ingestion import DocumentChunk
from src.embeddings import BaseEmbeddingService

class SimpleVectorStore:
    """In-memory vector store utilizing cosine similarity for document retrieval."""

    def __init__(self, embedding_service: BaseEmbeddingService):
        self.embedding_service = embedding_service
        self.chunks: List[DocumentChunk] = []
        self.vectors: List[np.ndarray] = []

    def add_chunk(self, chunks: List[DocumentChunk]):
        """Embed and index document chunks into memory."""
        print(f'[VectorStore] Generating embeddings for {len(chunks)} chunk(s)...')
        texts = [chunk.text for chunk in chunks]

        new_vectors = self.embedding_service.embed_batch(texts)
        self.chunks.extend(chunks)
        self.vectors.extend(new_vectors)

        print(f'[VectorStore] Successfully indexed {len(self.chunks)} total chunk(s).')

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve the top-k most relevant chunks using cosine similarity."""
        if not self.vectors:
            raise ValueError('Vector store is empty. Please add document chunks before searching.')

        query_vector = self.embedding_service.embed_text(query)
        scores = []
        q_norm = np.linalg.norm(query_vector)

        # Compute cosine similarity against all indexed vectors
        for doc_vector in self.vectors:
            d_norm = np.linalg.norm(doc_vector)
            if d_norm == 0 or q_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vector, doc_vector) / (d_norm * q_norm)
            scores.append(similarity)

        # Sort by similarity score in descending order
        sorted_indices = np.argsort(scores)[::-1]
        results = []
        for index in sorted_indices[:top_k]:
            results.append((self.chunks[index], float(scores[index])))

        return results