from abc import ABC, abstractmethod
from typing import List
import numpy as np
from fastembed import TextEmbedding

class BaseEmbeddingService(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding vector for a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embedding vectors for a batch of text strings."""
        pass

class LocalEmbeddingService(BaseEmbeddingService):
    """Local text embedding service powered by FastEmbed."""

    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        print(f'[Embedding] Loading model: {model_name}...')
        self.model = TextEmbedding(model_name=model_name)
        print('[Embedding] Model loaded successfully.')

    def embed_text(self, text: str) -> np.ndarray:
        vector = list(self.model.embed([text]))[0]
        return np.array(vector, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        vectors = list(self.model.embed(texts))
        return [
            np.array(v, dtype=np.float32) for v in vectors
        ]