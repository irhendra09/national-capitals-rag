from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import pymupdf4llm

@dataclass
class DocumentChunk:
    """Data transfer object representing a chunk extracted from a document."""
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    content_types: List[str]
    metadata: Dict[str, Any]

class DocumentIngestionService:
    """Service to process and chunk PDF documents for RAG indexing."""

    def __init__(self, max_tokens: int = 400):
        self.max_tokens = max_tokens

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Read a PDF document and split it into structured chunks."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f'File not found: {pdf_path}')

        print(f'[Ingestion] Processing PDF: {path.name}')

        # Extract semantic chunks using pymupdf4llm
        raw_chunks = pymupdf4llm.to_chunks(str(path), max_tokens=self.max_tokens)
        chunks: List[DocumentChunk] = []

        for c in raw_chunks:
            chunk_dto = DocumentChunk(
                chunk_id=c.id,
                text=c.text,
                page_start=c.metadata.page_start,
                page_end=c.metadata.page_end,
                content_types=c.metadata.types,
                metadata={
                    'source_file': path.name,
                    'content_hash': getattr(c, 'content_hash', None),
                }
            )
            chunks.append(chunk_dto)

        print(f'[Ingestion] Successfully extracted {len(chunks)} chunk(s).')
        return chunks