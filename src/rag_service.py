from typing import List, Tuple
import ollama
from src.vector_store import SimpleVectorStore
from src.ingestion import DocumentChunk

class LocalRAGService:
    """Retrieval-Augmented Generation service powered by local Ollama models."""

    def __init__(self, vector_store: SimpleVectorStore, model_name: str = 'qwen2.5:3b'):
        self.vector_store = vector_store
        self.model_name = model_name

    def _build_context(self, search_result: List[Tuple[DocumentChunk, float]]) -> str:
        """Format retrieved document chunks into a structured context string."""
        context_parts = []
        for i, (chunk, score) in enumerate(search_result, start=1):
            source_info = f'[Page: {chunk.page_start}-{chunk.page_end}, Score: {score:.2f}]'
            context_parts.append(f'--- Snippet {i} {source_info} ---\n{chunk.text}\n')
        return '\n'.join(context_parts)

    def answer_question(self, query: str, top_k: int = 2) -> str:
        """Retrieve relevant context and generate an answer using the local LLM."""
        print(f'\n[RAG] Searching for relevant chunks matching: \'{query}\'...')
        search_result = self.vector_store.search(query, top_k=top_k)
        context_str = self._build_context(search_result)

        system_instruction = (
            'You are an AI assistant specialized in answering questions based on document context.\n'
            'The context contains markdown tables of countries, capitals, and populations. '
            'Look carefully through the table rows to find the requested information.\n'
            'Your task: Answer the user\'s question ONLY using facts from the CONTEXT provided below.\n'
            'Rules:\n'
            '1. If the answer cannot be found in the context, truthfully state: \'Information not found in the document.\'\n'
            '2. Do not fabricate or assume any facts outside the provided context.\n'
            '3. Cite page numbers from the context if available.\n'
            '4. Keep the answer concise, accurate, and clear.'
        )

        user_content = f'DOCUMENT CONTEXT:\n{context_str}\n\nUSER QUESTION:\n{query}'
        print(f'[RAG] Generating response with local model ({self.model_name})...')

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_content}
            ]
        )

        return response['message']['content']