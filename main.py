# Main entry point for the Mini-RAG application
from src.ingestion import DocumentIngestionService
from src.embeddings import LocalEmbeddingService
from src.vector_store import SimpleVectorStore
from src.rag_service import LocalRAGService

def main():
    print('=' * 60)
    print(' MINI-RAG SYSTEM: QA ON WORLD NATIONAL CAPITALS (PDF)')
    print('=' * 60)

    # Path to the source document
    pdf_file = 'data/national-capitals.pdf'

    # 1. Ingest and chunk the document
    ingestion = DocumentIngestionService(max_tokens=400)
    chunks = ingestion.process_pdf(pdf_file)

    # 2. Initialize local embedding service
    embedding_service = LocalEmbeddingService()

    # 3. Store and index chunks in vector store
    vector_store = SimpleVectorStore(embedding_service)
    vector_store.add_chunk(chunks)

    # 4. Initialize RAG service with local LLM
    rag = LocalRAGService(vector_store, model_name='qwen2.5:3b')

    print('\n[READY] RAG system is ready! Ask your question.')
    print('(Type \'exit\' or \'q\' to quit)\n')

    # 5. Interactive query loop
    while True:
        try:
            question = input('Enter your question: > ')
            if question.lower().strip() in ['exit', 'q']:
                print('\nExiting system. Goodbye!')
                break
            if not question.strip():
                continue

            answer = rag.answer_question(question, top_k=2)
            print('\n--- Answer from LLM ---')
            print(answer)
            print('-' * 60 + '\n')

        except KeyboardInterrupt:
            print('\nSession interrupted. Goodbye!')
            break
        except Exception as e:
            print(f'[Error] An unexpected error occurred: {e}\n')

if __name__ == '__main__':
    main()

