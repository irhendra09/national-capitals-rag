# Mini-RAG: Local QA on World National Capitals 🌍

A lightweight, privacy-first **Retrieval-Augmented Generation (RAG)** system built in Python. This project allows users to query information from a PDF document (*National Capitals of the World*) completely offline using local embedding models and a local Large Language Model (LLM).

---

## 🚀 Key Features

- **100% Local & Offline**: Powered by [Ollama](https://ollama.com/) (`qwen2.5:3b`) and [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`). No external API keys or cloud services required.
- **Semantic PDF Ingestion**: Extracts structured document chunks with page references using `pymupdf4llm`.
- **In-Memory Vector Store**: Fast cosine similarity search implemented cleanly with `NumPy`.
- **Hallucination Prevention**: Uses strict system prompt guardrails requiring the LLM to only answer based on retrieved context and cite page numbers.
- **Interactive CLI**: Simple and responsive command-line interface for real-time querying.

---

## 🏗️ Architecture & Pipeline

```text
               ┌────────────────────────┐
               │  national-capitals.pdf │
               └───────────┬────────────┘
                           │ (pymupdf4llm)
                           ▼
                [ Document Ingestion ]
               (Chunking & Metadata)
                           │
                           ▼
                [ FastEmbed Service ]
               (BAAI/bge-small-en-v1.5)
                           │
                           ▼
               ┌────────────────────────┐
               │ In-Memory Vector Store │
               └───────────▲────────────┘
                           │
  [User Query] ──> [Query Embedding] ──> [Cosine Similarity Search]
                                                   │ (Top-K Chunks)
                                                   ▼
                                         [ Context Assembly ]
                                                   │
                                                   ▼
                                         [ Ollama Local LLM ]
                                            (qwen2.5:3b)
                                                   │
                                                   ▼
                                            [ Final Answer ]
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM Runner:** [Ollama](https://ollama.com/) (Model: `qwen2.5:3b`)
- **Embedding:** `fastembed` (`BAAI/bge-small-en-v1.5`)
- **PDF Processing:** `pymupdf4llm`
- **Vector Operations:** `numpy`

---

## 📂 Project Structure

```text
national-capitals-rag/
├── data/
│   └── national-capitals.pdf    # Source PDF document
├── src/
│   ├── __init__.py
│   ├── embeddings.py            # Embedding abstraction & FastEmbed service
│   ├── ingestion.py             # PDF reading & chunking pipeline
│   ├── rag_service.py           # RAG logic & Ollama prompt engineering
│   └── vector_store.py          # In-memory vector store with Cosine Similarity
├── main.py                      # Application entry point & interactive CLI
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## ⚡ Getting Started

### 1. Prerequisites

1. **Python 3.10+** installed on your system.
2. **[Ollama](https://ollama.com/)** installed and running.
   Pull the model used by this project:
   ```bash
   ollama pull qwen2.5:3b
   ```

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/national-capitals-rag.git
cd national-capitals-rag
```

### 3. Create a Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

Run the main application:

```bash
python main.py
```

### Example Interaction

```text
============================================================
 MINI-RAG SYSTEM: QA ON WORLD NATIONAL CAPITALS (PDF)
============================================================
[Ingestion] Processing PDF: national-capitals.pdf
[Ingestion] Successfully extracted 6 chunk(s).
[Embedding] Loading model: BAAI/bge-small-en-v1.5...
[Embedding] Model loaded successfully.
[VectorStore] Generating embeddings for 6 chunk(s)...
[VectorStore] Successfully indexed 6 total chunk(s).

[READY] RAG system is ready! Ask your question.
(Type 'exit' or 'q' to quit)

Enter your question: > What is the capital of Australia and its population?

[RAG] Searching for relevant chunks matching: 'What is the capital of Australia and its population?'...
[RAG] Generating response with local model (qwen2.5:3b)...

--- Answer from LLM ---
Answer: The capital of Australia is Canberra and its population is 431,380, as of 2020..
------------------------------------------------------------
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and adapt it for your own projects.
