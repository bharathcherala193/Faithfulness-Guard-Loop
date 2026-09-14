# 🛡️ Faithfulness-Guard RAG Agent

A simple RAG (Retrieval-Augmented Generation) app that answers questions about
an uploaded PDF using only the content of that PDF — not the model's general
knowledge. Runs entirely locally using Ollama, so nothing leaves your machine.

## What it does

1. Upload a PDF — it's split into overlapping chunks and stored in a local
   vector database (Chroma).
2. Ask a question — it rewrites vague follow-ups (e.g. "explain about it")
   into a clear standalone search query using recent chat history.
3. It retrieves the most relevant chunks and answers using only that text.
4. (Optional) A quick groundedness check retries the answer once if it isn't
   backed by the retrieved text.
5. (Optional) A deeper RAGAS evaluation pass gives numeric
   faithfulness/relevancy scores.

## Tech stack

| Component     | Tool                              |
|---------------|------------------------------------|
| Answering LLM | `llama3.1:8b` (via Ollama)        |
| Judge LLM     | `gemma3:12b` (via Ollama)         |
| Embeddings    | `nomic-embed-text` (via Ollama)   |
| Vector store  | Chroma                            |
| Orchestration | LangGraph                         |
| Evaluation    | RAGAS (faithfulness, relevancy)   |
| UI            | Streamlit                         |
| PDF loading   | LangChain `PyPDFLoader`           |

## Setup

**1. Pull the models:**
```bash
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama pull nomic-embed-text
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run it:**
```bash
streamlit run faithfulness_guard_rag.py
```

## Usage

- Upload a PDF in the browser tab that opens.
- Ask a question in the chat box.
- Optional sidebar toggles:
  - **Enable grounding retry check** — adds a fast yes/no check that
    retries once with a stricter prompt if the answer isn't grounded.
  - **Run deep RAGAS check too** — adds full numeric faithfulness/relevancy
    scoring (slower, uses a second, larger judge model).
- Expand **"Retrieved context (debug)"** in the sidebar to see exactly which
  chunks were retrieved and what search query was actually used.

## Notes

- Response speed depends heavily on your hardware — a GPU makes a big
  difference over CPU-only inference.
- Both accuracy-check toggles are off by default so you get one direct,
  fast answer; turn them on when you specifically want the extra checks.
