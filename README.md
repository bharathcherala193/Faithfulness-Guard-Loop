# Faithfulness-Guard RAG Agent

A simple RAG chatbot that checks its own answers before showing them to you.

Most RAG apps answer once and stop. This one scores its answer for **faithfulness** (is it actually backed by the document, or did the model make something up?) using RAGAS. If the score is too low, it doesn't fetch more chunks — it just re-answers the *same* question from the *same* context, but with a much stricter instruction to stick to only what's written.

```
Your Question ──► Retrieve ──► Answer ──► Score Faithfulness ──► ≥ 0.7? ──► ✅ Done
                                              │
                                        < 0.7?│
                                              ▼
                                   Re-answer with a stricter
                                   "only use exact context" prompt
```

## Why this loop (and not just "retrieve more")

- **Answer relevancy** asks: did the model answer the question?
- **Faithfulness** asks: is what it said actually supported by the document?

A model can be relevant and still hallucinate a detail that sounds right but isn't in the source. This project isolates that specific failure and fixes it by tightening the prompt, not by changing retrieval.

## Features

| Feature | Description |
|---|---|
| Faithfulness Guard | Re-answers automatically if the response isn't grounded in the retrieved context |
| Live Self-Evaluation | Scores every response with RAGAS before showing it |
| Dual-LLM Architecture | One model answers, a separate model judges |
| Multi-turn Memory | Full conversation history via LangGraph checkpointing |
| Score Tracking | See faithfulness score before/after the stricter retry |
| Fully Local | Runs entirely on Ollama — no API keys, no cloud |

## Tech Stack

| Role | Tool |
|---|---|
| Answering | `deepseek-r1:8b` (via Ollama) |
| Judging | `gemma3:12b` (via Ollama) |
| Embeddings | `nomic-embed-text` (via Ollama) |
| Vector Store | Chroma |
| Graph Engine | LangGraph (StateGraph + Memory) |
| Evaluation | RAGAS (faithfulness + answer relevancy) |
| UI | Streamlit |
| Doc Loader | LangChain PyPDFLoader |

## Quick Start

### 1. Pull the models
```bash
ollama pull deepseek-r1:8b
ollama pull gemma3:12b
ollama pull nomic-embed-text
```

### 2. Clone and install
```bash
git clone https://github.com/your-username/faithfulness-guard-rag
cd faithfulness-guard-rag
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run faithfulness_guard_rag.py
```

## How to Use

1. Upload a PDF
2. Ask a question in the chat
3. If the first answer scores low on faithfulness, watch it automatically re-answer more strictly
4. Check the sidebar for the faithfulness score before and after

## Example Run

```
Attempt 1  │ strict=False │ Faith: 0.42  │ Rel: 0.88  │ → etry (not grounded enough)
Attempt 2  │ strict=True  │ Faith: 0.91  │ Rel: 0.85  │ → ✅ accepted

Faithfulness improvement: 0.42 → 0.91
```

## Roadmap

- [ ] Persistent Chroma store across sessions
- [ ] Multi-document upload support
- [ ] Adjustable faithfulness threshold via UI slider
- [ ] Show which specific sentence(s) failed the faithfulness check
- [ ] OpenAI / Anthropic provider support as a drop-in swap
