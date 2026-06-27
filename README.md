# Kapilina Lesson Assistant

A RAG-based AI teaching assistant that helps educators find and use culturally grounded lesson plans. Teachers ask natural-language questions; the assistant retrieves the most relevant lesson content, streams a grounded answer, and cites its sources.

Built for a K–6 STEM curriculum program in Hawaiʻi. The architecture is fully generic — swap in any corpus of lesson documents.

## Architecture

```
Teacher's question
        │
        ▼
  multilingual-e5-large        ← sentence-transformers embedding model
  (query embedding)
        │
        ▼
  ChromaDB vector store        ← cosine similarity search, grade filter
  (top-K chunks, threshold 0.5)
        │
        ▼
  FastAPI /query endpoint      ← rate-limited (20 req/min), CORS-gated
        │
        ▼
  Ollama / Llama 3 8B          ← local LLM, system-prompted to stay
  (streaming SSE)                 grounded in retrieved content only
        │
        ▼
  Token stream + sources + glossary hits → browser
```

**Key design choices:**
- **Local-first**: embeddings and LLM run entirely on-device (no OpenAI or cloud API required)
- **Grounded**: the system prompt forbids answering outside retrieved content; it cites lessons by name
- **Idempotent ingest**: SHA-256 chunk IDs mean re-running ingest never duplicates data
- **Multilingual embeddings**: `intfloat/multilingual-e5-large` handles Hawaiian diacritics (ʻokina, kahākō) correctly
- **Injection-safe**: conversation history is passed as structured Ollama `messages`, not concatenated into a user prompt

## Stack

| Layer | Tech |
|---|---|
| Embedding | `intfloat/multilingual-e5-large` via sentence-transformers |
| Vector store | ChromaDB (persistent, cosine similarity) |
| LLM | Llama 3 8B via Ollama (local) |
| API | FastAPI + SSE streaming |
| Rate limiting | slowapi (20 req/min per IP) |
| Document parsing | python-docx, python-pptx, pdfplumber |

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Llama 3 pulled: `ollama pull llama3:8b`

### Install

```bash
git clone https://github.com/5ninefish/kapilina-lesson-assistant
cd kapilina-lesson-assistant
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn chromadb sentence-transformers slowapi httpx \
            python-docx python-pptx pdfplumber
```

### Add your lesson corpus

Place your lesson files (`.docx`, `.pptx`, or `.pdf`) in `data/lessons/`.

Name them with a lesson prefix: `L1_intro.docx`, `L2_water.pdf`, etc.

Create `data/grade_map.json` to map lesson IDs to metadata:

```json
{
  "L1": { "title": "Introduction to Soil Science", "grades": ["3", "4"], "filename": "L1_intro.docx" },
  "L2": { "title": "Water Cycle", "grades": ["4", "5"], "filename": "L2_water.pdf" }
}
```

Optionally create `data/glossary.json` for domain-specific terms:

```json
{
  "photosynthesis": "The process by which plants convert sunlight into food.",
  "watershed": "An area of land that drains into a common body of water."
}
```

### Ingest

```bash
python scripts/ingest.py
```

This embeds and stores all lesson chunks in ChromaDB. Idempotent — safe to re-run.

### Run

```bash
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health`
Query: `POST /query` with `{ "question": "...", "grade": "4" }`

### Frontend

A minimal HTML/JS frontend lives in `public/`. Serve it from any static host or open `index.html` directly.

## API

### `POST /query`

```json
{
  "question": "What lessons cover the water cycle?",
  "grade": "4",
  "messages": []
}
```

Streams `text/event-stream`. Each event is one of:

```json
{ "token": "..." }
{ "done": true, "sources": [...], "glossary": [...] }
```

### `GET /lesson/{id}`

Returns the full extracted text + HTML of a lesson file by ID (e.g. `/lesson/L3`).

## Adapting to a new domain

1. Replace `data/lessons/` with your corpus
2. Update `data/grade_map.json` (or remove grade filtering from the query endpoint)
3. Update the `SYSTEM_PROMPT` in `server.py` to match your domain
4. Optionally replace Llama 3 with any Ollama-compatible model

The embedding model, ChromaDB schema, and chunking logic need no changes.
