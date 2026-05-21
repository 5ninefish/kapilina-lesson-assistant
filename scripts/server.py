#!/usr/bin/env python3
"""
Kaipilina Noeau — RAG API server
FastAPI + streaming SSE + conversation history + Hawaiian glossary tooltips.
"""

import json
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import chromadb
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

BASE = Path.home() / "kapilinanoeau"
CHROMA_DIR = BASE / "data/chroma"
GLOSSARY_PATH = BASE / "data/glossary.json"
GRADE_MAP_PATH = BASE / "data/grade_map.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"
TOP_K = 5
SIM_THRESHOLD = 0.65
MAX_CONTEXT_CHARS = 3000 * 4
MAX_STREAM_TOKENS = 2048

SYSTEM_PROMPT = """\
You are Kaipilina Noeau, a teaching assistant that helps K–6 teachers in Hawaiʻi find and use culturally relevant lesson plans.

Answer questions ONLY using the lesson plan content provided below. Do not add information from outside these lessons. If the provided content does not answer the question, say: "I don't see that covered in the matched lessons — try rephrasing or selecting a different grade."

Treat Hawaiian cultural content with care. Use Hawaiian terms as written (with ʻokina and kahākō intact). Do not translate or interpret Hawaiian terms beyond what the lesson provides.

Always end your response by citing the lesson(s) you drew from, e.g.: "Source: L3 Kiʻi Pōhaku (Grade 4–5)."\
"""

state: dict = {}
limiter = Limiter(key_func=get_remote_address)

GradeValue = Literal["all", "K", "1", "2", "3", "4", "5", "6"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    state["model"] = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    state["collection"] = client.get_or_create_collection(
        name="lessons",
        metadata={"hnsw:space": "cosine"},
    )

    state["glossary"] = json.loads(GLOSSARY_PATH.read_text()) if GLOSSARY_PATH.exists() else {}
    state["grade_map"] = json.loads(GRADE_MAP_PATH.read_text()) if GRADE_MAP_PATH.exists() else {}

    print("Pre-warming Ollama...")
    async with httpx.AsyncClient(timeout=30) as c:
        try:
            await c.post(f"{OLLAMA_HOST}/api/generate",
                         json={"model": OLLAMA_MODEL, "prompt": "hi", "stream": False})
        except Exception as e:
            print(f"Ollama pre-warm failed (non-fatal): {e}")

    print("Ready.")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kapilinanoeau.org", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Message(BaseModel):
    role: str
    content: Annotated[str, Field(max_length=8000)]


class QueryRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=500)]
    grade: GradeValue = "all"
    subject: str | None = None
    messages: Annotated[list[Message], Field(max_length=50)] = []


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=3) as c:
        try:
            r = await c.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(503, f"Ollama unreachable: {e}")
    count = state.get("collection", None)
    return {"status": "ok", "chunks": count.count() if count else 0}


def match_glossary(text: str) -> list[dict]:
    norm = unicodedata.normalize("NFC", text.lower())
    return [
        {"term": term, "definition": defn}
        for term, defn in state["glossary"].items()
        if unicodedata.normalize("NFC", term.lower()) in norm
    ]


async def sse_stream(request: QueryRequest):
    model: SentenceTransformer = state["model"]
    collection = state["collection"]

    query_emb = model.encode(f"query: {request.question}", normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K * 3,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    def grade_ok(m):
        return request.grade == "all" or request.grade in m.get("grades", "").split(",")

    filtered = [(d, m) for d, m, dist in zip(docs, metas, dists)
                if (1 - dist) >= SIM_THRESHOLD and grade_ok(m)]

    if not filtered:
        msg = "I don't see that covered in the matched lessons — try rephrasing or selecting a different grade."
        yield f"data: {json.dumps({'token': msg})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'glossary': []})}\n\n"
        return

    per_doc = MAX_CONTEXT_CHARS // len(filtered)
    context = "\n\n---\n\n".join(d[:per_doc] for d, _ in filtered)

    sources, seen = [], set()
    for _, m in filtered:
        fn = m.get("filename", "")
        if fn not in seen:
            seen.add(fn)
            sources.append({"title": m.get("title", fn), "lesson": m.get("lesson", ""),
                            "grades": m.get("grades", ""), "filename": fn})

    # Build structured messages for /api/chat — Ollama enforces role boundaries,
    # preventing prompt injection via user-controlled question or history content.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Retrieved lesson content:\n{context}"},
    ]
    for m in request.messages[-8:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.question})

    full_tokens: list[str] = []
    token_count = 0
    async with httpx.AsyncClient(timeout=60) as c:
        async with c.stream("POST", f"{OLLAMA_HOST}/api/chat",
                            json={"model": OLLAMA_MODEL, "messages": messages, "stream": True}) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_tokens.append(token)
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    if token_count >= MAX_STREAM_TOKENS:
                        break
                if chunk.get("done"):
                    break

    glossary_hits = match_glossary("".join(full_tokens))
    yield f"data: {json.dumps({'done': True, 'sources': sources, 'glossary': glossary_hits})}\n\n"


@app.post("/query")
@limiter.limit("20/minute")
async def query(body: QueryRequest, request: Request):
    return StreamingResponse(
        sse_stream(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
