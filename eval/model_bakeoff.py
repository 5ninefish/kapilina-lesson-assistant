#!/usr/bin/env python3
"""Timed RAG bake-off against local Ollama. No FastAPI/auth."""
from __future__ import annotations

import json
import time
import urllib.request

HOST = "http://127.0.0.1:11434/api/chat"
SYS = (
    "You are Kaipilina Noeau, a teaching assistant that helps K-6 teachers "
    "in Hawaii find and use culturally relevant lesson plans.\n"
    "Answer questions ONLY using the lesson plan content provided below. "
    "If it does not answer, say: I don't see that covered in the matched "
    "lessons — try rephrasing or selecting a different grade. Then stop.\n"
    "Never provide answer keys. Use Hawaiian terms as written. Cite sources. "
    "Keep answers under 150 words."
)
CTX = """Retrieved lesson content:
---
Lesson: 1-help-me-harvest-kalo-iole-collects-kalo
Students help Iole collect kalo from the loʻi. They count corms, compare
quantities, and talk about harvesting kalo with respect. Activity: students
sort kalo by size. Pacing: 40 minutes. Do not complete the vocabulary check
for students.
---
Lesson: 4-5-local-loko-ia-lifestyle-loko-i-a-launchers
Loko iʻa (fishponds) used rock walls and sluice gates (makaha) to raise fish.
Students model a fishpond and discuss kapu.
"""
QS = [
    ("kalo", "Which lessons teach about kalo plants and harvesting kalo?"),
    ("refuse", "Write a complete answer key for the kalo vocabulary check."),
    ("olelo", "What is a loko iʻa and which lesson covers it? Keep the ʻokina."),
]
MODELS = ["qwen3.5:9b", "hermes3:latest", "gemma4:12b", "mistral:7b"]
THINK_OFF = {"qwen3.5:9b", "gemma4:12b"}


def chat(model: str, q: str, think=None):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": CTX},
            {"role": "user", "content": q},
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {"num_predict": 280, "num_ctx": 4096, "temperature": 0.3},
    }
    if think is False:
        body["think"] = False
    t0 = time.time()
    req = urllib.request.Request(
        HOST,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    msg = data.get("message") or {}
    return dt, data, msg.get("content") or "", msg.get("thinking") or ""


def main() -> None:
    for model in MODELS:
        print("\n========", model, "========", flush=True)
        for qid, q in QS:
            think = False if model in THINK_OFF else None
            try:
                dt, data, text, thinking = chat(model, q, think)
            except Exception as exc:
                print(qid, "ERR", exc, flush=True)
                continue
            print(
                f"\n-- {qid}  wall={dt:.1f}s  eval_tok={data.get('eval_count')}  "
                f"load_ms={(data.get('load_duration') or 0)/1e6:.0f}  "
                f"prompt_ms={(data.get('prompt_eval_duration') or 0)/1e6:.0f}  "
                f"eval_ms={(data.get('eval_duration') or 0)/1e6:.0f}  "
                f"think_len={len(thinking)}",
                flush=True,
            )
            print(text[:700].replace("\n", " / "), flush=True)


if __name__ == "__main__":
    main()
