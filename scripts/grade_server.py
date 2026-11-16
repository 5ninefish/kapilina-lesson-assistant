#!/usr/bin/env python3
"""Separate Wrap-apply grader. Does not import or change the KPN lesson API."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = Path(os.environ.get("GRADE_RUBRIC") or (ROOT / "grader" / "wrap_apply_rubric.txt"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("GRADE_MODEL", "qwen3.5:9b")
PORT = int(os.environ.get("GRADE_PORT", "8011"))
RATE_PER_MIN = int(os.environ.get("GRADE_RATE_PER_MIN", "30"))
OLLAMA_TIMEOUT_S = float(os.environ.get("GRADE_OLLAMA_TIMEOUT_S", "4.5"))

HOW_MARKERS = (
    "learn",
    "practice",
    "skill",
    "eager",
    "train",
    "study",
    "already",
    "related",
    "improv",
    "shadow",
    "volunteer",
    "course",
    "experience",
)
STANCE_MARKERS = (
    "yes",
    "no",
    "would",
    "will",
    "won't",
    "wouldn",
    "going",
    "gonna",
    "wait",
    "apply",
    "ready",
)
JUNK_WORDS = {
    "apply",
    "learn",
    "qualif",
    "qualified",
    "qualification",
    "qualifications",
    "and",
    "the",
    "a",
    "to",
    "i",
    "would",
    "will",
    "not",
    "even",
    "if",
    "am",
    "is",
    "for",
    "of",
    "my",
}

_lock = threading.Lock()
_hits: dict[str, list[float]] = {}
_rubric_lines: list[str] = []


def load_secret() -> str:
    secret = os.environ.get("GRADE_SECRET") or ""
    path = os.environ.get("GRADE_SECRET_FILE") or ""
    if not secret and path:
        secret = Path(path).read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if len(secret) < 16:
        raise SystemExit("GRADE_SECRET missing")
    return secret


def load_rubric() -> str:
    global _rubric_lines
    text = RUBRIC_PATH.read_text(encoding="utf-8").strip()
    _rubric_lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 40]
    return text


def authorized(header: str, secret: str) -> bool:
    if not header.lower().startswith("bearer "):
        return False
    got = header.split(None, 1)[1].strip()
    return secrets.compare_digest(got.encode("utf-8"), secret.encode("utf-8"))


def allow(ip: str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    with _lock:
        window = [t for t in _hits.get(ip, []) if now - t < 60]
        if len(window) >= RATE_PER_MIN:
            _hits[ip] = window
            return False
        window.append(now)
        _hits[ip] = window
        return True


def has_marker(text: str, markers: tuple[str, ...]) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    for word in words:
        for marker in markers:
            if word == marker or (len(marker) >= 4 and word.startswith(marker)):
                return True
    return False


def structural_retry(answer: str) -> str | None:
    words = re.findall(r"[a-z']+", answer.lower())
    extra = [word for word in words if word not in JUNK_WORDS]
    if len(extra) < 4:
        return "Say what you would do, not only the words apply and learn."
    if not has_marker(answer, HOW_MARKERS):
        return "Add how you would learn the gap, use a related skill, or treat the interview as practice."
    if not has_marker(answer, STANCE_MARKERS):
        return "State whether you would apply, and why."
    return None


def feedback_leaks(feedback: str) -> bool:
    folded = feedback.lower()
    for line in _rubric_lines:
        if line.lower() in folded:
            return True
    return False


def parse_model(text: str) -> dict | None:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict") or "").strip().lower()
    feedback = re.sub(r"\s+", " ", str(obj.get("feedback") or "")).strip()
    if verdict not in {"pass", "retry"} or not feedback or len(feedback) > 180:
        return None
    if "\n" in feedback:
        return None
    return {"verdict": verdict, "feedback": feedback}


def decide(answer: str, model_text: str) -> tuple[int, dict]:
    parsed = parse_model(model_text)
    if not parsed:
        return 502, {"error": "malformed"}
    verdict = parsed["verdict"]
    feedback = parsed["feedback"]
    if verdict == "pass":
        missing = structural_retry(answer)
        if missing:
            verdict = "retry"
            feedback = missing
    if feedback_leaks(feedback):
        feedback = "Name the missing part of your answer."
    return 200, {"verdict": verdict, "feedback": feedback}


def ollama_grade(answer: str, rubric: str, timeout: float | None = None) -> str:
    system = (
        "You grade one student answer. Reply with JSON only. "
        "Keys are verdict and feedback. verdict is pass or retry. "
        "feedback is one line under 140 characters that names only the missing piece, "
        "or Complete. when the answer passes. Never write a sample answer. Never copy the rubric.\n\n"
        + rubric
    )
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": -1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": answer},
        ],
        "options": {"temperature": 0, "num_predict": 60, "num_ctx": 2048},
    }
    req = urllib.request.Request(
        OLLAMA_HOST.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S if timeout is None else timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    message = body.get("message") or {}
    return str(message.get("content") or "")


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_HOST.rstrip("/") + "/api/tags", timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    names = [str(item.get("name") or "") for item in body.get("models") or []]
    return any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") or OLLAMA_MODEL.startswith(name) for name in names)


class Handler(BaseHTTPRequestHandler):
    secret = ""
    rubric = ""

    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _ip(self) -> str:
        return self.client_address[0] if self.client_address else "0"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/health":
            self._json(404, {"error": "not_found"})
            return
        if not allow(self._ip()):
            self._json(429, {"error": "rate_limit"})
            return
        if not ollama_up():
            self._json(503, {"error": "model_down"})
            return
        self._json(200, {"status": "ok", "model": OLLAMA_MODEL})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/grade":
            self._json(404, {"error": "not_found"})
            return
        if not authorized(self.headers.get("Authorization") or "", self.secret):
            self._json(401, {"error": "unauthorized"})
            return
        if not allow(self._ip()):
            self._json(429, {"error": "rate_limit"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > 8000:
            self._json(400, {"error": "bad_body"})
            return
        try:
            incoming = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._json(400, {"error": "bad_body"})
            return
        if not isinstance(incoming, dict):
            self._json(400, {"error": "bad_body"})
            return
        if str(incoming.get("rubric") or "") != "wrap-apply":
            self._json(400, {"error": "unknown_rubric"})
            return
        answer = str(incoming.get("answer") or "").strip()
        if len(answer) < 12 or len(answer) > 2000:
            self._json(400, {"error": "bad_answer"})
            return
        try:
            model_text = ollama_grade(answer, self.rubric)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            self._json(504, {"error": "timeout"})
            return
        code, obj = decide(answer, model_text)
        self._json(code, obj)


def warm() -> None:
    try:
        ollama_grade(
            "Yes I would apply, because the interview is practice and I am eager to learn.",
            Handler.rubric,
            timeout=30,
        )
    except Exception:
        return


def main() -> None:
    Handler.secret = load_secret()
    Handler.rubric = load_rubric()
    warm()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
