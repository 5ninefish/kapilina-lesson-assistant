"""Title/alias lexical ranking + grade-aware merge for the lesson RAG path."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

SKIP_LNUM = re.compile(r"^L\d+$", re.I)
SKIP_NAME = re.compile(r"(?i)(vocabulary check|answer sheet|answer key|rubric)")
TOKEN_RE = re.compile(r"[a-z0-9]+")
OKINA_RE = re.compile(r"[ʻ''`´]")
KAHAKO = str.maketrans("āēīōūäëïöü", "aeiouaeiou")
STOP = {
    "the", "a", "an", "and", "or", "of", "in", "for", "to", "with", "about",
    "how", "did", "do", "does", "what", "which", "who", "their", "they",
    "own", "lesson", "lessons", "plan", "teacher", "grade",
}
# Short tokens that still count as topical.
KEEP_SHORT = {"3d", "k", "waa"}

PHRASE_SCORE = 10
ALIAS_SCORE = 8
TOKEN_SCORE = 2
LEX_MIN = 8


def fold_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = OKINA_RE.sub("", s)
    return s.translate(KAHAKO)


def load_aliases(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    raw = data.get("aliases") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, vals in raw.items():
        kf = fold_text(str(key)).strip()
        if not kf:
            continue
        items = vals if isinstance(vals, list) else [vals]
        out[kf] = [fold_text(str(v)).strip() for v in items if str(v).strip()]
    return out


def _distinctive_tokens(folded: str) -> list[str]:
    toks = TOKEN_RE.findall(folded)
    keep = []
    for t in toks:
        if t in STOP:
            continue
        if len(t) >= 3 or t in KEEP_SHORT:
            keep.append(t)
    return keep


def expand_query(question: str, aliases: dict[str, list[str]]) -> tuple[str, set[str], bool]:
    """Folded question, expanded match terms, whether an alias key fired."""
    qf = fold_text(question).strip()
    terms: set[str] = set()
    alias_fired = False
    for key, vals in aliases.items():
        if not key:
            continue
        keyed = (key == qf) or (f" {key} " in f" {qf} ") or (key in qf and " " in key)
        if not keyed:
            continue
        alias_fired = True
        terms.add(key)
        terms.update(v for v in vals if v)
    terms.update(_distinctive_tokens(qf))
    if qf:
        terms.add(qf)
    return qf, {t for t in terms if t}, alias_fired


def lexical_hits(
    question: str,
    grade_map: dict,
    aliases: dict[str, list[str]],
    limit: int = 8,
) -> list[tuple[int, str]]:
    """Return (score, lesson_id) for title/slug/filename matches, highest first."""
    qf, terms, alias_fired = expand_query(question, aliases)
    if not qf:
        return []
    scored: list[tuple[int, str]] = []
    for lid, info in (grade_map or {}).items():
        if SKIP_LNUM.match(str(lid) or ""):
            continue
        info = info or {}
        blob = fold_text(
            f"{lid} {info.get('title') or ''} {info.get('filename') or ''}"
        )
        if not blob:
            continue
        score = 0
        if len(qf) >= 3 and qf in blob:
            score += PHRASE_SCORE
        if alias_fired:
            for t in terms:
                if len(t) >= 3 and t in blob:
                    score += ALIAS_SCORE
                    break
            else:
                if any(t in blob for t in terms if t in KEEP_SHORT):
                    score += ALIAS_SCORE
        else:
            for t in _distinctive_tokens(qf):
                if t in blob:
                    score += TOKEN_SCORE
        if score >= LEX_MIN:
            term_hits = sum(1 for t in terms if len(t) >= 3 and t in blob)
            scored.append((score, term_hits, str(lid)))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    # de-dupe ids keeping best score
    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for score, _hits, lid in scored:
        if lid in seen:
            continue
        seen.add(lid)
        out.append((score, lid))
        if len(out) >= limit:
            break
    return out


def grade_ok(meta: dict, grade: str) -> bool:
    if grade == "all":
        return True
    grades = (meta.get("grades") or "")
    return grade in grades.split(",")


def split_vector_hits(
    docs: list,
    metas: list,
    dists: list,
    grade: str,
    sim_threshold: float,
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    same: list[tuple[str, dict]] = []
    other: list[tuple[str, dict]] = []
    seen_same: set[str] = set()
    seen_other: set[str] = set()
    for d, m, dist in zip(docs, metas, dists):
        m = m or {}
        if (1 - float(dist)) < sim_threshold:
            continue
        if SKIP_NAME.search(m.get("filename") or ""):
            continue
        lid = m.get("lesson") or m.get("filename") or ""
        if SKIP_LNUM.match(lid or ""):
            continue
        bucket = same if grade_ok(m, grade) else other
        seen = seen_same if bucket is same else seen_other
        if lid in seen:
            continue
        seen.add(lid)
        bucket.append((d, m))
    return same, other


def merge_ranked(
    same_vector: list[tuple[str, dict]],
    same_lex: list[tuple[str, dict]],
    other_vector: list[tuple[str, dict]],
    other_lex: list[tuple[str, dict]],
    top_k: int,
    prefer_lex: bool = True,
) -> tuple[list[tuple[str, dict]], bool]:
    """Prefer same-grade. If only other-grade title/alias hits exist, use those.

    prefer_lex=True (short topic queries): title/alias first.
    prefer_lex=False (full sentences): vector first so a weak alias does not
    bury the passage that actually answers the question.
    """
    seen: set[str] = set()
    out: list[tuple[str, dict]] = []

    def take(src: list[tuple[str, dict]]) -> None:
        for d, m in src:
            if len(out) >= top_k:
                return
            lid = (m or {}).get("lesson") or (m or {}).get("filename") or ""
            if SKIP_LNUM.match(lid or "") or lid in seen:
                continue
            seen.add(lid)
            out.append((d, m))

    if prefer_lex:
        take(same_lex)
        if out:
            take(same_vector)
            return out, False
    else:
        take(same_vector)
        take(same_lex)
        if out:
            return out, False
    if other_lex:
        take(other_lex)
        take(other_vector)
        return out, True
    take(same_vector)
    if out:
        return out, False
    take(other_vector)
    return out, bool(out)


def is_short_query(question: str) -> bool:
    return len(_distinctive_tokens(fold_text(question))) <= 2


def grade_phrase(grade: str) -> str:
    if grade == "all":
        return "All Grades"
    if grade == "PreK":
        return "PreK"
    if grade == "K":
        return "Kindergarten"
    return f"Grade {grade}"


def lesson_grade_phrase(grades: str) -> str:
    parts = [p for p in (grades or "").split(",") if p]
    if parts == ["PreK"]:
        return "PreK"
    if parts == ["K"]:
        return "Kindergarten"
    if not parts:
        return "another grade"
    if len(parts) == 1:
        return f"Grade {parts[0]}"
    return f"Grades {'–'.join(parts)}"
