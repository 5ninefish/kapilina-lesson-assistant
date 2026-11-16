#!/usr/bin/env python3
"""Score frozen Wrap-apply answers against the separate /grade service."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEMS = HERE / "grade_items.json"
OUT = HERE / "grade_run.json"
ROOT = HERE.parent


def secret_from(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().splitlines()[0].strip()


def post(url: str, token: str, answer: str) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps({"rubric": "wrap-apply", "answer": answer}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": "malformed"}
        return exc.code, body


def leaks(feedback: str, rubric: str) -> bool:
    folded = feedback.lower()
    for line in rubric.splitlines():
        line = line.strip()
        if len(line) >= 40 and line.lower() in folded:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011/grade")
    parser.add_argument("--secret-file", default=str(ROOT / "data" / ".grade-secret"))
    parser.add_argument("--sleep", type=float, default=2.1)
    args = parser.parse_args()
    token = secret_from(Path(args.secret_file))
    spec = json.loads(ITEMS.read_text(encoding="utf-8"))
    rubric = (ROOT / "grader" / "wrap_apply_rubric.txt").read_text(encoding="utf-8")
    rows = []
    false_passes = 0
    leak_n = 0
    errors = 0
    for index, item in enumerate(spec["items"]):
        if index:
            time.sleep(args.sleep)
        try:
            code, body = post(args.url, token, item["answer"])
        except Exception as exc:  # noqa: BLE001
            code, body = 0, {"error": type(exc).__name__}
        verdict = str(body.get("verdict") or "")
        feedback = str(body.get("feedback") or "")
        leaked = bool(feedback) and leaks(feedback, rubric)
        false_pass = item["expect"] != "pass" and verdict == "pass"
        error = code != 200 or verdict not in {"pass", "retry"}
        if false_pass:
            false_passes += 1
        if leaked:
            leak_n += 1
        if error:
            errors += 1
        rows.append(
            {
                "id": item["id"],
                "expect": item["expect"],
                "verdict": verdict,
                "code": code,
                "false_pass": false_pass,
                "leak": leaked,
                "error": error,
                "feedback": feedback[:180],
            }
        )
    passed = sum(1 for row in rows if row["expect"] == "pass" and row["verdict"] == "pass")
    report = {
        "at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "count": len(rows),
        "pass_hits": passed,
        "false_passes": false_passes,
        "leaks": leak_n,
        "errors": errors,
        "ok": false_passes == 0 and leak_n == 0 and errors == 0,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("count", "pass_hits", "false_passes", "leaks", "errors", "ok")}))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
