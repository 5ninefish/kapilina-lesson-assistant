#!/bin/bash
# Loads the grader secret into the process environment, then serves /grade.
set -euo pipefail
ROOT="/Users/peakforge/kapilinanoeau"
export GRADE_SECRET="$(tr -d '\r\n' < "$ROOT/data/.grade-secret")"
export GRADE_PORT="${GRADE_PORT:-8011}"
export GRADE_RUBRIC="$ROOT/grader/wrap_apply_rubric.txt"
exec /usr/bin/python3 "$ROOT/scripts/grade_server.py"
