#!/usr/bin/env python3
"""Lexical + merge tests for topic aliases (no Chroma / Ollama)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.retrieve import is_short_query, lexical_hits, load_aliases, merge_ranked

ROOT = Path(__file__).resolve().parents[1]
GRADE_MAP = json.loads((ROOT / "data/grade_map.json").read_text())
ALIASES = load_aliases(ROOT / "data/topic_aliases.json")


def ids(hits: list[tuple[int, str]]) -> list[str]:
    return [lid for _, lid in hits]


class LexicalAliasTests(unittest.TestCase):
    def test_navigation_hits_wayfinding(self):
        got = ids(lexical_hits("navigation", GRADE_MAP, ALIASES))
        self.assertIn("4-5-wayfinding-coordinate-battleship", got)

    def test_wayfinding_hits_title(self):
        got = ids(lexical_hits("wayfinding", GRADE_MAP, ALIASES))
        self.assertIn("4-5-wayfinding-coordinate-battleship", got)

    def test_3d_shapes_hits_hale_lessons(self):
        got = ids(lexical_hits("3d shapes", GRADE_MAP, ALIASES))
        self.assertIn("k-hale-3d-shapes-three-little-puaa-challenge", got)
        self.assertIn("1-hale-3d-composite-shapes-three-little-puaa-3d-hale-challenge", got)
        self.assertIn("prek-the-3ps-3d-hale-challenge", got)

    def test_stars_hits_lewa_or_wayfinding(self):
        got = ids(lexical_hits("stars", GRADE_MAP, ALIASES))
        self.assertTrue(
            {"prek-look-up-at-the-lewa-celestial-objects", "4-5-wayfinding-coordinate-battleship"}
            & set(got),
            got,
        )

    def test_unrelated_stays_quiet(self):
        got = ids(lexical_hits("photosynthesis on mars", GRADE_MAP, ALIASES))
        self.assertEqual(got, [])


class MergeTests(unittest.TestCase):
    def test_cross_grade_when_same_empty(self):
        other = [("doc", {"lesson": "4-5-wayfinding-coordinate-battleship", "grades": "4,5"})]
        out, cross = merge_ranked([], [], [], other, top_k=4)
        self.assertTrue(cross)
        self.assertEqual(out[0][1]["lesson"], "4-5-wayfinding-coordinate-battleship")

    def test_weak_same_grade_yields_to_alias_other_grade(self):
        same_v = [("fish", {"lesson": "k-fishy-traits-ia-ai", "grades": "K"})]
        other_lex = [("wf", {"lesson": "4-5-wayfinding-coordinate-battleship", "grades": "4,5"})]
        out, cross = merge_ranked(same_v, [], [], other_lex, top_k=4)
        self.assertTrue(cross)
        self.assertEqual(out[0][1]["lesson"], "4-5-wayfinding-coordinate-battleship")

    def test_long_query_keeps_vector_first(self):
        same_v = [("star", {"lesson": "prek-look-up-at-the-lewa-celestial-objects", "grades": "PreK"})]
        same_lex = [("voy", {"lesson": "4-5-wa-a-voyaging-making-weight", "grades": "4,5"})]
        out, cross = merge_ranked(same_v, same_lex, [], [], top_k=4, prefer_lex=False)
        self.assertFalse(cross)
        self.assertEqual(out[0][1]["lesson"], "prek-look-up-at-the-lewa-celestial-objects")

    def test_short_query_helper(self):
        self.assertTrue(is_short_query("navigation"))
        self.assertTrue(is_short_query("3d shapes"))
        self.assertFalse(is_short_query("How did Hawaiians navigate with the stars?"))

    def test_same_grade_alias_wins(self):
        same_lex = [("d", {"lesson": "k-hale-3d-shapes-three-little-puaa-challenge", "grades": "K"})]
        other = [("d2", {"lesson": "4-5-a-hale-made-of-angles-hale-angles", "grades": "4,5"})]
        out, cross = merge_ranked([], same_lex, [], other, top_k=4)
        self.assertFalse(cross)
        self.assertEqual(out[0][1]["lesson"], "k-hale-3d-shapes-three-little-puaa-challenge")


if __name__ == "__main__":
    unittest.main()
