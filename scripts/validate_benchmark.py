"""Validate the multimodal benchmark JSON against task constraints.

Checks per video:
- exactly 10 items
- type distribution: 2 of each (factual, temporal, sequential, cross_modal, summary)
- at least 6 items require image or both modalities
- at least 4 items require both text and image
- every non-NEEDS_VALIDATION answer must have at least one gold_evidence span
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "benchmark" / "multimodal_benchmark_v2.json"

REQUIRED_TYPES = {"factual", "temporal", "sequential", "cross_modal", "summary"}


def main() -> int:
    data = json.loads(PATH.read_text())
    overall_ok = True
    for video in data:
        vid = video["video_id"]
        items = video["items"]
        problems = []

        if len(items) != 10:
            problems.append(f"  expected 10 items, got {len(items)}")

        type_counts = Counter(it["question_type"] for it in items)
        for t in REQUIRED_TYPES:
            if type_counts.get(t, 0) != 2:
                problems.append(f"  type {t}: expected 2, got {type_counts.get(t, 0)}")

        unknown_types = set(type_counts) - REQUIRED_TYPES
        if unknown_types:
            problems.append(f"  unknown types: {sorted(unknown_types)}")

        visual_count = 0
        both_count = 0
        for it in items:
            mods = set(it.get("evidence_modalities_needed", []))
            if mods & {"image", "both"}:
                visual_count += 1
            if "text" in mods and "image" in mods:
                both_count += 1

        if visual_count < 6:
            problems.append(
                f"  modality: need >=6 visual/both items, got {visual_count}"
            )
        if both_count < 4:
            problems.append(
                f"  modality: need >=4 text+image items, got {both_count}"
            )

        for it in items:
            qid = it["question_id"]
            ans = it["ideal_answer"]
            evidence = it.get("gold_evidence", [])
            if ans != "NEEDS_VALIDATION" and not evidence:
                problems.append(f"  {qid}: missing gold_evidence")
            if ans == "NEEDS_VALIDATION" and not it.get("validation_note"):
                problems.append(f"  {qid}: NEEDS_VALIDATION missing validation_note")

        if problems:
            overall_ok = False
            print(f"[FAIL] {vid}")
            for p in problems:
                print(p)
        else:
            print(f"[ OK ] {vid}: 10 items, type 2/2/2/2/2, visual={visual_count}, both={both_count}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
