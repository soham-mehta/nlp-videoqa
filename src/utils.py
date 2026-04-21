from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def safe_filename(name: str, *, max_len: int = 120) -> str:
    """
    Make a reasonably safe filename component for macOS/Linux/Windows.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip().replace(" ", "_")
    if not name:
        name = "untitled"
    return name[:max_len]


def nearest_index(sorted_values: list[float], target: float) -> int:
    """
    Return the index of the nearest value in a sorted list.
    If two values are equally near, returns the lower index.
    """
    if not sorted_values:
        raise ValueError("nearest_index() requires a non-empty list")
    lo, hi = 0, len(sorted_values) - 1
    if target <= sorted_values[lo]:
        return lo
    if target >= sorted_values[hi]:
        return hi

    # binary search for insertion point
    while lo <= hi:
        mid = (lo + hi) // 2
        v = sorted_values[mid]
        if math.isclose(v, target, rel_tol=0.0, abs_tol=1e-9):
            return mid
        if v < target:
            lo = mid + 1
        else:
            hi = mid - 1

    # lo is insertion point, hi is lo-1
    if lo <= 0:
        return 0
    if lo >= len(sorted_values):
        return len(sorted_values) - 1
    before = sorted_values[lo - 1]
    after = sorted_values[lo]
    if (target - before) <= (after - target):
        return lo - 1
    return lo


def overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Half-open interval overlap check: [start, end)."""
    return (start_a < end_b) and (start_b < end_a)
