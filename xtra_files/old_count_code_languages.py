#!/usr/bin/env python3
"""
Count files by language category from a JSONL dataset.

- Reads a JSON Lines file where each line is a JSON object that contains a "path_in_repo" field.
- Categorizes each file path into one of:
  "C++", "Java", "C/C++ Header", "Python", "C", "Documentation"
- Summarizes counts and percentages and prints a table with totals.

Usage:
    Edit INPUT_JSONL at the top, or run as:
        python count_code_languages.py /path/to/data.jsonl

Notes:
- Only the categories above are counted. Files with other extensions are ignored.
- "Documentation" includes: .md, .rst, .adoc, .dox
"""

import sys
import json
from pathlib import Path
from collections import Counter, OrderedDict

# ---- Configure the default JSONL path here ----
INPUT_JSONL = "data.jsonl"   # <-- Change this to your JSONL file

# ---- Category mapping by file extension ----
CPP_EXTS = {".cpp", ".cc", ".cxx", ".c++", ".C"}
JAVA_EXTS = {".java"}
HEADER_EXTS = {".h", ".hpp", ".hh", ".hxx", ".h++"}
PY_EXTS = {".py"}
C_EXTS = {".c"}
DOC_EXTS = {".md", ".rst", ".adoc", ".dox"}

CATEGORY_ORDER = [
    "C++",
    "Java",
    "C/C++ Header",
    "Python",
    "C",
    "Documentation",
]

def categorize(path: str) -> str | None:
    p = Path(path)
    ext = p.suffix  # last suffix only (".tar.gz" -> ".gz", which we ignore anyway)
    ext_lower = ext.lower()

    if ext_lower in CPP_EXTS:
        return "C++"
    if ext_lower in JAVA_EXTS:
        return "Java"
    if ext_lower in HEADER_EXTS:
        return "C/C++ Header"
    if ext_lower in PY_EXTS:
        return "Python"
    if ext_lower in C_EXTS:
        return "C"
    if ext_lower in DOC_EXTS:
        return "Documentation"
    return None  # not in our target categories

def load_paths(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = obj.get("path_in_repo")
            if path:
                yield path

def main(argv=None):
    argv = argv or sys.argv[1:]
    jsonl_file = Path(argv[0]) if argv else Path(INPUT_JSONL)
    if not jsonl_file.exists():
        print(f"ERROR: JSONL file not found: {jsonl_file}")
        sys.exit(2)

    counts = Counter()
    total_considered = 0

    for path in load_paths(jsonl_file):
        cat = categorize(path)
        if cat is not None:
            counts[cat] += 1
            total_considered += 1

    # Ensure all categories are present (even if zero)
    for c in CATEGORY_ORDER:
        counts.setdefault(c, 0)

    # Prepare ordered results following CATEGORY_ORDER, then any unexpected
    ordered_counts = OrderedDict()
    for c in CATEGORY_ORDER:
        ordered_counts[c] = counts[c]
    # Append any other categories that might have been introduced
    for c, v in counts.items():
        if c not in ordered_counts:
            ordered_counts[c] = v

    # Print table
    print()
    print("{:<16} {:>10} {:>11}".format("Category", "Count", "Percentage"))
    print("-" * 39)
    for cat, cnt in ordered_counts.items():
        pct = (cnt / total_considered * 100.0) if total_considered else 0.0
        print("{:<16} {:>10} {:>10.1f}%".format(cat, cnt, pct))
    print("-" * 39)
    print("{:<16} {:>10} {:>10}".format("Total", total_considered, "100.0%" if total_considered else "0.0%"))
    print()

if __name__ == "__main__":
    main()
