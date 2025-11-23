import sys
import csv
import json
from pathlib import Path
from collections import Counter, OrderedDict

INPUT_JSONL = "final_dataset_core.jsonl"   # <-- Change this to dataset JSONL file

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

def categorize(path: str):
    p = Path(path)
    ext = p.suffix
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
    return None

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

def write_csv(output_csv: Path, ordered_counts: OrderedDict, total: int):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Count", "Percentage"])
        for cat, cnt in ordered_counts.items():
            pct = (cnt / total * 100.0) if total else 0.0
            w.writerow([cat, cnt, f"{pct:.2f}%"])
        w.writerow(["Total", total, "100.0%" if total else "0.0%"])

def main(argv=None):
    argv = argv or sys.argv[1:]
    jsonl_file = Path(argv[0]) if argv else Path(INPUT_JSONL)
    if not jsonl_file.exists():
        print(f"ERROR: JSONL file not found: {jsonl_file}")
        sys.exit(2)

    # Optional explicit CSV path
    output_csv = Path(argv[1]) if len(argv) > 1 else jsonl_file.with_name(jsonl_file.stem + "_language_counts.csv")

    counts = Counter()
    total_considered = 0

    for path in load_paths(jsonl_file):
        cat = categorize(path)
        if cat is not None:
            counts[cat] += 1
            total_considered += 1

    for c in CATEGORY_ORDER:
        counts.setdefault(c, 0)

    ordered_counts = OrderedDict()
    for c in CATEGORY_ORDER:
        ordered_counts[c] = counts[c]
    for c, v in counts.items():
        if c not in ordered_counts:
            ordered_counts[c] = v

    print()
    print("{:<16} {:>10} {:>11}".format("Category", "Count", "Percentage"))
    print("-" * 39)
    for cat, cnt in ordered_counts.items():
        pct = (cnt / total_considered * 100.0) if total_considered else 0.0
        print("{:<16} {:>10} {:>10.1f}%".format(cat, cnt, pct))
    print("-" * 39)
    print("{:<16} {:>10} {:>10}".format("Total", total_considered, "100.0%" if total_considered else "0.0%"))
    print()

    write_csv(output_csv, ordered_counts, total_considered)
    print(f"CSV written to: {output_csv}")

if __name__ == "__main__":
    main()
