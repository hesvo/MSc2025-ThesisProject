import sys
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

# Setup environment
load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")

PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

DEFAULT_MODEL_SIZE = "12b"
DEFAULT_DATASET_TYPE = "extended"

EVAL_ROOT_BASE = PROJECT_ROOT / "results" / "evaluation" / "baseline" / "base_12b"
EVAL_ROOT_ADAPTED = PROJECT_ROOT / "results" / "evaluation" / "adapted" / f"adapted_12b_{DEFAULT_DATASET_TYPE}"

OUT_ROOT = PROJECT_ROOT / "results" / "stats"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Helpers
def allowed_metrics_for_task(task: str) -> Dict[str, str]:
    if task == "summarization":
        return {
            "bleu": "bleu_score",
            "rouge1": "rouge1",
            "rouge2": "rouge2",
            "rougel": "rougeL",
            "bertscore": "bertscore_f1",
        }
    elif task == "generation":
        return {"codebleu": "codebleu"}
    else:
        raise ValueError(f"Unknown task: {task}")

def expected_eval_filenames(
    task: str,
    source: str,
    summary_length: str,
    shot_count: str,
    model_size: str,
    dataset_type: str,
    use_rci: bool,
) -> Tuple[Path, Path]:
    base_model_name = f"base_{model_size}"
    adapted_model_name = f"adapted_{model_size}_{dataset_type}"

    suffix = f"{task}_{source}_{summary_length}_{shot_count}"
    base_csv = EVAL_ROOT_BASE / f"evaluation_results_{base_model_name}_{suffix}{'_rci' if use_rci else ''}.csv"
    adapted_csv = EVAL_ROOT_ADAPTED / f"evaluation_results_{adapted_model_name}_{suffix}{'_rci' if use_rci else ''}.csv"
    return base_csv, adapted_csv

def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 5000, alpha: float = 0.05, random_state: int = 0):
    rng = np.random.default_rng(random_state)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = stat_fn(values[idx])
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return lo, hi

def load_and_pair(
    base_path: Path,
    adapted_path: Path,
    metric_col: str,
    id_col: str = "id",
) -> pd.DataFrame:
    if not base_path.exists():
        raise FileNotFoundError(f"Base results not found: {base_path}")
    if not adapted_path.exists():
        raise FileNotFoundError(f"Adapted results not found: {adapted_path}")

    base_df = pd.read_csv(base_path)
    adapted_df = pd.read_csv(adapted_path)

    missing_b = [c for c in [id_col, metric_col] if c not in base_df.columns]
    missing_a = [c for c in [id_col, metric_col] if c not in adapted_df.columns]
    if missing_b:
        raise KeyError(f"Base CSV missing columns: {missing_b} in {base_path}")
    if missing_a:
        raise KeyError(f"Adapted CSV missing columns: {missing_a} in {adapted_path}")

    merged = pd.merge(
        base_df[[id_col, metric_col]].rename(columns={metric_col: "base_score"}),
        adapted_df[[id_col, metric_col]].rename(columns={metric_col: "adapted_score"}),
        on=id_col,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping problem IDs between base and adapted CSVs.")
    return merged

def signed_rank_wilcoxon(d: np.ndarray, alternative: str = "two-sided"):
    res = stats.wilcoxon(d, zero_method="wilcox", alternative=alternative, method="auto", correction=False)
    return res.statistic, res.pvalue

def summarize_pairwise(merged: pd.DataFrame) -> Dict[str, float]:
    d = (merged["adapted_score"] - merged["base_score"]).to_numpy(dtype=float)

    n_total = d.size
    n_zero = int((d == 0).sum())
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())

    mean_diff = float(np.mean(d))
    median_diff = float(np.median(d))
    cl_effect = float(n_pos / n_total)

    mean_lo, mean_hi = bootstrap_ci(d, np.mean)
    med_lo, med_hi = bootstrap_ci(d, np.median)

    return {
        "n_pairs": n_total,
        "n_ties": n_zero,
        "n_improved": n_pos,
        "n_worse": n_neg,
        "mean_diff": mean_diff,
        "mean_diff_ci_lo": mean_lo,
        "mean_diff_ci_hi": mean_hi,
        "median_diff": median_diff,
        "median_diff_ci_lo": med_lo,
        "median_diff_ci_hi": med_hi,
        "p_d_gt_0": cl_effect,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon signed-rank tests for base vs adapted per-problem scores."
    )
    parser.add_argument("--task", choices=["summarization", "generation"], required=True,
                        help="Which benchmark task.")
    parser.add_argument("--metric", required=False,
                        help="summarization: bleu|rouge1|rouge2|rougel|bertscore; generation: codebleu (default).")
    parser.add_argument("--source", choices=["xl", "auto"], required=True)
    parser.add_argument("--summary_length", choices=["short", "long"], required=True)
    parser.add_argument("--shot_count", choices=["zero", "one", "three"], required=True)
    parser.add_argument("--use_rci", action="store_true",
                        help="If true, read the '*_rci.csv' per-problem file variant (generation only unless you made RCI for summarization as well).")
    parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided",
                        help="Directional alternative for Wilcoxon (e.g., 'greater' tests adapted > base).")
    parser.add_argument("--model_size", default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--dataset_type", default=DEFAULT_DATASET_TYPE,
                        help="Only affects adapted filename (e.g., 'core').")
    parser.add_argument("--out_tag", default="", help="Optional string to append to output filenames.")
    return parser.parse_args()

def main():
    args = parse_args()

    metric_map = allowed_metrics_for_task(args.task)
    metric_key = (args.metric or ("codebleu" if args.task == "generation" else None))
    if metric_key is None:
        print("Error: --metric is required for summarization.")
        sys.exit(1)
    if metric_key not in metric_map:
        print(f"Error: unknown metric '{args.metric}'. Allowed: {list(metric_map.keys())}")
        sys.exit(1)

    metric_col = metric_map[metric_key]

    base_csv, adapted_csv = expected_eval_filenames(
        task=args.task,
        source=args.source,
        summary_length=args.summary_length,
        shot_count=args.shot_count,
        model_size=args.model_size,
        dataset_type=args.dataset_type,
        use_rci=args.use_rci,
    )

    print(f"Reading per-problem scores:\n  BASE    = {base_csv}\n  ADAPTED = {adapted_csv}\n  METRIC  = {metric_key} (column='{metric_col}')\n")

    merged = load_and_pair(base_csv, adapted_csv, metric_col)
    merged = merged[np.isfinite(merged["base_score"]) & np.isfinite(merged["adapted_score"])]
    if merged.empty:
        print("No valid (finite) paired scores after filtering.")
        sys.exit(1)

    d = (merged["adapted_score"] - merged["base_score"]).to_numpy(dtype=float)

    # Summary & Wilcoxon
    summary = summarize_pairwise(merged)
    W, p = signed_rank_wilcoxon(d, alternative=args.alternative)
    summary.update({"wilcoxon_statistic": float(W), "wilcoxon_pvalue": float(p), "alternative": args.alternative})

    print("----- Wilcoxon signed-rank (paired) -----")
    print(f"Pairs (non-missing): {summary['n_pairs']} | ties: {summary['n_ties']} | improved: {summary['n_improved']} | worse: {summary['n_worse']}")
    print(f"Mean Δ = {summary['mean_diff']:.4f}  (95% CI [{summary['mean_diff_ci_lo']:.4f}, {summary['mean_diff_ci_hi']:.4f}])")
    print(f"Median Δ (HL) = {summary['median_diff']:.4f}  (95% CI [{summary['median_diff_ci_lo']:.4f}, {summary['median_diff_ci_hi']:.4f}])")
    print(f"Common-language effect P(Δ>0) = {summary['p_d_gt_0']:.3f}")
    print(f"Wilcoxon W = {summary['wilcoxon_statistic']:.1f}, p = {summary['wilcoxon_pvalue']:.4g} (alternative='{args.alternative}')")

    # Save differences and summary
    tag = f"{args.task}-{metric_key}-{args.source}-{args.summary_length}-{args.shot_count}"
    if args.use_rci:
        tag += "-rci"
    if args.out_tag:
        tag += f"-{args.out_tag}"

    diffs_out = OUT_ROOT / f"paired_diffs_{tag}.csv"
    merged.assign(diff=d).to_csv(diffs_out, index=False)

    summary_out = OUT_ROOT / f"wilcoxon_summary_{tag}.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved:\n  Per-problem diffs → {diffs_out}\n  Summary JSON     → {summary_out}")

if __name__ == "__main__":
    main()
