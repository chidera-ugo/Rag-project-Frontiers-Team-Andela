"""
Shared evaluation logger — appends results to evaluation_log.jsonl.
Run `uv run logs.py` to print a formatted history table.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "evaluation_log.jsonl"


def log_run(
    version: str,
    eval_type: str,  # "retrieval" or "answer"
    metrics: dict,
    by_category: dict,
):
    """Append a timestamped evaluation result to the log file."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "eval_type": eval_type,
        "metrics": metrics,
        "by_category": by_category,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── Display ───────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
RESET = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _color_metric(value: float, green: float, amber: float) -> str:
    if value >= green:
        return f"{GREEN}{value:.4f}{RESET}"
    elif value >= amber:
        return f"{YELLOW}{value:.4f}{RESET}"
    return f"{RED}{value:.4f}{RESET}"


def print_history():
    entries = load_log()
    if not entries:
        print("No evaluation runs logged yet.")
        return

    print(f"\n{BOLD}{'─' * 80}{RESET}")
    print(f"{BOLD}  EVALUATION HISTORY  ({LOG_FILE.name}){RESET}")
    print(f"{BOLD}{'─' * 80}{RESET}\n")

    for i, e in enumerate(entries):
        ts       = e["timestamp"]
        version  = e["version"]
        etype    = e["eval_type"]
        metrics  = e["metrics"]

        label = f"[{i+1}] {ts}  {BOLD}{version:<12}{RESET}  {etype}"
        print(f"  {label}")

        if etype == "retrieval":
            mrr  = metrics.get("mrr", 0)
            ndcg = metrics.get("ndcg", 0)
            cov  = metrics.get("coverage", 0)
            print(f"       MRR {_color_metric(mrr, 0.9, 0.75)}  "
                  f"nDCG {_color_metric(ndcg, 0.9, 0.75)}  "
                  f"Coverage {_color_metric(cov / 100, 0.9, 0.75)} ({cov:.1f}%)")
        else:
            acc  = metrics.get("accuracy", 0)
            comp = metrics.get("completeness", 0)
            rel  = metrics.get("relevance", 0)
            print(f"       Accuracy {_color_metric(acc/5, 0.9, 0.8)} ({acc:.2f}/5)  "
                  f"Completeness {_color_metric(comp/5, 0.9, 0.8)} ({comp:.2f}/5)  "
                  f"Relevance {_color_metric(rel/5, 0.9, 0.8)} ({rel:.2f}/5)")

        by_cat = e.get("by_category", {})
        if by_cat:
            key = "mrr" if etype == "retrieval" else "accuracy"
            cat_line = "  ".join(
                f"{cat}: {_color_metric(score, 0.9, 0.75)}"
                for cat, score in sorted(by_cat.items())
            )
            print(f"       {DIM}{cat_line}{RESET}")

        print()

    # ── Comparison table if multiple runs ──
    retrieval_runs = [e for e in entries if e["eval_type"] == "retrieval"]
    if len(retrieval_runs) >= 2:
        print(f"{BOLD}  Retrieval comparison{RESET}")
        print(f"  {'#':<3} {'Timestamp':<22} {'Version':<14} {'MRR':>8} {'nDCG':>8} {'Coverage':>10}")
        print(f"  {'─'*3} {'─'*22} {'─'*14} {'─'*8} {'─'*8} {'─'*10}")
        for i, e in enumerate(retrieval_runs):
            m = e["metrics"]
            print(f"  {i+1:<3} {e['timestamp']:<22} {e['version']:<14} "
                  f"{m.get('mrr',0):>8.4f} {m.get('ndcg',0):>8.4f} {m.get('coverage',0):>9.1f}%")
        print()

    answer_runs = [e for e in entries if e["eval_type"] == "answer"]
    if len(answer_runs) >= 2:
        print(f"{BOLD}  Answer comparison{RESET}")
        print(f"  {'#':<3} {'Timestamp':<22} {'Version':<14} {'Accuracy':>10} {'Complete':>10} {'Relevance':>10}")
        print(f"  {'─'*3} {'─'*22} {'─'*14} {'─'*10} {'─'*10} {'─'*10}")
        for i, e in enumerate(answer_runs):
            m = e["metrics"]
            print(f"  {i+1:<3} {e['timestamp']:<22} {e['version']:<14} "
                  f"{m.get('accuracy',0):>9.2f}/5 {m.get('completeness',0):>9.2f}/5 {m.get('relevance',0):>9.2f}/5")
        print()


if __name__ == "__main__":
    print_history()
