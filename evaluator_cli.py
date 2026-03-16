import argparse
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(override=True)

# Route evaluation through answer (reranking + query rewriting + 3-large)
import implementation.answer as answer
sys.modules["implementation.answer"] = answer

from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers
from logs import log_run


# ── ANSI colours ──────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def color(value: float, green_thresh: float, amber_thresh: float, fmt: str) -> str:
    text = fmt.format(value)
    if value >= green_thresh:
        return f"{GREEN}{text}{RESET}"
    elif value >= amber_thresh:
        return f"{YELLOW}{text}{RESET}"
    else:
        return f"{RED}{text}{RESET}"


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def run_retrieval(category_filter: str | None = None):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  RETRIEVAL EVALUATION{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")

    total_mrr = total_ndcg = total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, progress in evaluate_all_retrieval():
        if category_filter and test.category != category_filter:
            continue

        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)

        bar_len = int(progress * 30)
        bar = f"[{'█' * bar_len}{'░' * (30 - bar_len)}]"
        mrr_str = color(result.mrr, 0.9, 0.75, "{:.2f}")
        cov_str = color(result.keyword_coverage, 90.0, 75.0, "{:.0f}%")
        print(f"\r  {bar} {count:>3}  MRR {mrr_str}  cov {cov_str}  {test.category:<14}", end="", flush=True)

    if count == 0:
        print(f"\n  No tests matched category '{category_filter}'.")
        return

    avg_mrr      = total_mrr      / count
    avg_ndcg     = total_ndcg     / count
    avg_coverage = total_coverage / count

    print(f"\n\n{BOLD}  Summary ({count} tests){RESET}")
    print(f"  MRR:              {color(avg_mrr,      0.9,  0.75, '{:.4f}')}")
    print(f"  nDCG:             {color(avg_ndcg,     0.9,  0.75, '{:.4f}')}")
    print(f"  Keyword Coverage: {color(avg_coverage, 90.0, 75.0, '{:.1f}%')}")

    print(f"\n{BOLD}  By category{RESET}")
    by_cat = {}
    for cat, scores in sorted(category_mrr.items()):
        avg = sum(scores) / len(scores)
        by_cat[cat] = round(avg, 4)
        print(f"  {cat:<18} MRR {color(avg, 0.9, 0.75, '{:.4f}')}  ({len(scores)} tests)")

    log_run("v2", "retrieval", {"mrr": round(avg_mrr, 4), "ndcg": round(avg_ndcg, 4), "coverage": round(avg_coverage, 1)}, by_cat)
    print(f"\n  {BOLD}Logged.{RESET}")


# ── Answer evaluation ─────────────────────────────────────────────────────────

def run_answer(category_filter: str | None = None):
    model_note = " (gpt-4.1)"

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  ANSWER EVALUATION{model_note}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")

    total_acc = total_comp = total_rel = 0.0
    category_acc = defaultdict(list)
    count = 0

    for test, result, progress in evaluate_all_answers():
        if category_filter and test.category != category_filter:
            continue

        count += 1
        total_acc  += result.accuracy
        total_comp += result.completeness
        total_rel  += result.relevance
        category_acc[test.category].append(result.accuracy)

        bar_len = int(progress * 30)
        bar = f"[{'█' * bar_len}{'░' * (30 - bar_len)}]"
        acc_str = color(result.accuracy, 4.5, 4.0, "{:.1f}")
        print(f"\r  {bar} {count:>3}  acc {acc_str}/5  {test.category:<14}", end="", flush=True)

    if count == 0:
        print(f"\n  No tests matched category '{category_filter}'.")
        return

    avg_acc  = total_acc  / count
    avg_comp = total_comp / count
    avg_rel  = total_rel  / count

    print(f"\n\n{BOLD}  Summary ({count} tests){RESET}")
    print(f"  Accuracy:     {color(avg_acc,  4.5, 4.0, '{:.2f}/5')}")
    print(f"  Completeness: {color(avg_comp, 4.5, 4.0, '{:.2f}/5')}")
    print(f"  Relevance:    {color(avg_rel,  4.5, 4.0, '{:.2f}/5')}")

    print(f"\n{BOLD}  By category{RESET}")
    by_cat = {}
    for cat, scores in sorted(category_acc.items()):
        avg = sum(scores) / len(scores)
        by_cat[cat] = round(avg, 4)
        print(f"  {cat:<18} acc {color(avg, 4.5, 4.0, '{:.4f}')}  ({len(scores)} tests)")

    log_run("v2", "answer", {"accuracy": round(avg_acc, 4), "completeness": round(avg_comp, 4), "relevance": round(avg_rel, 4)}, by_cat)
    print(f"\n  {BOLD}Logged.{RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLI evaluator for the Insurellm RAG system (v2)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", action="store_true", help="Run retrieval evaluation only")
    group.add_argument("-a", action="store_true", help="Run answer evaluation only")
    parser.add_argument("-c", metavar="CAT", help="Filter to a specific category")
    args = parser.parse_args()

    run_both = not args.r and not args.a

    if args.r or run_both:
        run_retrieval(category_filter=args.c)

    if args.a or run_both:
        run_answer(category_filter=args.c)

    print()


if __name__ == "__main__":
    main()
