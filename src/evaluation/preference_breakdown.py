"""Per-judge preference breakdown for a pairwise cell: how often each judge picked its
OWN family's answer vs the OTHER vs tie, with a binomial significance test.

Self-preference in the win/lose DECISIONS (complement to the score-based SPI). Reported
PER JUDGE (self-preference is a property of each judge) -- no pooled cross-judge average,
which would mask opposite-direction asymmetry.

For each judge, among its non-tie comparisons, we test H0: P(pick own) = 0.5 with an exact
binomial test. At the smoke's n it means nothing; at n=125 non-tie decisions it has power.

Usage:
    python src/evaluation/preference_breakdown.py \
        --judge_files pairwise_A_vs_B_A_Judge.json pairwise_A_vs_B_B_Judge.json \
        [--output report.json]
"""

import sys
import json
import argparse
from pathlib import Path

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.analyze_self_preference_pairwise import model_family  # noqa: E402


def breakdown_one(path):
    d = json.load(open(path))
    judge = d["metadata"]["judge_model"]
    jf = model_family(judge)
    own = other = tie = 0
    for c in d["comparisons"]:
        p = c.get("preference")
        if p == "tie":
            tie += 1
            continue
        winner = c["conversation_a_model"] if p == "A" else c["conversation_b_model"]
        if model_family(winner) == jf:
            own += 1
        else:
            other += 1
    n = own + other + tie
    n_dec = own + other  # non-tie decisions
    bt = binomtest(own, n_dec, 0.5, alternative="two-sided") if n_dec > 0 else None
    p_val = float(bt.pvalue) if bt else float("nan")
    return {
        "judge": judge, "judge_family": jf, "n": n,
        "own": own, "other": other, "tie": tie,
        "own_pct": (own / n * 100) if n else float("nan"),
        "other_pct": (other / n * 100) if n else float("nan"),
        "tie_pct": (tie / n * 100) if n else float("nan"),
        "own_rate_nontie": (own / n_dec) if n_dec else float("nan"),
        "p_binom_vs_0.5": p_val,
        "significant_0.05": bool(bt and p_val < 0.05 and own > other),
    }


def main():
    ap = argparse.ArgumentParser(description="Per-judge own/other/tie preference breakdown + binomial test.")
    ap.add_argument("--judge_files", nargs=2, required=True, metavar=("JUDGE_1", "JUDGE_2"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = [breakdown_one(f) for f in args.judge_files]

    print(f"\n{'='*82}")
    print("PER-JUDGE PREFERENCE  (own vs other vs tie; binomial test of own-rate vs 0.5)")
    print(f"{'='*82}")
    print(f"  {'judge':<24}{'n':>3}{'OWN':>7}{'OTHER':>8}{'TIE':>7}{'p(own=.5)':>11}   sig?")
    for r in rows:
        print(f"  {r['judge']:<24}{r['n']:>3}{r['own_pct']:>6.0f}%{r['other_pct']:>7.0f}%"
              f"{r['tie_pct']:>6.0f}%{r['p_binom_vs_0.5']:>11.3f}   {'YES' if r['significant_0.05'] else 'no'}")
    print(f"{'='*82}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"judges": rows}, f, indent=2)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
