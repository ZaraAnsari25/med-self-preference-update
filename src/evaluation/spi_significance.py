"""Per-model self-preference significance for a pairwise cell.

ADDITIVE: leaves analyze_self_preference_pairwise.py untouched; reuses its loader
(read-only) to get, per scenario, each generator's score from each judge.

Self-preference is reported PER MODEL (not summed into one index -- summing hides
asymmetry/cancellation). For each of the two models in the pair, and each rubric
criterion, the per-scenario difference

    d_i = own_family_judge_score(model, i) - other_family_judge_score(model, i)

(same answers, different judge) is a paired sample whose mean is the model's
self-preference bias. We report bias, 95% CI (Student-t) and a one-sample t-test p
(with a bootstrap CI kept in the JSON as a distribution-free robustness check).
"Significant" = 95% CI excludes 0. No multiple-comparison adjustment is applied.

Usage:
    python src/evaluation/spi_significance.py \
        --judge_files pairwise_A_vs_B_A_Judge.json pairwise_A_vs_B_B_Judge.json \
        [--output report.json]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.analyze_self_preference_pairwise import (  # noqa: E402
    load_judge_file, model_family, CRITERIA,
)


def _stats(diffs, n_boot, rng):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    bias = float(d.mean()) if n else float("nan")
    if n >= 2:
        se = float(d.std(ddof=1) / np.sqrt(n))
        tc = float(stats.t.ppf(0.975, n - 1))
        ci = (bias - tc * se, bias + tc * se)
        p = float(stats.ttest_1samp(d, 0.0).pvalue)
        boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
        b_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    else:
        se = float("nan"); ci = (float("nan"),) * 2; p = float("nan"); b_ci = ci
    sig = (n >= 2) and (ci[0] > 0 or ci[1] < 0)
    return {"n": n, "bias": bias, "se": se, "ci95": list(ci), "p_ttest": p,
            "ci95_boot": list(b_ci), "significant_0.05": bool(sig)}


def compute(judge_files, n_boot=10000, seed=42):
    jm1, sc1 = load_judge_file(judge_files[0])
    jm2, sc2 = load_judge_file(judge_files[1])
    fam1, fam2 = model_family(jm1), model_family(jm2)
    if fam1 == fam2:
        raise SystemExit(f"Both judges are family '{fam1}'; own-vs-other is undefined.")
    judges = {jm1: (fam1, sc1), jm2: (fam2, sc2)}
    paired = sorted(set(sc1) & set(sc2))
    generators = sorted({g for sid in paired for g in sc1[sid]}
                        | {g for sid in paired for g in sc2[sid]})
    rng = np.random.default_rng(seed)

    out = {"judges": {jm1: fam1, jm2: fam2}, "n_paired": len(paired), "per_model": {}}
    for g in generators:
        gf = model_family(g)
        own = next((jm for jm, (f, _) in judges.items() if f == gf), None)
        other = next((jm for jm, (f, _) in judges.items() if f != gf), None)
        if own is None or other is None:
            continue
        os_, ot_ = judges[own][1], judges[other][1]
        out["per_model"][g] = {"own_judge": own, "other_judge": other, "by_criterion": {}}
        for crit in CRITERIA:
            diffs = [os_[s][g][crit] - ot_[s][g][crit] for s in paired
                     if g in os_.get(s, {}) and g in ot_.get(s, {})]
            out["per_model"][g]["by_criterion"][crit] = _stats(diffs, n_boot, rng)
    return jm1, jm2, out


def main():
    ap = argparse.ArgumentParser(description="Per-model self-preference bias: 95% CI + p.")
    ap.add_argument("--judge_files", nargs=2, required=True, metavar=("JUDGE_1", "JUDGE_2"))
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    jm1, jm2, out = compute(args.judge_files, n_boot=args.n_boot, seed=args.seed)
    print(f"\n{'='*84}")
    print(f"SELF-PREFERENCE BIAS (per model)  judges: {jm1} vs {jm2}   n={out['n_paired']}")
    print("  bias = own-judge − other-judge on the model's OWN answers; + = favors own. (overall)")
    print("  significant = 95% CI excludes 0 (no multiple-comparison adjustment)")
    print(f"{'='*84}")
    print(f"  {'model':<22}{'other judge':<22}{'bias':>7}{'95% CI':>18}{'p':>9}  sig?")
    for g, r in out["per_model"].items():
        ov = r["by_criterion"]["overall"]
        ci = f"[{ov['ci95'][0]:+.2f},{ov['ci95'][1]:+.2f}]"
        print(f"  {g:<22}{r['other_judge']:<22}{ov['bias']:>+7.2f}{ci:>18}{ov['p_ttest']:>9.3f}"
              f"  {'YES' if ov['significant_0.05'] else 'no'}")
    print(f"{'='*84}  (full per-criterion detail in the JSON)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"judge_1": jm1, "judge_2": jm2, **out}, f, indent=2)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
