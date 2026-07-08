"""
Self-preference analysis for multi-turn (pipeline 1) pairwise evaluations.

Given the SAME conversation pairs judged by two different judges (e.g. a GPT judge
and a Claude judge), this measures whether a judge inflates the scores of its OWN
model family's generations relative to the other judge.

It reads two pairwise result files produced by pairwise_evaluation.py (one per
judge), restricts to the scenarios judged by BOTH judges, and reports:

  - Per-judge mean score for each generator model, per criterion.
  - Cross-judge bias per generator: (own-family judge) - (other-family judge).
      Positive => a model's own-family judge scores that model higher than the
      neutral other-family judge does  == self-preference for that model.
  - Self-Preference Index (SPI) per criterion = sum of the per-model
      self-preference biases. Positive SPI => net mutual self-preference.

Because a judge scores BOTH generators per comparison, "self" vs "cross" is
determined by matching the judge's model family to the generator's family
(gpt-*/o1-*/o3- => openai; claude* => anthropic).

Usage:
    python src/evaluation/analyze_self_preference_pairwise.py \
        --judge_files <gpt_judge.json> <claude_judge.json> \
        --output multi_turn_zara_run/Evaluation/self_preference_6t.json
"""

import json
import argparse
from pathlib import Path

CRITERIA = ["faithfulness", "completeness", "safety", "clarity", "conciseness", "overall"]


def model_family(model_name: str) -> str:
    """Map a model id to a provider family used to define 'self' vs 'cross'."""
    m = (model_name or "").lower()
    if m.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    if "qwen" in m:
        return "qwen"
    if "deepseek" in m:
        return "deepseek"
    return "other"


def load_judge_file(path: str):
    """Return (judge_model, {scenario_id: {generator_model: {criterion: score}}})."""
    with open(path) as f:
        d = json.load(f)
    judge_model = d["metadata"]["judge_model"]
    scores = {}
    for c in d["comparisons"]:
        sid = c["scenario_id"]
        entry = scores.setdefault(sid, {})
        # A/B order was randomized per comparison; map each side back to its real
        # generator model so scores are keyed by generator, not by A/B slot.
        for side in ("a", "b"):
            gen = c[f"conversation_{side}_model"]
            entry[gen] = {crit: float(c[f"{side}_{crit}"]) for crit in CRITERIA}
    return judge_model, scores


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Measure self-preference between two judges over the same pairwise comparisons."
    )
    parser.add_argument(
        "--judge_files", nargs=2, required=True, metavar=("JUDGE_1", "JUDGE_2"),
        help="Two pairwise result JSONs (one per judge) over the SAME generators/turns.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save the JSON report.")
    args = parser.parse_args()

    judge1, scores1 = load_judge_file(args.judge_files[0])
    judge2, scores2 = load_judge_file(args.judge_files[1])
    fam1, fam2 = model_family(judge1), model_family(judge2)

    if fam1 == fam2:
        print(f"WARNING: both judges are the same family ({fam1}); self-preference is undefined.")

    # Scenarios judged by BOTH judges.
    paired = sorted(set(scores1) & set(scores2))
    only1 = len(set(scores1) - set(scores2))
    only2 = len(set(scores2) - set(scores1))

    # Generators present across the paired set.
    generators = sorted({g for sid in paired for g in scores1[sid]}
                        | {g for sid in paired for g in scores2[sid]})

    judges = {judge1: (fam1, scores1), judge2: (fam2, scores2)}

    # Per-judge mean score for each generator, per criterion (over paired scenarios
    # where that judge scored that generator).
    per_judge_means = {}
    for jm, (fam, sc) in judges.items():
        per_judge_means[jm] = {}
        for gen in generators:
            per_judge_means[jm][gen] = {
                crit: mean([sc[sid][gen][crit] for sid in paired
                            if gen in sc[sid] and crit in sc[sid][gen]])
                for crit in CRITERIA
            }

    # Self-preference per generator = (own-family judge mean) - (other-family judge mean).
    self_pref = {}
    for gen in generators:
        gfam = model_family(gen)
        own = next((jm for jm, (fam, _) in judges.items() if fam == gfam), None)
        other = next((jm for jm, (fam, _) in judges.items() if fam != gfam), None)
        if own is None or other is None:
            continue
        self_pref[gen] = {
            "own_judge": own,
            "other_judge": other,
            "bias": {crit: per_judge_means[own][gen][crit] - per_judge_means[other][gen][crit]
                     for crit in CRITERIA},
        }

    # Self-Preference Index per criterion = sum of per-model self-preference biases.
    spi = {crit: sum(self_pref[g]["bias"][crit] for g in self_pref) for crit in CRITERIA}

    report = {
        "judges": {judge1: fam1, judge2: fam2},
        "generators": {g: model_family(g) for g in generators},
        "paired_scenarios": len(paired),
        "dropped": {f"only_{judge1}": only1, f"only_{judge2}": only2},
        "per_judge_mean_scores": per_judge_means,
        "self_preference_by_model": self_pref,
        "self_preference_index": spi,
    }

    # ---- Printed report ----
    print(f"\n{'='*72}\nSELF-PREFERENCE ANALYSIS (pipeline 1, pairwise)\n{'='*72}")
    print(f"Judge 1: {judge1}  [{fam1}]")
    print(f"Judge 2: {judge2}  [{fam2}]")
    print(f"Paired scenarios (judged by both): {len(paired)}"
          f"   (dropped: {only1} only-{judge1}, {only2} only-{judge2})")

    print(f"\nMEAN 'overall' SCORE BY GENERATOR (paired set):")
    print(f"  {'generator':<28}{judge1:<22}{judge2:<22}")
    for gen in generators:
        print(f"  {gen:<28}{per_judge_means[judge1][gen]['overall']:<22.3f}"
              f"{per_judge_means[judge2][gen]['overall']:<22.3f}")

    print(f"\nSELF-PREFERENCE BY MODEL  (own-family judge minus other judge; + = inflates own):")
    for gen, sp in self_pref.items():
        print(f"  {gen}  (own judge: {sp['own_judge']})")
        for crit in CRITERIA:
            print(f"      {crit:<14}{sp['bias'][crit]:+.3f}")

    print(f"\nSELF-PREFERENCE INDEX (SPI) per criterion  (+ = net mutual self-preference):")
    for crit in CRITERIA:
        print(f"  {crit:<14}{spi[crit]:+.3f}")
    print(f"{'='*72}\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
