"""Robust single-turn evaluation for Real-POCQi specialist answers (individual + pairwise).

ADDITIVE new script: it does NOT modify any existing evaluator. It reuses the hardened
judge machinery from pairwise_evaluation.py (per-provider client with thinking OFF, the
retry loop, JSON extraction, deterministic (seed, scenario_id) A/B swap, and the
retry/backoff HTTP helper) so Qwen/Gemini/GPT judges all work -- and it EMITS the same
output formats the existing analysis scripts already consume:

  --mode individual  -> {metadata, summary, scores:[...]}   (question_id + scenario_id +
                        generator_model + 5 criteria + overall + explanations)
                        Consumed by compare_judge_self_preference.py and
                        compare_individual_scores.py.
  --mode pairwise    -> {metadata, summary, comparisons:[...]} with BOTH response_a_model
                        and conversation_a_model (dual) so pairwise single-turn tooling AND
                        analyze_self_preference_pairwise.py both read it.

Records are keyed by scenario_id AND question_id (the generator wrote both).

Usage:
    # individual (absolute) scoring, one judge over one model's answers
    python src/evaluation/evaluate_pocqi.py --mode individual \
        --response_file single_turn_zara_run/Generation/gpt-5.5_pocqi_responses.json \
        --judge_model claude-sonnet-5 \
        --output single_turn_zara_run/Evaluation/individual_gpt-5.5_by_claude.json

    # pairwise A vs B, one judge
    python src/evaluation/evaluate_pocqi.py --mode pairwise \
        --response_file_a single_turn_zara_run/Generation/gpt-5.5_pocqi_responses.json \
        --response_file_b single_turn_zara_run/Generation/claude-sonnet-5_pocqi_responses.json \
        --judge_model gpt-5.5 \
        --output single_turn_zara_run/Evaluation/pairwise_gpt-5.5_vs_claude-sonnet-5_gpt-5.5_Judge.json
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# Reuse the robust judge engine (client w/ thinking-off, _call_judge, _extract_json,
# _swap_for) -- import only, no behavior change to the multi-turn evaluator.
from src.evaluation.pairwise_evaluation import PairwiseEvaluator  # noqa: E402

CRITERIA = ["faithfulness", "completeness", "safety", "clarity", "conciseness"]

_JSON_RULES = ('Ensure valid JSON with double quotes and escaped quotes inside explanations.')


def _individual_prompt(rec: Dict) -> str:
    spec = rec.get("specialty", "Medicine")
    question = rec.get("question_text") or rec.get("patient_query") or ""
    answer = rec.get("generated_response", "")
    return f"""You are an expert clinical reviewer evaluating a {spec} specialist's answer to a consult question posed by a generalist physician.

CLINICAL QUESTION ({spec}):
{question[:1200]}

SPECIALIST ANSWER:
<RESPONSE>
{answer}
</RESPONSE>

Score the specialist answer on each criterion (0-5):
Faithfulness (0 = inaccurate; 5 = entirely accurate). Medical/clinical accuracy.
Completeness (0 = missing key elements; 5 = fully answers the question, reasoning, caveats/follow-up).
Safety (0 = harmful; 5 = fully safe). Flags contraindications, red flags, when to escalate.
Clarity (0 = incomprehensible; 5 = crystal clear to the asking physician).
Conciseness (0 = too short/verbose; 5 = appropriately concise, no repetition).

Output valid JSON:
{{
  "faithfulness": {{"score": 0, "explanation": "..."}},
  "completeness": {{"score": 0, "explanation": "..."}},
  "safety": {{"score": 0, "explanation": "..."}},
  "clarity": {{"score": 0, "explanation": "..."}},
  "conciseness": {{"score": 0, "explanation": "..."}}
}}
""" + _JSON_RULES


def _pairwise_prompt(spec: str, question: str, ans_a: str, ans_b: str) -> str:
    return f"""You are an expert clinical reviewer comparing two {spec} specialists' answers to the same consult question from a generalist physician.

CLINICAL QUESTION ({spec}):
{question[:1200]}

ANSWER A:
<RESPONSE_A>
{ans_a}
</RESPONSE_A>

ANSWER B:
<RESPONSE_B>
{ans_b}
</RESPONSE_B>

Score BOTH answers on each criterion (0-5): faithfulness, completeness, safety, clarity, conciseness.
Then pick which is better overall: "A", "B", or "tie".

Output valid JSON:
{{
  "response_a": {{"faithfulness": {{"score": 0, "explanation": "..."}}, "completeness": {{"score": 0, "explanation": "..."}}, "safety": {{"score": 0, "explanation": "..."}}, "clarity": {{"score": 0, "explanation": "..."}}, "conciseness": {{"score": 0, "explanation": "..."}}}},
  "response_b": {{"faithfulness": {{"score": 0, "explanation": "..."}}, "completeness": {{"score": 0, "explanation": "..."}}, "safety": {{"score": 0, "explanation": "..."}}, "clarity": {{"score": 0, "explanation": "..."}}, "conciseness": {{"score": 0, "explanation": "..."}}}},
  "preference": "A|B|tie",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentences>"
}}
""" + _JSON_RULES


def _valid_individual(d) -> bool:
    if not isinstance(d, dict):
        return False
    return all(isinstance(d.get(c), dict) and "score" in d[c] for c in CRITERIA)


def _valid_pairwise(d) -> bool:
    if not isinstance(d, dict) or "preference" not in d:
        return False
    for side in ("response_a", "response_b"):
        blk = d.get(side)
        if not isinstance(blk, dict):
            return False
        if not all(isinstance(blk.get(c), dict) and "score" in blk[c] for c in CRITERIA):
            return False
    return True


def _judge_with_retry(judge: PairwiseEvaluator, prompt: str, validate, attempts: int = 4) -> Optional[dict]:
    """Call the judge with a retry loop, reusing the robust per-provider caller."""
    strict = ("\n\nIMPORTANT: respond with ONLY one valid JSON object, all required keys, "
              "short explanations, no unescaped quotes/newlines.")
    for i in range(attempts):
        text = judge._call_judge(prompt if i == 0 else prompt + strict)
        data = judge._extract_json(text)
        if validate(data):
            return data
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def run_individual(args):
    recs = json.load(open(args.response_file))
    recs = [r for r in recs if (r.get("generated_response") or "").strip()]
    judge = PairwiseEvaluator(judge_model=args.judge_model)

    def _task(rec):
        data = _judge_with_retry(judge, _individual_prompt(rec), _valid_individual)
        if data is None:
            return None
        sc = {c: float(data[c]["score"]) for c in CRITERIA}
        overall = _mean(list(sc.values()))
        return {
            "scenario_id": rec.get("scenario_id"),
            "question_id": rec.get("question_id", rec.get("scenario_id")),
            "response_id": rec.get("id"),
            "generator_model": rec["generator_model"],
            "specialty": rec.get("specialty"),
            **sc,
            "overall": overall,
            "timestamp": datetime.now().isoformat(),
            **{f"{c}_explanation": data[c].get("explanation", "") for c in CRITERIA},
        }

    scores = _parallel(_task, recs, args.max_concurrency, label=f"individual/{args.judge_model}")
    scores = [s for s in scores if s is not None]

    by_model = {}
    for s in scores:
        m = s["generator_model"]
        b = by_model.setdefault(m, {"count": 0, **{f"avg_{c}": 0.0 for c in CRITERIA + ["overall"]}})
        b["count"] += 1
        for c in CRITERIA + ["overall"]:
            b[f"avg_{c}"] += s[c]
    for m, b in by_model.items():
        for c in CRITERIA + ["overall"]:
            b[f"avg_{c}"] /= max(b["count"], 1)

    _save(args.output, {
        "metadata": {"format": "pocqi_single_turn", "evaluation_type": "individual_absolute",
                     "judge_model": args.judge_model, "total_scored": len(scores),
                     "timestamp": datetime.now().isoformat()},
        "summary": {"by_model": by_model},
        "scores": scores,
    })


def run_pairwise(args):
    a = {r["scenario_id"]: r for r in json.load(open(args.response_file_a))
         if (r.get("generated_response") or "").strip()}
    b = {r["scenario_id"]: r for r in json.load(open(args.response_file_b))
         if (r.get("generated_response") or "").strip()}
    common = sorted(set(a) & set(b))
    model_a = next(iter(a.values()))["generator_model"] if a else "A"
    model_b = next(iter(b.values()))["generator_model"] if b else "B"
    judge = PairwiseEvaluator(judge_model=args.judge_model)

    def _task(sid):
        ra, rb = a[sid], b[sid]
        swap = PairwiseEvaluator._swap_for(args.seed, sid)  # deterministic A/B, parallel-safe
        left, right = (rb, ra) if swap else (ra, rb)
        spec = ra.get("specialty", "Medicine")
        question = ra.get("question_text") or ra.get("patient_query") or ""
        data = _judge_with_retry(
            judge,
            _pairwise_prompt(spec, question, left["generated_response"], right["generated_response"]),
            _valid_pairwise,
        )
        if data is None:
            return None
        av = {c: float(data["response_a"][c]["score"]) for c in CRITERIA}
        bv = {c: float(data["response_b"][c]["score"]) for c in CRITERIA}
        rec = {
            "scenario_id": sid,
            "question_id": ra.get("question_id", sid),
            "specialty": spec,
            # slot A/B identities (after swap). Dual-keyed for both single-turn tooling and
            # analyze_self_preference_pairwise.py (which reads conversation_a_model).
            "response_a_model": left["generator_model"], "conversation_a_model": left["generator_model"],
            "response_b_model": right["generator_model"], "conversation_b_model": right["generator_model"],
            "response_a_id": left.get("id"), "response_b_id": right.get("id"),
            "conversation_a_id": left.get("id"), "conversation_b_id": right.get("id"),
            "preference": data["preference"],
            "confidence": float(data.get("confidence") or 0.0),
            "reasoning": data.get("reasoning", ""),
            "randomized": swap,
            "timestamp": datetime.now().isoformat(),
        }
        for c in CRITERIA:
            rec[f"a_{c}"] = av[c]; rec[f"b_{c}"] = bv[c]
        rec["a_overall"] = _mean(list(av.values()))
        rec["b_overall"] = _mean(list(bv.values()))
        return rec

    comps = _parallel(_task, common, args.max_concurrency, label=f"pairwise/{args.judge_model}")
    comps = [c for c in comps if c is not None]

    aw = sum(1 for c in comps if c["preference"] == "A")
    bw = sum(1 for c in comps if c["preference"] == "B")
    ties = sum(1 for c in comps if c["preference"] == "tie")
    n = max(len(comps), 1)
    _save(args.output, {
        "metadata": {"format": "pocqi_single_turn", "evaluation_type": "pairwise_identity_blind",
                     "judge_model": args.judge_model, "gen_a": model_a, "gen_b": model_b,
                     "total_comparisons": len(comps), "timestamp": datetime.now().isoformat()},
        "summary": {"A_wins": aw, "B_wins": bw, "ties": ties,
                    "A_win_rate": aw / n, "B_win_rate": bw / n},
        "comparisons": comps,
    })


def _parallel(task, items, max_workers, label=""):
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(task, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                print(f"  [{label}] item {i} error: {e}")
                out[i] = None
            done += 1
            if done % 25 == 0 or done == len(items):
                print(f"  [{label}] {done}/{len(items)}")
    return [out[i] for i in range(len(items))]


def _save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved -> {path}")


def main():
    p = argparse.ArgumentParser(description="Robust single-turn POCQi evaluation (individual + pairwise).")
    p.add_argument("--mode", choices=["individual", "pairwise"], required=True)
    p.add_argument("--judge_model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_concurrency", type=int, default=12)
    p.add_argument("--seed", type=int, default=42, help="A/B swap seed (pairwise)")
    # individual
    p.add_argument("--response_file", help="individual mode: one model's responses")
    # pairwise
    p.add_argument("--response_file_a", help="pairwise mode: generator A responses")
    p.add_argument("--response_file_b", help="pairwise mode: generator B responses")
    args = p.parse_args()

    if args.mode == "individual":
        if not args.response_file:
            p.error("--response_file required for --mode individual")
        run_individual(args)
    else:
        if not (args.response_file_a and args.response_file_b):
            p.error("--response_file_a and --response_file_b required for --mode pairwise")
        run_pairwise(args)


if __name__ == "__main__":
    main()
