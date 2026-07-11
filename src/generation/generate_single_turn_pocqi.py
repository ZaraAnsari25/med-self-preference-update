"""Generate single-turn SPECIALIST answers for the Real-POCQi dataset.

Real-POCQi (jjfenglab/Real-POCQi) is a point-of-care clinical Q&A set: each row is a
real question a generalist physician asked, tagged with a `specialty`. Here we treat
the question as coming from a generalist and have each model answer it AS A SPECIALIST
in that question's specialty -- a "curbside consult" framing. One question -> one answer
per model (single-turn), which is the shape self-preference judging then scores.

This is an ADDITIVE new script: it does not modify any existing generator/evaluator, and
it reuses the hardened client factory from generate_conversations.py (get_client), so it
inherits thinking-OFF (gpt reasoning_effort=none, gemini thinkingBudget=0, qwen think:false),
the temperature guard, Gemini REST, Qwen-native, and retry/backoff for free.

Records are written in a DUAL schema so every downstream evaluator can read them:
  - MedDialog-style keys:      scenario_id, patient_query
  - question-bank-style keys:  question_id, question_text
  plus: specialty, generated_response, generator_model, id, timestamp.

Usage:
    python src/generation/generate_single_turn_pocqi.py \
        --models gpt-5.5 claude-sonnet-5 gemini-3.1-flash-lite qwen3.6-35b \
        --num_questions 125 --shuffle --seed 42 \
        --output_dir single_turn_zara_run/Generation
    # optional: --specialty "Cardiology,Neurology"   (restrict to specific specialties)
"""

import json
import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm

# Reuse the hardened client factory (no-think, temperature guard, Gemini REST, Qwen
# native, retry) from the multi-turn generator. Import-only: does not change its behavior.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.generation.generate_conversations import get_client, LLMClient  # noqa: E402


SPECIALIST_SYSTEM_PROMPT = """You are an experienced {specialty} specialist responding to a consult question from a generalist physician at the point of care.

Provide specific, evidence-based clinical guidance:
- Give a clear, actionable answer to the exact question asked.
- State concise clinical reasoning for your recommendation.
- Note key caveats, contraindications, monitoring, or when to escalate/refer.
- Be precise and specific; avoid generic boilerplate.
- Do not invent patient details, labs, or findings that were not provided."""


SPECIALIST_USER_PROMPT = """A generalist physician asks you, the {specialty} specialist, the following clinical question:

{question}

Provide your specialist response."""


def load_realpocqi_questions(
    num_questions: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = False,
    specialties: Optional[List[str]] = None,
) -> List[Dict]:
    """Load Real-POCQi questions, optionally filtered by specialty, then sampled.

    Returns dicts with: question_id, question_text, specialty.
    """
    print("Loading Real-POCQi questions...")
    ds = load_dataset("jjfenglab/Real-POCQi", data_files="questions.parquet", split="train")

    if specialties:
        wanted = {s.strip().lower() for s in specialties}
        ds = ds.filter(lambda r: (r.get("specialty") or "").strip().lower() in wanted)
        print(f"  filtered to specialties {sorted(wanted)}: {len(ds)} questions")

    if shuffle:
        ds = ds.shuffle(seed=seed)

    rows = []
    for i, item in enumerate(ds):
        if num_questions is not None and i >= num_questions:
            break
        rows.append({
            "question_id": item["question_id"],
            "question_text": item["question_text"],
            "specialty": item.get("specialty", "Medicine"),
        })
    print(f"Loaded {len(rows)} questions")
    return rows


async def generate_one_answer(
    question: Dict,
    client: LLMClient,
    temperature: float,
    max_tokens: int,
) -> Optional[Dict]:
    """Generate one specialist answer for a question; return a dual-schema record."""
    specialty = question["specialty"]
    system = SPECIALIST_SYSTEM_PROMPT.format(specialty=specialty)
    user = SPECIALIST_USER_PROMPT.format(specialty=specialty, question=question["question_text"])
    response = await client.generate(system, user, temperature=temperature, max_tokens=max_tokens)
    response = (response or "").strip()
    if not response:
        return None
    qid = question["question_id"]
    return {
        "id": f"{qid}_{client.model_name}",
        # dual schema so single-turn (scenario_id) and question-bank (question_id)
        # evaluators both read these records unchanged:
        "scenario_id": qid,
        "question_id": qid,
        "patient_query": question["question_text"],
        "question_text": question["question_text"],
        "specialty": specialty,
        "generated_response": response,
        "generator_model": client.model_name,
        "timestamp": datetime.now().isoformat(),
    }


async def generate_for_model(
    questions: List[Dict],
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_concurrency: int,
) -> List[Dict]:
    """Generate answers for all questions with one model, up to max_concurrency at once."""
    client = get_client(model_name)
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(idx: int, q: Dict):
        async with sem:
            try:
                return idx, await generate_one_answer(q, client, temperature, max_tokens)
            except Exception as e:
                print(f"Error on {q['question_id']} with {model_name}: {e}")
                return idx, None

    tasks = [asyncio.ensure_future(_one(i, q)) for i, q in enumerate(questions)]
    by_idx: Dict[int, Optional[Dict]] = {}
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=model_name):
        idx, rec = await fut
        by_idx[idx] = rec
    return [by_idx[i] for i in range(len(questions)) if by_idx.get(i) is not None]


async def main():
    parser = argparse.ArgumentParser(description="Generate single-turn specialist answers for Real-POCQi.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Specialist models (each answers every question). e.g. gpt-5.5 claude-sonnet-5 ...")
    parser.add_argument("--num_questions", type=int, default=None,
                        help="Cap on number of questions (default: all in the filtered set)")
    parser.add_argument("--specialty", type=str, default=None,
                        help="Comma-separated specialties to keep (e.g. 'Cardiology,Neurology'). Default: all.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for question sampling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking --num_questions")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (guarded per model)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per answer")
    parser.add_argument("--output_dir", type=str, default="single_turn_zara_run/Generation")
    parser.add_argument("--max_concurrency", type=int, default=12,
                        help="Max answers generated concurrently (cloud fans out; Qwen bounded by OLLAMA_NUM_PARALLEL)")
    args = parser.parse_args()

    # Enlarge the thread pool so blocking to_thread clients (Gemini REST, Qwen native)
    # actually run max_concurrency at once.
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=max(32, args.max_concurrency * 2))
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specialties = args.specialty.split(",") if args.specialty else None
    questions = load_realpocqi_questions(
        num_questions=args.num_questions, seed=args.seed, shuffle=args.shuffle, specialties=specialties,
    )
    # Save the question set (with specialties) for reference / evaluators.
    with open(out_dir / "questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    all_records = []
    for model_name in args.models:
        print(f"\nGenerating {len(questions)} specialist answers with {model_name} "
              f"(up to {args.max_concurrency} concurrent)")
        records = await generate_for_model(
            questions, model_name, args.temperature, args.max_tokens, args.max_concurrency,
        )
        out_path = out_dir / f"{model_name}_pocqi_responses.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} responses to {out_path}")
        all_records.extend(records)

    with open(out_dir / "all_pocqi_responses.json", "w") as f:
        json.dump(all_records, f, indent=2)

    print("\nGeneration complete")
    print(f"  Questions: {len(questions)} | Models: {args.models}")
    print(f"  Total responses: {len(all_records)} | Output: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
