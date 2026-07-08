"""
Truncate long multi-turn conversations into shorter ones.

Because generation does NOT condition on the total turn count (each turn is produced
from the conversation-so-far plus a fixed system prompt), the first 2k turns of an
8-turn conversation are a valid k-turn conversation. So you can generate ONCE at the
longest length (e.g. 8t) and derive 2t/4t/6t by truncation -- cheaper, and giving a
clean nested design (all lengths share the same trajectory).

For each source record this:
  - slices turns[:k]
  - rewrites conversation_id -> {scenario_id}_{generator_model}_{k}t
  - sets total_turns = k and generation_params.num_turns = k
and writes {generator_model}_{k}t_conversations.json (matching the generator's naming).

Even target lengths (2,4,6,8) end on a physician turn, which is what you evaluate.

Usage:
    python src/generation/truncate_conversations.py \
        --input_files multi_turn_zara_run/Generation/gpt-5.5_8t_conversations.json \
                      multi_turn_zara_run/Generation/claude-sonnet-5_8t_conversations.json \
        --targets 2 4 6 \
        --output_dir multi_turn_zara_run/Generation
"""

import json
import argparse
import copy
from pathlib import Path


def truncate_record(rec: dict, k: int) -> dict:
    """Return a copy of `rec` truncated to the first k turns (or None if too short)."""
    turns = rec.get("turns", [])
    if len(turns) < k:
        return None
    out = copy.deepcopy(rec)
    out["turns"] = turns[:k]
    out["total_turns"] = k
    if isinstance(out.get("generation_params"), dict):
        out["generation_params"]["num_turns"] = k
    # Rebuild the id in the generator's canonical format (model name may contain
    # dots/hyphens, so rebuild from fields rather than string-splitting the suffix).
    sid = rec.get("scenario_id", "unknown")
    model = rec.get("generator_model", "model")
    out["conversation_id"] = f"{sid}_{model}_{k}t"
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Truncate long conversation files into shorter turn counts."
    )
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Source conversation JSON files (e.g. the 8t files).")
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 4, 6],
                        help="Target turn counts to produce (default: 2 4 6).")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write outputs (default: same dir as each input file).")
    args = parser.parse_args()

    for infile in args.input_files:
        inpath = Path(infile)
        with open(inpath) as f:
            records = json.load(f)
        if not records:
            print(f"[skip] {infile}: empty")
            continue

        model = records[0].get("generator_model", inpath.stem)
        src_turns = max((r.get("total_turns", len(r.get("turns", []))) for r in records), default=0)
        out_dir = Path(args.output_dir) if args.output_dir else inpath.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{infile}  ({len(records)} records, source ~{src_turns}t, model={model})")

        for k in args.targets:
            if k >= src_turns:
                print(f"  target {k}t: skipped (>= source {src_turns}t)")
                continue
            truncated = [truncate_record(r, k) for r in records]
            kept = [r for r in truncated if r is not None]
            dropped = len(truncated) - len(kept)
            outpath = out_dir / f"{model}_{k}t_conversations.json"
            with open(outpath, "w") as f:
                json.dump(kept, f, indent=2)
            note = f" ({dropped} dropped: fewer than {k} turns)" if dropped else ""
            print(f"  target {k}t: wrote {len(kept)} -> {outpath}{note}")


if __name__ == "__main__":
    main()
