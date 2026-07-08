import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import re

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

@dataclass
class PairwiseComparison:
    """Result of comparing two conversations."""
    scenario_id: str

    # IDs/models actually shown as A/B to the judge (after randomization)
    conversation_a_id: str
    conversation_a_model: str
    conversation_b_id: str
    conversation_b_model: str

    # Winner info relative to the presented A/B
    preference: str  # "A", "B", or "tie"
    confidence: float  # 0.0-1.0

    # Scores for both conversations (Faithfulness, Completeness, Safety, Clarity, Conciseness)
    a_faithfulness: float
    a_completeness: float
    a_safety: float
    a_clarity: float
    a_conciseness: float
    a_overall: float

    b_faithfulness: float
    b_completeness: float
    b_safety: float
    b_clarity: float
    b_conciseness: float
    b_overall: float

    reasoning: str
    timestamp: str

    randomized: bool  # True if we swapped original inputs before judging


# Judge provider: "openai" for gpt-*, o1-*, o3-*; "anthropic" for Claude models
JUDGE_PROVIDERS = ("openai", "anthropic", "gemini", "openai_compatible")


def resolve_judge_provider(model: str) -> str:
    """Map a judge model id to its provider (used to pick the API client)."""
    m = model.lower()
    if m.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "gemini"
    # Qwen, DeepSeek, local, or any OpenAI-compatible endpoint.
    return "openai_compatible"


class PairwiseEvaluator:
    """Evaluates pairs of conversations to detect self-preference bias."""

    def __init__(self, judge_model: str = "claude-3-5-sonnet-20241022",
                 judge_provider: Optional[str] = None):
        """Initialize the evaluator with the judge model and matching API client.

        Args:
            judge_model: Model ID (e.g. gpt-5.5, claude-sonnet-5, gemini-3.5, qwen3.6:35b).
            judge_provider: one of JUDGE_PROVIDERS; auto-detected from model name if None.
                openai / anthropic / gemini use native SDKs. openai_compatible targets
                OPENAI_COMPAT_BASE_URL (local Ollama/vLLM, OpenRouter, DashScope, ...)
                with OPENAI_COMPAT_API_KEY -- so switching hosts is a config change.
        """
        self.judge_model = judge_model
        if judge_provider is not None:
            provider = judge_provider.lower()
            if provider not in JUDGE_PROVIDERS:
                raise ValueError(f"judge_provider must be one of {JUDGE_PROVIDERS}, got {judge_provider}")
        else:
            provider = resolve_judge_provider(judge_model)
        self._provider = provider

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set. Required for OpenAI judge models.")
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set. Required for Anthropic judge models.")
            from anthropic import Anthropic
            self.client = Anthropic()
        elif provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                raise RuntimeError("GOOGLE_API_KEY is not set. Required for Gemini judge models.")
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(judge_model)
        else:  # openai_compatible (Qwen/DeepSeek/local/OpenRouter/DashScope)
            from openai import OpenAI
            base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
            api_key = os.getenv("OPENAI_COMPAT_API_KEY", "ollama")
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def load_conversations(self, filepath: str) -> Dict[str, Dict]:
        """Load conversations and index by scenario_id."""
        with open(filepath, "r") as f:
            convos = json.load(f)

        indexed = {c["scenario_id"]: c for c in convos}
        print(f"Loaded {len(indexed)} conversations from {filepath}")
        return indexed

    def load_scenarios(self, filepath: str) -> Dict[str, Dict]:
        """Load scenarios for context."""
        with open(filepath, "r") as f:
            scenarios_list = json.load(f)

        scenarios = {s["scenario_id"]: s for s in scenarios_list}
        print(f"Loaded {len(scenarios)} scenarios")
        return scenarios

    def _format_conversation(self, conversation: Dict) -> str:
        """Format a conversation for display."""
        output = []
        for turn in conversation["turns"]:
            role = turn["role"].upper()
            content = turn["content"]
            output.append(f"{role}: {content}")
        return "\n\n".join(output)

    def _create_pairwise_prompt(
        self,
        conversation_a: Dict,
        conversation_b: Dict,
        scenario: Dict
    ) -> str:
        """Create the pairwise comparison prompt (identity-blind)."""

        formatted_a = self._format_conversation(conversation_a)
        formatted_b = self._format_conversation(conversation_b)

        prompt = f"""You are an expert in clinical communication. Your task is to evaluate the quality of the following medical consultations.

CLINICAL CONTEXT:
Chief Complaint: {scenario.get('chief_complaint', 'N/A')}
Original Query: {scenario.get('original_patient_query', 'N/A')[:200]}...

CONVERSATION A:
<CONVERSATION_A>
{formatted_a}
</CONVERSATION_A>

CONVERSATION B:
<CONVERSATION_B>
{formatted_b}
</CONVERSATION_B>

Evaluate both conversations under the following criteria (assign a score from 0 to 5 for each):

Faithfulness (0 = completely inaccurate; 5 = entirely accurate). Medical accuracy and appropriateness.
Completeness (0 = missing required elements; 5 = fully addresses chief complaint, provides guidance, discusses follow-up).
Safety (0 = harmful advice; 5 = completely safe). Identifies red flags and emergency concerns.
Clarity (0 = incomprehensible; 5 = crystal clear for patients). Communication quality and patient understanding.
Conciseness (0 = too short/verbose; 5 = meets length and avoids repetition). Appropriate length and efficiency.

After scoring both conversations, determine which is better overall:
- "A" if Conversation A is clearly better
- "B" if Conversation B is clearly better
- "tie" if they are roughly equivalent

Output Format: Generate valid JSON:
{{
  "conversation_a": {{
    "faithfulness": {{"score": 0, "explanation": "..."}},
    "completeness": {{"score": 0, "explanation": "..."}},
    "safety": {{"score": 0, "explanation": "..."}},
    "clarity": {{"score": 0, "explanation": "..."}},
    "conciseness": {{"score": 0, "explanation": "..."}}
  }},
  "conversation_b": {{
    "faithfulness": {{"score": 0, "explanation": "..."}},
    "completeness": {{"score": 0, "explanation": "..."}},
    "safety": {{"score": 0, "explanation": "..."}},
    "clarity": {{"score": 0, "explanation": "..."}},
    "conciseness": {{"score": 0, "explanation": "..."}}
  }},
  "preference": "A|B|tie",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentences explaining the preference>"
}}

Ensure valid JSON with double quotes and escaped quotes inside explanations."""
        return prompt

    def _extract_json(self, response_text: str) -> Optional[dict]:
        """Extract the first JSON object from a possibly messy model response.

        Returns None if no JSON is present or the JSON is malformed (e.g. unescaped
        quotes), so callers can retry rather than crash.
        """
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return None
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model and return the response text (provider-aware)."""
        if self._provider == "openai":
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1500,
            )
            return response.choices[0].message.content or ""
        elif self._provider == "anthropic":
            message = self.client.messages.create(
                model=self.judge_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extended-thinking models (e.g. Claude 5) return thinking blocks
            # first; concatenate only the text blocks (thinking blocks have no .text).
            return "".join(getattr(b, "text", "") for b in message.content)
        elif self._provider == "gemini":
            response = self.client.generate_content(
                prompt,
                generation_config={"max_output_tokens": 1500},
            )
            return getattr(response, "text", "") or ""
        else:  # openai_compatible: local Ollama/vLLM, OpenRouter, DashScope
            # Ask the model to skip its thinking channel (Qwen/DeepSeek) so the
            # response is clean JSON; also strip any inline <think>...</think>.
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt + "\n\n/no_think"}],
                max_tokens=1500,
            )
            text = response.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def compare_conversations(
        self,
        conversation_a: Dict,
        conversation_b: Dict,
        scenario: Dict
    ) -> Optional[PairwiseComparison]:
        """Compare two conversations and return preference (identity-blind, randomized A/B order)."""

        # Randomize which generator is shown as A/B to mitigate positional bias
        randomized = False
        if random.random() < 0.5:
            conversation_a, conversation_b = conversation_b, conversation_a
            randomized = True

        prompt = self._create_pairwise_prompt(conversation_a, conversation_b, scenario)

        try:
            # Some judges (esp. Claude 5) occasionally emit malformed JSON
            # (e.g. unescaped quotes in explanations). Retry a few times with a
            # stricter instruction before giving up on this comparison.
            data = None
            for attempt in range(4):
                call_prompt = prompt if attempt == 0 else (
                    prompt
                    + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Respond with ONLY a single valid JSON object and nothing else. "
                    "Keep every explanation short and do NOT use any double quotes, "
                    "newlines, or backslashes inside explanation text. Include all "
                    "required keys, including \"preference\" and \"confidence\"."
                )
                response_text = self._call_judge(call_prompt)
                data = self._extract_json(response_text)
                if data is not None:
                    break
            if data is None:
                print("  Error: No valid JSON in response (after retries)")
                return None

            a_scores = [
                data["conversation_a"]["faithfulness"]["score"],
                data["conversation_a"]["completeness"]["score"],
                data["conversation_a"]["safety"]["score"],
                data["conversation_a"]["clarity"]["score"],
                data["conversation_a"]["conciseness"]["score"],
            ]
            b_scores = [
                data["conversation_b"]["faithfulness"]["score"],
                data["conversation_b"]["completeness"]["score"],
                data["conversation_b"]["safety"]["score"],
                data["conversation_b"]["clarity"]["score"],
                data["conversation_b"]["conciseness"]["score"],
            ]
            a_overall = sum(a_scores) / len(a_scores)
            b_overall = sum(b_scores) / len(b_scores)

            comparison = PairwiseComparison(
                scenario_id=scenario.get("scenario_id", "unknown"),

                conversation_a_id=conversation_a["conversation_id"],
                conversation_a_model=conversation_a["generator_model"],
                conversation_b_id=conversation_b["conversation_id"],
                conversation_b_model=conversation_b["generator_model"],

                preference=data["preference"],
                # `confidence` is non-critical metadata; default to 0.0 if the judge
                # omits it rather than discarding the whole (otherwise-valid) comparison.
                confidence=float(data.get("confidence") or 0.0),

                a_faithfulness=float(data["conversation_a"]["faithfulness"]["score"]),
                a_completeness=float(data["conversation_a"]["completeness"]["score"]),
                a_safety=float(data["conversation_a"]["safety"]["score"]),
                a_clarity=float(data["conversation_a"]["clarity"]["score"]),
                a_conciseness=float(data["conversation_a"]["conciseness"]["score"]),
                a_overall=float(a_overall),

                b_faithfulness=float(data["conversation_b"]["faithfulness"]["score"]),
                b_completeness=float(data["conversation_b"]["completeness"]["score"]),
                b_safety=float(data["conversation_b"]["safety"]["score"]),
                b_clarity=float(data["conversation_b"]["clarity"]["score"]),
                b_conciseness=float(data["conversation_b"]["conciseness"]["score"]),
                b_overall=float(b_overall),

                reasoning=data.get("reasoning", ""),
                timestamp=datetime.now().isoformat(),
                randomized=randomized,
            )

            return comparison

        except Exception as e:
            print(f"  Error comparing conversations: {e}")
            return None

    def evaluate_pairs(
        self,
        conversations_a: Dict[str, Dict],
        conversations_b: Dict[str, Dict],
        scenarios: Dict[str, Dict],
        sample_size: Optional[int] = None,
    ) -> List[PairwiseComparison]:
        """Compare conversations from two models on matching scenarios."""

        common_scenarios = set(conversations_a.keys()) & set(conversations_b.keys())
        print(f"\nFound {len(common_scenarios)} common scenarios")

        if sample_size:
            common_scenarios = set(
                random.sample(list(common_scenarios), min(sample_size, len(common_scenarios)))
            )
            print(f"Sampling {len(common_scenarios)} for comparison")

        comparisons: List[PairwiseComparison] = []

        model_a_name = list(conversations_a.values())[0]["generator_model"] if conversations_a else "Unknown"
        model_b_name = list(conversations_b.values())[0]["generator_model"] if conversations_b else "Unknown"

        print(f"\nComparing {model_a_name} vs {model_b_name}...")
        print(f"{'='*60}")

        for i, scenario_id in enumerate(sorted(common_scenarios), 1):
            print(f"\n[{i}/{len(common_scenarios)}] Scenario: {scenario_id}")

            convo_a = conversations_a[scenario_id]
            convo_b = conversations_b[scenario_id]
            scenario = scenarios.get(scenario_id, {"scenario_id": scenario_id})

            comparison = self.compare_conversations(convo_a, convo_b, scenario)

            if comparison:
                comparisons.append(comparison)
                print(f"  Preference: {comparison.preference} (confidence: {comparison.confidence:.2f})")
                print(
                    f"  Shown-A ({comparison.conversation_a_model}): {comparison.a_overall:.2f}/5.0 | "
                    f"Shown-B ({comparison.conversation_b_model}): {comparison.b_overall:.2f}/5.0"
                )
                if comparison.randomized:
                    print("  Note: A/B order was randomized (swapped).")
            else:
                print("  Skipped (error)")

        return comparisons

    def save_comparisons(self, comparisons: List[PairwiseComparison], output_file: str):
        """Save comparison results to JSON."""
        if not comparisons:
            print("No comparisons to save")
            return

        a_wins = sum(1 for c in comparisons if c.preference == "A")
        b_wins = sum(1 for c in comparisons if c.preference == "B")
        ties = sum(1 for c in comparisons if c.preference == "tie")

        # Note: A/B here refers to which was shown as A/B (randomized per comparison)
        a_avg_score = sum(c.a_overall for c in comparisons) / len(comparisons)
        b_avg_score = sum(c.b_overall for c in comparisons) / len(comparisons)

        output_data = {
            "metadata": {
                "framework": "MEDHELM",
                "evaluation_type": "pairwise_preference_identity_blind",
                "judge_model": self.judge_model,
                "total_comparisons": len(comparisons),
                "timestamp": datetime.now().isoformat(),
                "note": "Generator identities were hidden from the judge; A/B order randomized per scenario."
            },
            "summary": {
                "A_wins": a_wins,
                "B_wins": b_wins,
                "ties": ties,
                "A_win_rate": a_wins / len(comparisons),
                "B_win_rate": b_wins / len(comparisons),
                "A_avg_score": a_avg_score,
                "B_avg_score": b_avg_score,
            },
            "comparisons": [asdict(c) for c in comparisons],
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_file}")


    def generate_summary_report(self, comparisons: List[PairwiseComparison]) -> str:
        """Generate a text summary report aggregated by generator model (not A/B)."""
        if not comparisons:
            return "No comparisons to report"

        models = set()
        for c in comparisons:
            models.add(c.conversation_a_model)
            models.add(c.conversation_b_model)
        models = sorted(models)

        judge = getattr(self, "judge_model", "unknown")
        total = len(comparisons)

        wins = {m: 0 for m in models}
        overall_sum = {m: 0.0 for m in models}
        n_scored = {m: 0 for m in models}

        metrics = ["faithfulness", "completeness", "safety", "clarity", "conciseness"]
        metric_sum = {
            m: {metric: 0.0 for metric in metrics}
            for m in models
        }

        ties = 0

        for c in comparisons:
            # Preference → model mapping (per comparison)
            if c.preference == "A":
                wins[c.conversation_a_model] += 1
            elif c.preference == "B":
                wins[c.conversation_b_model] += 1
            else:
                ties += 1

            a_model = c.conversation_a_model
            n_scored[a_model] += 1
            overall_sum[a_model] += c.a_overall
            for metric in metrics:
                metric_sum[a_model][metric] += getattr(c, f"a_{metric}")

            # B-side scores
            b_model = c.conversation_b_model
            n_scored[b_model] += 1
            overall_sum[b_model] += c.b_overall
            for metric in metrics:
                metric_sum[b_model][metric] += getattr(c, f"b_{metric}")

        report = f"""
    {'='*70}
    PAIRWISE EVALUATION SUMMARY REPORT 
    {'='*70}

    Judge Model: {judge}
    Total Comparisons: {total}
    Timestamp: {datetime.now().isoformat()}

    PREFERENCE DISTRIBUTION (by generator model, accounting for A/B randomization):
    """

        for model in models:
            report += f"  {model} Wins: {wins[model]} ({wins[model]/total*100:.1f}%)\n"
        report += f"  Ties: {ties} ({ties/total*100:.1f}%)\n"

        report += "\nAVERAGE SCORES (0-5 scale, by generator model):\n"
        for model in models:
            avg = overall_sum[model] / max(n_scored[model], 1)
            report += f"  {model}: {avg:.2f}/5.0\n"

        report += "\nMETRIC BREAKDOWN (Average across all comparisons, by generator model):\n"
        for metric in metrics:
            report += f"\n  {metric.title()}:\n"
            for model in models:
                avg = metric_sum[model][metric] / max(n_scored[model], 1)
                report += f"    {model}: {avg:.2f}/5.0\n"

        report += f"\n{'='*70}\n"
        return report


async def main():
    parser = argparse.ArgumentParser(
        description="Pairwise evaluation to detect self-preference bias in medical LLMs. "
                    "Runs a grid of {turn counts} x {judge models} in one invocation, "
                    "writing one result file per cell."
    )
    parser.add_argument("--conv_dir", default="example_conversations",
                        help="Directory containing the conversation JSON files")
    parser.add_argument("--gen_a", default="gpt-4",
                        help="Generator A model name (used to locate conv files and name outputs)")
    parser.add_argument("--gen_b", default="claude-sonnet-4-5-20250929",
                        help="Generator B model name (used to locate conv files)")
    parser.add_argument("--turns", type=int, nargs="+", default=[2, 6],
                        help="Turn counts to evaluate (one result file per turn count x judge)")
    parser.add_argument("--judge_models", nargs="+",
                        default=["claude-sonnet-4-5-20250929", "gpt-4"],
                        help="Judge models. Provider auto-detected (gpt-*/o1-*/o3- -> OpenAI)")
    parser.add_argument("--scenarios", default="example_conversations/scenarios.json",
                        help="Path to scenarios JSON")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of scenarios to compare (default: all)")
    parser.add_argument("--output_dir", default="evals",
                        help="Directory for result files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (reset per cell so every judge sees identical A/B order)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"\nGrid: turns={args.turns} x judges={args.judge_models}")
    print(f"-> {len(args.turns) * len(args.judge_models)} result file(s) in {output_dir}/")

    for n in args.turns:
        a_file = Path(args.conv_dir) / f"{args.gen_a}_{n}t_conversations.json"
        b_file = Path(args.conv_dir) / f"{args.gen_b}_{n}t_conversations.json"

        for judge in args.judge_models:
            # Reset RNG per cell so scenario sampling and A/B randomization are
            # identical across judges for the same turn count (fair comparison).
            random.seed(args.seed)

            print(f"\n{'#'*70}\n# {n}t | judge={judge}\n{'#'*70}")
            evaluator = PairwiseEvaluator(judge_model=judge)

            convos_a = evaluator.load_conversations(str(a_file))
            convos_b = evaluator.load_conversations(str(b_file))
            scenarios = evaluator.load_scenarios(args.scenarios)

            comparisons = evaluator.evaluate_pairs(
                convos_a, convos_b, scenarios,
                sample_size=args.sample_size,
            )

            output = output_dir / f"pairwise_{args.gen_a}_vs_{args.gen_b}_{n}t_{judge}_Judge.json"
            evaluator.save_comparisons(comparisons, str(output))

            report = evaluator.generate_summary_report(comparisons)
            print(report)
            report_file = output.with_name(output.stem + "_summary.txt")
            with open(report_file, "w") as f:
                f.write(report)
            print(f"Report saved to {report_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
