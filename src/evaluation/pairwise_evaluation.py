import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def _http_post_json(url: str, payload: dict, timeout: int = 120, retries: int = 5) -> dict:
    """POST JSON and return the parsed response, retrying with exponential backoff on
    transient errors (HTTP 429/500/502/503/504, network timeouts). Used for the raw HTTP
    endpoints (Gemini REST, Ollama native), which have no SDK-level retry."""
    data = json.dumps(payload).encode()
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            time.sleep(min(2 ** attempt, 30))
    raise last_exc


def _think_headroom() -> int:
    """Extra output tokens to allow when a judge is THINKING, so hidden reasoning doesn't
    consume the JSON budget and truncate it. Override THINK_TOKEN_HEADROOM."""
    return int(os.getenv("THINK_TOKEN_HEADROOM", "8192"))


def _claude_uses_adaptive_thinking(model_name: str) -> bool:
    """Claude 4.6+/5 use the adaptive-thinking API (thinking={"type":"adaptive"} +
    output_config.effort); the old {"type":"enabled","budget_tokens":N} is rejected (400).
    These models also think by DEFAULT, so no-think mode must disable explicitly."""
    m = model_name.lower()
    return bool(re.search(r"claude-(sonnet|opus|haiku|fable)-5", m)) or bool(
        re.search(r"claude-(sonnet|opus)-4-[678]", m)
    )


def _claude_effort() -> str:
    """Requested Claude thinking effort. Empty/none/off/disabled -> thinking OFF;
    otherwise low/medium/high/xhigh/max -> adaptive thinking at that effort."""
    return os.getenv("CLAUDE_THINKING_EFFORT", "").strip().lower()

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
            judge_model: Model ID (e.g. gpt-5.5, claude-sonnet-5, gemini-3.1-flash-lite, qwen3.6:35b).
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
            key = os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GOOGLE_API_KEY is not set. Required for Gemini judge models.")
            # Use REST directly: Gemini 3.x flash "thinks" and the old SDK can't disable it,
            # so thinking ate the token budget and truncated the JSON. REST lets us set
            # thinkingConfig.thinkingBudget=0 (thinking OFF) -> faster, complete JSON.
            self.client = None
            self._gemini_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{judge_model}"
                f":generateContent?key={key}"
            )
        else:  # openai_compatible (Qwen/DeepSeek/local/OpenRouter/DashScope)
            from openai import OpenAI
            self.compat_base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
            self.compat_api_key = os.getenv("OPENAI_COMPAT_API_KEY", "ollama")
            self.client = OpenAI(base_url=self.compat_base_url, api_key=self.compat_api_key)
            # Ollama detection: for thinking models the OpenAI-compat endpoint ignores
            # think/`/no_think` and returns EMPTY content (reasoning eats the budget).
            # Only native /api/chat with "think": false works. Verified on qwen3.6:35b.
            # Port-independent: the sbatch may serve Ollama on a job-specific port, so detect
            # by loopback host or the dummy "ollama" key rather than the literal 11434.
            self._is_ollama = (self.compat_api_key.lower() == "ollama") or any(
                s in self.compat_base_url for s in ("11434", "127.0.0.1", "localhost")
            )
            native = self.compat_base_url.rstrip("/")
            if native.endswith("/v1"):
                native = native[:-3]
            self._native_chat_url = native.rstrip("/") + "/api/chat"

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

    def _valid_judge_json(self, data: Optional[dict]) -> bool:
        """True only if the parsed judge JSON has the full expected schema: both sides
        scored on all five criteria plus a preference. Lets the caller retry on a parsed-
        but-malformed response (e.g. a truncated one missing 'conversation_a') instead of
        crashing/dropping it."""
        if not isinstance(data, dict) or "preference" not in data:
            return False
        crits = ("faithfulness", "completeness", "safety", "clarity", "conciseness")
        for side in ("conversation_a", "conversation_b"):
            block = data.get(side)
            if not isinstance(block, dict):
                return False
            for c in crits:
                cell = block.get(c)
                if not isinstance(cell, dict) or "score" not in cell:
                    return False
        return True

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model and return the response text (provider-aware)."""
        if self._provider == "openai":
            kw = dict(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
            # Turn reasoning OFF for GPT-5.x/o-series judges (consistency + speed). Valid:
            # none/low/medium/high/xhigh; override via OPENAI_REASONING_EFFORT.
            jm = self.judge_model.lower()
            if jm.startswith(("gpt-5", "o1", "o3", "o4")):
                effort = os.getenv("OPENAI_REASONING_EFFORT", "none")
                kw["reasoning_effort"] = effort
                if effort and effort.lower() != "none":
                    kw["max_completion_tokens"] = 4096 + _think_headroom()
            response = self.client.chat.completions.create(**kw)
            return response.choices[0].message.content or ""
        elif self._provider == "anthropic":
            akw = dict(model=self.judge_model, max_tokens=4096,
                       messages=[{"role": "user", "content": prompt}])
            # Thinking control for Claude 4.6+/5 (adaptive-thinking API; the old
            # {"type":"enabled","budget_tokens":N} is rejected with a 400). These models
            # think by default, so no-think disables explicitly; thinking shares the
            # max_tokens budget, so add headroom to avoid truncating the JSON verdict.
            if _claude_uses_adaptive_thinking(self.judge_model):
                effort = _claude_effort()
                if effort in ("", "none", "off", "0", "disabled"):
                    akw["thinking"] = {"type": "disabled"}
                else:
                    akw["thinking"] = {"type": "adaptive"}
                    akw["output_config"] = {"effort": effort}
                    akw["max_tokens"] = 4096 + _think_headroom()
            message = self.client.messages.create(**akw)
            # Extended-thinking models (e.g. Claude 5) return thinking blocks
            # first; concatenate only the text blocks (thinking blocks have no .text).
            return "".join(getattr(b, "text", "") for b in message.content)
        elif self._provider == "gemini":
            # thinkingBudget: 0=off (default), -1=dynamic, N=fixed. Thinking shares
            # maxOutputTokens, so add headroom when on to avoid truncating the JSON.
            budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
            max_out = 8192 + (_think_headroom() if budget != 0 else 0)
            gen_cfg = {"maxOutputTokens": max_out, "thinkingConfig": {"thinkingBudget": budget}}
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
            data = _http_post_json(self._gemini_url, payload, timeout=120)
            cand = (data.get("candidates") or [{}])[0]
            return "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        else:  # openai_compatible: local Ollama/vLLM, OpenRouter, DashScope
            # Ollama: native /api/chat. QWEN_THINK=1 enables thinking (+ headroom); the
            # compat endpoint ignores think and returns empty content for thinking judges.
            if self._is_ollama:
                think = os.getenv("QWEN_THINK", "0") == "1"
                body = {
                    "model": self.judge_model,
                    "stream": False,
                    "think": think,
                    "options": {"num_predict": 4096 + (_think_headroom() if think else 0)},
                    "messages": [{"role": "user", "content": prompt}],
                }
                data = _http_post_json(self._native_chat_url, body, timeout=300)
                text = data.get("message", {}).get("content", "") or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Generic OpenAI-compatible host: best-effort /no_think soft switch + strip.
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt + "\n\n/no_think"}],
                max_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def compare_conversations(
        self,
        conversation_a: Dict,
        conversation_b: Dict,
        scenario: Dict,
        swap: bool = False,
    ) -> Optional[PairwiseComparison]:
        """Compare two conversations and return preference (identity-blind).

        `swap` (whether to show gen_b in slot A) is decided by the CALLER from a
        deterministic (seed, scenario_id) key -- NOT from a shared RNG here -- so the A/B
        presentation is identical across judges and independent of execution order. That
        keeps positional bias a common factor that cancels in the cross-judge SPI, even
        under parallel evaluation. See evaluate_pairs._swap_for."""

        randomized = swap
        if swap:
            conversation_a, conversation_b = conversation_b, conversation_a

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
                if self._valid_judge_json(data):
                    break
                data = None  # parsed but malformed (e.g. truncated) -> retry
            if data is None:
                print("  Error: No valid/complete JSON in response (after retries)")
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

    @staticmethod
    def _swap_for(seed: int, scenario_id: str) -> bool:
        """Deterministic A/B swap decision for a scenario, keyed by (seed, scenario_id).

        Independent of execution order and of which judge is running, so every judge
        presents the same scenario in the same slot -> positional bias cancels in the
        cross-judge SPI, and it's safe under parallel (thread) evaluation."""
        return random.Random(f"{seed}:{scenario_id}").random() < 0.5

    def evaluate_pairs(
        self,
        conversations_a: Dict[str, Dict],
        conversations_b: Dict[str, Dict],
        scenarios: Dict[str, Dict],
        sample_size: Optional[int] = None,
        seed: int = 42,
        max_workers: int = 12,
    ) -> List[PairwiseComparison]:
        """Compare conversations from two models on matching scenarios.

        Comparisons are independent single calls, so we run up to `max_workers` at once in
        a thread pool. A/B swaps are pre-decided deterministically from (seed, scenario_id)
        so parallelism can't change the presentation (see _swap_for)."""

        common_scenarios = sorted(set(conversations_a.keys()) & set(conversations_b.keys()))
        print(f"\nFound {len(common_scenarios)} common scenarios")

        if sample_size:
            # Local seeded RNG so sampling is reproducible and independent of the A/B keying.
            common_scenarios = sorted(
                random.Random(seed).sample(common_scenarios, min(sample_size, len(common_scenarios)))
            )
            print(f"Sampling {len(common_scenarios)} for comparison")

        model_a_name = list(conversations_a.values())[0]["generator_model"] if conversations_a else "Unknown"
        model_b_name = list(conversations_b.values())[0]["generator_model"] if conversations_b else "Unknown"
        print(f"\nComparing {model_a_name} vs {model_b_name} "
              f"({len(common_scenarios)} scenarios, up to {max_workers} concurrent)...")
        print(f"{'='*60}")

        def _task(scenario_id: str) -> Optional[PairwiseComparison]:
            return self.compare_conversations(
                conversations_a[scenario_id],
                conversations_b[scenario_id],
                scenarios.get(scenario_id, {"scenario_id": scenario_id}),
                swap=self._swap_for(seed, scenario_id),
            )

        by_sid: Dict[str, Optional[PairwiseComparison]] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_task, sid): sid for sid in common_scenarios}
            for fut in as_completed(futures):
                sid = futures[fut]
                done += 1
                try:
                    by_sid[sid] = fut.result()
                except Exception as e:
                    print(f"  [{sid}] error: {e}")
                    by_sid[sid] = None
                if done % 25 == 0 or done == len(common_scenarios):
                    ok = sum(1 for v in by_sid.values() if v is not None)
                    print(f"  ...{done}/{len(common_scenarios)} done ({ok} ok)")

        # Preserve deterministic (sorted-scenario) order in the output; drop failures.
        comparisons = [by_sid[sid] for sid in common_scenarios if by_sid.get(sid) is not None]
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
                        help="Random seed governing A/B order (per-scenario, so every judge "
                             "sees identical A/B regardless of order/parallelism) and sampling")
    parser.add_argument("--max_concurrency", type=int, default=12,
                        help="Max comparisons judged concurrently per cell. For a Qwen judge "
                             "this is gated by OLLAMA_NUM_PARALLEL. Lower if you hit 429s.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"\nGrid: turns={args.turns} x judges={args.judge_models}")
    print(f"-> {len(args.turns) * len(args.judge_models)} result file(s) in {output_dir}/")

    for n in args.turns:
        a_file = Path(args.conv_dir) / f"{args.gen_a}_{n}t_conversations.json"
        b_file = Path(args.conv_dir) / f"{args.gen_b}_{n}t_conversations.json"

        for judge in args.judge_models:
            # A/B order is keyed per-scenario by (seed, scenario_id) inside evaluate_pairs,
            # so every judge sees identical A/B regardless of order or parallelism -- no
            # global RNG reseed needed.
            print(f"\n{'#'*70}\n# {n}t | judge={judge}\n{'#'*70}")
            evaluator = PairwiseEvaluator(judge_model=judge)

            convos_a = evaluator.load_conversations(str(a_file))
            convos_b = evaluator.load_conversations(str(b_file))
            scenarios = evaluator.load_scenarios(args.scenarios)

            comparisons = evaluator.evaluate_pairs(
                convos_a, convos_b, scenarios,
                sample_size=args.sample_size,
                seed=args.seed,
                max_workers=args.max_concurrency,
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
