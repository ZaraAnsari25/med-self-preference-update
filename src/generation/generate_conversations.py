"""Generate multi-turn medical dialogues for self-preference research."""

import json
import asyncio
import argparse
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import hashlib
import re
import time
import urllib.request
import urllib.error

import yaml
from dotenv import load_dotenv

from datasets import load_dataset
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


def _http_post_json(url: str, payload: dict, timeout: int = 120, retries: int = 5) -> dict:
    """POST JSON and return the parsed response, retrying with exponential backoff on
    transient errors (HTTP 429/500/502/503/504, network timeouts). Used for the raw
    HTTP endpoints (Gemini REST, Ollama native) which — unlike the OpenAI/Anthropic
    SDKs — have no built-in retry, so a transient blip would otherwise drop a turn."""
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
    """Extra output tokens to allow when a model is THINKING, so hidden reasoning doesn't
    consume the answer budget and leave empty/truncated output. Override THINK_TOKEN_HEADROOM."""
    return int(os.getenv("THINK_TOKEN_HEADROOM", "8192"))

@dataclass
class MedicalScenario:
    """Seed scenario extracted from HealthCareMagic dataset."""
    scenario_id: str
    original_patient_query: str
    original_doctor_response: str
    chief_complaint: str
    risk_level: str
    task_type: str
    
@dataclass
class ConversationTurn:
    """Single turn in a dialogue."""
    turn_number: int
    role: str
    content: str
    timestamp: str
    
@dataclass
class GeneratedConversation:
    """Complete multi-turn conversation."""
    conversation_id: str
    scenario_id: str
    generator_model: str
    patient_simulator: str
    turns: List[ConversationTurn]
    total_turns: int
    generation_params: Dict
    created_at: str



PHYSICIAN_SYSTEM_PROMPT = """You are an experienced clinician conducting a telemedicine consultation.

Goal: provide high-quality, specific clinical reasoning while staying safe and concise.

Rules:
- Be concrete and targeted (avoid generic boilerplate).
- Ask at most 3 focused questions per message.
- Keep it brief: ~2–6 sentences unless you must include safety advice.
- No role labels (do NOT write "Physician:" / "Doctor:").
- Do not invent exam findings, vitals, labs, imaging, or confirmed diagnoses.
- If symptoms suggest a potential emergency, clearly recommend urgent/emergency care.

Progression:
- Early turns: clarify history (onset, location, severity, triggers, associated symptoms, red flags, PMH/meds/allergies).
- After ~3 clinician turns: summarize leading differential (1–3 items with a short why), give a plan (2–5 actionable steps), and include safety-net red flags."""


PATIENT_SIMULATOR_SYSTEM_PROMPT = """You are simulating a patient in a telemedicine consultation.

SCENARIO CONTEXT:
{scenario_context}

ORIGINAL COMPLAINT: {original_complaint}
REFERENCE CLINICIAN NOTE (PRIVATE, do not quote/copy): {reference_note}

Your role:
1. Respond naturally as a patient would - use casual language, express concerns
2. Reveal information GRADUALLY when asked - don't dump all symptoms at once
3. Stay consistent with the original complaint; you do not know any diagnosis
4. Express appropriate emotions (worry, confusion, relief)
5. Sometimes be vague or uncertain, as real patients often are
6. You may add small realistic details ONLY if consistent (e.g., timing, triggers); do not invent major new symptoms

Important:
- Do NOT reveal or guess any diagnosis.
- Do NOT mention tests you didn't have or results you don't know.
- Keep responses conversational and realistic (1–3 short sentences typically)."""


PHYSICIAN_TURN_PROMPT = """Continue the medical consultation. Write ONLY your next message to the patient.

Current conversation:
{conversation_history}

Constraints reminder: max 3 questions; be specific; no role labels."""


PATIENT_TURN_PROMPT = """Continue the consultation as the patient. Write ONLY your next message.

Current conversation:
{conversation_history}

Respond naturally as the patient would. Remember to stay consistent with your condition and reveal information gradually."""



def load_healthcaremagic_scenarios(
    num_scenarios: int = 100,
    seed: int = 42,
    shuffle: bool = False,
) -> List[Dict]:
    """Load and return raw HealthCareMagic samples."""
    print("Loading HealthCareMagic dataset...")
    
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    scenarios = []
    for i, item in enumerate(dataset):
        if i >= num_scenarios:
            break
        scenarios.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })
    
    print(f"Loaded {len(scenarios)} scenarios")
    return scenarios


def classify_risk_level(patient_query: str, doctor_response: str) -> str:
    """Heuristic risk classification based on keywords."""
    high_risk_keywords = [
        "chest pain", "difficulty breathing", "shortness of breath",
        "severe", "emergency", "unconscious", "stroke", "heart attack",
        "suicide", "bleeding heavily", "can't breathe", "anaphylaxis",
        "pediatric", "infant", "newborn", "pregnancy complication"
    ]
    
    medium_risk_keywords = [
        "fever", "infection", "swelling", "persistent", "worsening",
        "medication", "chronic", "diabetes", "hypertension"
    ]
    
    text = (patient_query + " " + doctor_response).lower()
    
    if any(kw in text for kw in high_risk_keywords):
        return "high"
    elif any(kw in text for kw in medium_risk_keywords):
        return "medium"
    else:
        return "low"


def classify_task_type(patient_query: str, doctor_response: str) -> str:
    """Classify the primary task type of the consultation."""
    text = (patient_query + " " + doctor_response).lower()
    
    if any(kw in text for kw in ["what is", "diagnos", "what do i have", "what could"]):
        return "diagnosis"
    elif any(kw in text for kw in ["treatment", "medication", "prescri", "how to treat"]):
        return "treatment"
    elif any(kw in text for kw in ["explain", "why", "what does", "understand"]):
        return "explanation"
    elif any(kw in text for kw in ["follow up", "check", "return", "getting better"]):
        return "followup"
    else:
        return "diagnosis"


def prepare_scenarios(raw_data: List[Dict]) -> List[MedicalScenario]:
    """Convert raw dataset items to MedicalScenario objects."""
    scenarios = []
    
    for i, item in enumerate(raw_data):
        scenario_id = hashlib.md5(item["input"].encode()).hexdigest()[:12]
        
        scenario = MedicalScenario(
            scenario_id=f"hcm_{scenario_id}",
            original_patient_query=item["input"],
            original_doctor_response=item["output"],
            chief_complaint=item["input"][:200],
            risk_level=classify_risk_level(item["input"], item["output"]),
            task_type=classify_task_type(item["input"], item["output"])
        )
        scenarios.append(scenario)
    
    return scenarios



class LLMClient:
    """Base class for LLM API clients."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        raise NotImplementedError


def _model_supports_temperature(model_name: str) -> bool:
    """Some newer models (GPT-5.x, OpenAI o-series, Claude 5 family) only allow the
    default temperature and reject an explicit `temperature` argument. Return False
    for those so callers can omit the parameter."""
    m = model_name.lower()
    if m.startswith(("gpt-5", "o1", "o3")):
        return False
    if re.search(r"claude-(sonnet|opus|haiku|fable)-5", m):
        return False
    return True


def _claude_uses_adaptive_thinking(model_name: str) -> bool:
    """Claude 4.6+ / 5 use the adaptive-thinking API (thinking={"type":"adaptive"} +
    output_config.effort); the old thinking={"type":"enabled","budget_tokens":N} is
    rejected (400). These models also think by DEFAULT when `thinking` is omitted, so
    no-think mode must send {"type":"disabled"} explicitly."""
    m = model_name.lower()
    return bool(re.search(r"claude-(sonnet|opus|haiku|fable)-5", m)) or bool(
        re.search(r"claude-(sonnet|opus)-4-[678]", m)
    )


def _claude_effort() -> str:
    """Requested Claude thinking effort. Empty/none/off/disabled -> thinking OFF.
    Otherwise one of low/medium/high/xhigh/max -> adaptive thinking at that effort."""
    return os.getenv("CLAUDE_THINKING_EFFORT", "").strip().lower()


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model_name: str = "gpt-4"):
        super().__init__(model_name)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running generation."
            )
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max_tokens,
        )
        if _model_supports_temperature(self.model_name):
            kwargs["temperature"] = temperature
        # Reasoning models (GPT-5.x, o-series) spend part of max_completion_tokens on
        # hidden reasoning. We turn it OFF (reasoning_effort="none") for consistency with
        # the other models (all thinking disabled) and speed. Valid: none/low/medium/high/
        # xhigh; override via OPENAI_REASONING_EFFORT, or set it empty to omit the param
        # (e.g. for non-reasoning models like gpt-4o that reject it).
        m = self.model_name.lower()
        if m.startswith(("gpt-5", "o1", "o3", "o4")):
            effort = os.getenv("OPENAI_REASONING_EFFORT", "none")
            if effort:
                kwargs["reasoning_effort"] = effort
            if effort and effort.lower() != "none":
                # reasoning shares max_completion_tokens; add headroom so the answer fits.
                kwargs["max_completion_tokens"] = max_tokens + _think_headroom()
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic API client."""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        super().__init__(model_name)
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Please export ANTHROPIC_API_KEY before running generation."
            )
        import anthropic
        self.client = anthropic.AsyncAnthropic()
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        kwargs = dict(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        if _model_supports_temperature(self.model_name):
            kwargs["temperature"] = temperature
        # Thinking control for Claude 4.6+/5 (adaptive-thinking API; the old
        # {"type":"enabled","budget_tokens":N} is rejected with a 400). These models
        # think by default, so no-think mode disables explicitly. Effort levels:
        # low/medium/high/xhigh/max. Adaptive thinking shares the max_tokens budget, so
        # add headroom to avoid empty/truncated answers.
        if _claude_uses_adaptive_thinking(self.model_name):
            effort = _claude_effort()
            if effort in ("", "none", "off", "0", "disabled"):
                kwargs["thinking"] = {"type": "disabled"}
            else:
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": effort}
                kwargs["max_tokens"] = max_tokens + _think_headroom()
                kwargs.pop("temperature", None)
        response = await self.client.messages.create(**kwargs)
        # Extended-thinking models (e.g. Claude 5) return thinking blocks first;
        # concatenate only the text blocks (thinking blocks have no .text).
        return "".join(getattr(b, "text", "") for b in response.content)


class GeminiClient(LLMClient):
    """Google Gemini client (REST).

    Uses the REST API directly rather than the google-generativeai SDK because Gemini
    3.x flash "thinks" by default and the old SDK cannot disable it (no thinking_config
    field). We call generateContent with thinkingConfig.thinkingBudget=0 so thinking is
    OFF -- consistent with the other models and ~3x faster.
    """

    def __init__(self, model_name: str = "gemini-pro"):
        super().__init__(model_name)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Please export GOOGLE_API_KEY before running generation."
            )
        self.api_key = api_key
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}"
            f":generateContent?key={api_key}"
        )

    def _rest(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        # thinkingBudget: 0=off (default), -1=dynamic, N=fixed. Thinking shares
        # maxOutputTokens, so add headroom when on to keep the answer from truncating.
        budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
        max_out = max_tokens + (_think_headroom() if budget != 0 else 0)
        gen_cfg = {
            "temperature": temperature,
            "maxOutputTokens": max_out,
            "thinkingConfig": {"thinkingBudget": budget},
        }
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": gen_cfg,
        }
        data = _http_post_json(self.url, payload, timeout=120)
        cand = (data.get("candidates") or [{}])[0]
        return "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        return await asyncio.to_thread(self._rest, system_prompt, user_prompt, temperature, max_tokens)


class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible endpoint: local (Ollama/vLLM) or hosted
    (OpenRouter/DashScope). Endpoint + key come from env so switching hosts is a
    config change, not a code change:

        OPENAI_COMPAT_BASE_URL  (default http://localhost:11434/v1  -> local Ollama)
        OPENAI_COMPAT_API_KEY   (default "ollama"; a dummy is fine for local)

    Used for models that aren't native OpenAI/Anthropic/Gemini (e.g. Qwen, DeepSeek).
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        from openai import AsyncOpenAI
        self.base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("OPENAI_COMPAT_API_KEY", "ollama")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        # Ollama detection: for thinking models (Qwen3/DeepSeek) the OpenAI-compat
        # endpoint IGNORES `think`/`/no_think` and returns EMPTY content (all budget
        # goes to the hidden reasoning channel). Only Ollama's native /api/chat with
        # "think": false actually disables it. Verified empirically on qwen3.6:35b.
        # Port-independent: the sbatch may serve Ollama on a job-specific port, so detect
        # by loopback host or the dummy "ollama" key rather than the literal 11434.
        self._is_ollama = (self.api_key.lower() == "ollama") or any(
            s in self.base_url for s in ("11434", "127.0.0.1", "localhost")
        )
        native = self.base_url.rstrip("/")
        if native.endswith("/v1"):
            native = native[:-3]
        self._native_chat_url = native.rstrip("/") + "/api/chat"

    def _ollama_native(self, system_prompt, user_prompt, temperature, max_tokens):
        """Blocking call to Ollama's native /api/chat. QWEN_THINK=1 enables thinking (with
        token headroom so the answer isn't truncated); default (0) disables it."""
        think = os.getenv("QWEN_THINK", "0") == "1"
        body = {
            "model": self.model_name,
            "stream": False,
            "think": think,
            "options": {"num_predict": max_tokens + (_think_headroom() if think else 0)},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if _model_supports_temperature(self.model_name):
            body["options"]["temperature"] = temperature
        data = _http_post_json(self._native_chat_url, body, timeout=300)
        text = data.get("message", {}).get("content", "") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        # Ollama: use the native endpoint with think:false (the OpenAI-compat endpoint
        # ignores it and yields empty turns for thinking models). Run the blocking call
        # in a thread so we don't stall the event loop.
        if self._is_ollama:
            return await asyncio.to_thread(
                self._ollama_native, system_prompt, user_prompt, temperature, max_tokens
            )
        # Generic OpenAI-compatible host (OpenRouter/DashScope/vLLM): best-effort with
        # the /no_think soft switch + inline <think> stripping. Classic `max_tokens`.
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\n/no_think"},
            ],
            max_tokens=max_tokens,
        )
        if _model_supports_temperature(self.model_name):
            kwargs["temperature"] = temperature
        response = await self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        # Strip any <think>...</think> reasoning some models (e.g. Qwen) emit inline.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_client(model_name: str) -> LLMClient:
    """Factory: route a model id to the right client (native or OpenAI-compatible)."""
    model_lower = model_name.lower()

    if model_lower.startswith(("gpt-", "o1-", "o3-")):
        return OpenAIClient(model_name)
    elif "claude" in model_lower:
        return AnthropicClient(model_name)
    elif "gemini" in model_lower:
        return GeminiClient(model_name)
    else:
        # Qwen, DeepSeek, local, or any OpenAI-compatible endpoint.
        return OpenAICompatibleClient(model_name)



def format_conversation_history(turns: List[ConversationTurn]) -> str:
    """Format turns into a readable conversation history."""
    lines = []
    for turn in turns:
        role_label = "Physician" if turn.role == "physician" else "Patient"
        lines.append(f"{role_label}: {turn.content}")
    return "\n\n".join(lines)

_ROLE_PREFIX_LINE_RE = re.compile(
    r"^\s*(?:physician|doctor|clinician|patient|assistant)\s*:\s*",
    re.IGNORECASE,
)


def cleanup_model_text(text: str) -> str:
    """Remove role labels and collapse excessive blank lines."""
    if not text:
        return text

    cleaned = _ROLE_PREFIX_LINE_RE.sub("", text.strip())
    cleaned_lines = []
    for line in cleaned.splitlines():
        cleaned_lines.append(_ROLE_PREFIX_LINE_RE.sub("", line).rstrip())
    cleaned = "\n".join(cleaned_lines).strip()

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def physician_needs_repair(text: str) -> bool:
    if not text:
        return True
    if _ROLE_PREFIX_LINE_RE.match(text.strip()):
        return True
    if text.count("?") > 3:
        return True
    return False


def patient_needs_repair(text: str) -> bool:
    if not text:
        return True
    if _ROLE_PREFIX_LINE_RE.match(text.strip()):
        return True
    if len(text) > 800:
        return True
    return False


async def maybe_repair(
    *,
    client: LLMClient,
    role: str,
    original_text: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Rewrite to comply with constraints without changing meaning."""
    role_desc = "clinician" if role == "physician" else "patient"
    system = "You rewrite text to comply with constraints. Preserve meaning. Output only the rewritten text."
    user = (
        f"Rewrite the following {role_desc} message to comply with these constraints:\n"
        f"- No role labels like 'Physician:' or 'Patient:'\n"
        f"- Keep it concise\n"
        f"- Preserve the same content and intent\n\n"
        f"Message:\n{original_text}"
    )
    repaired = await client.generate(system, user, temperature=temperature, max_tokens=max_tokens)
    return cleanup_model_text(repaired)


async def generate_single_conversation(
    scenario: MedicalScenario,
    physician_client: LLMClient,
    patient_client: LLMClient,
    num_turns: int = 8,
    physician_temperature: float = 0.3,
    patient_temperature: float = 0.8,
    max_tokens_per_turn: int = 500,
    enable_repair: bool = False,
) -> GeneratedConversation:
    """Generate a complete multi-turn conversation for one scenario."""
    turns = []

    patient_system = PATIENT_SIMULATOR_SYSTEM_PROMPT.format(
        scenario_context=f"Chief complaint: {scenario.chief_complaint}",
        original_complaint=scenario.original_patient_query,
        reference_note=scenario.original_doctor_response
    )

    initial_turn = ConversationTurn(
        turn_number=0,
        role="patient",
        content=scenario.original_patient_query,
        timestamp=datetime.now().isoformat()
    )
    turns.append(initial_turn)

    for turn_num in range(1, num_turns):
        history = format_conversation_history(turns)

        if turn_num % 2 == 1:
            prompt = PHYSICIAN_TURN_PROMPT.format(conversation_history=history)
            response = await physician_client.generate(
                PHYSICIAN_SYSTEM_PROMPT, 
                prompt, 
                temperature=physician_temperature,
                max_tokens=max_tokens_per_turn,
            )
            role = "physician"
            response = cleanup_model_text(response)
            if enable_repair and physician_needs_repair(response):
                response = await maybe_repair(
                    client=physician_client,
                    role=role,
                    original_text=response,
                    temperature=0.2,
                    max_tokens=max_tokens_per_turn,
                )
        else:
            prompt = PATIENT_TURN_PROMPT.format(conversation_history=history)
            response = await patient_client.generate(
                patient_system,
                prompt,
                temperature=patient_temperature,
                max_tokens=max_tokens_per_turn,
            )
            role = "patient"
            response = cleanup_model_text(response)
            if enable_repair and patient_needs_repair(response):
                response = await maybe_repair(
                    client=patient_client,
                    role=role,
                    original_text=response,
                    temperature=0.5,
                    max_tokens=max_tokens_per_turn,
                )

        turn = ConversationTurn(
            turn_number=turn_num,
            role=role,
            content=response.strip(),
            timestamp=datetime.now().isoformat()
        )
        turns.append(turn)

    conv_id = f"{scenario.scenario_id}_{physician_client.model_name}_{num_turns}t"

    return GeneratedConversation(
        conversation_id=conv_id,
        scenario_id=scenario.scenario_id,
        generator_model=physician_client.model_name,
        patient_simulator=patient_client.model_name,
        turns=turns,
        total_turns=num_turns,
        generation_params={
            "physician_temperature": physician_temperature,
            "patient_temperature": patient_temperature,
            "max_tokens_per_turn": max_tokens_per_turn,
            "num_turns": num_turns
        },
        created_at=datetime.now().isoformat()
    )


async def generate_all_conversations(
    scenarios: List[MedicalScenario],
    physician_models: List[str],
    patient_simulator_model: str = "gpt-4",
    num_turns: int = 8,
    output_dir: Path = Path("./output"),
    physician_temperature: float = 0.3,
    patient_temperature: float = 0.8,
    max_tokens_per_turn: int = 500,
    enable_repair: bool = False,
    max_concurrency: int = 12,
) -> Dict[str, List[GeneratedConversation]]:
    """Generate conversations for all scenarios across all models.

    Conversations are independent, so we run up to `max_concurrency` of them at once
    (turns WITHIN a conversation stay sequential -- each turn conditions on the prior).
    Cloud models fan out freely; Qwen serializes on its one GPU (bounded by Ollama's
    OLLAMA_NUM_PARALLEL). Output order is preserved (sorted by original scenario index).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_client = get_client(patient_simulator_model)
    sem = asyncio.Semaphore(max_concurrency)

    results = {model: [] for model in physician_models}

    for model_name in physician_models:
        print(f"\nGenerating {len(scenarios)} conversations with {model_name} "
              f"(up to {max_concurrency} concurrent)")

        physician_client = get_client(model_name)

        async def _one(idx: int, scenario: MedicalScenario):
            async with sem:
                try:
                    conv = await generate_single_conversation(
                        scenario=scenario,
                        physician_client=physician_client,
                        patient_client=patient_client,
                        num_turns=num_turns,
                        physician_temperature=physician_temperature,
                        patient_temperature=patient_temperature,
                        max_tokens_per_turn=max_tokens_per_turn,
                        enable_repair=enable_repair,
                    )
                    return idx, conv
                except Exception as e:
                    print(f"Error generating for scenario {scenario.scenario_id}: {e}")
                    return idx, None

        tasks = [asyncio.ensure_future(_one(i, s)) for i, s in enumerate(scenarios)]
        by_idx: Dict[int, Optional[GeneratedConversation]] = {}
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{model_name}"):
            idx, conv = await fut
            by_idx[idx] = conv
        # Re-sort into original scenario order and drop failures.
        convs = [by_idx[i] for i in range(len(scenarios)) if by_idx.get(i) is not None]
        results[model_name] = convs

        save_conversations(convs, output_dir / f"{model_name}_{num_turns}t_conversations.json")

    return results



def conversation_to_dict(conv: GeneratedConversation) -> Dict:
    """Convert conversation to serializable dict."""
    return {
        "conversation_id": conv.conversation_id,
        "scenario_id": conv.scenario_id,
        "generator_model": conv.generator_model,
        "patient_simulator": conv.patient_simulator,
        "total_turns": conv.total_turns,
        "generation_params": conv.generation_params,
        "created_at": conv.created_at,
        "turns": [
            {
                "turn_number": t.turn_number,
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp
            }
            for t in conv.turns
        ]
    }


def save_conversations(conversations: List[GeneratedConversation], filepath: Path):
    """Save conversations to JSON file."""
    data = [conversation_to_dict(c) for c in conversations]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(conversations)} conversations to {filepath}")


def save_scenarios(scenarios: List[MedicalScenario], filepath: Path):
    """Save scenario metadata."""
    data = [asdict(s) for s in scenarios]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(scenarios)} scenarios to {filepath}")



def load_yaml_config(path: str) -> Dict:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception:
        return {}


def cfg_get(cfg: Dict, keys: List[str], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


async def main():
    parser = argparse.ArgumentParser(description="Generate medical multi-turn conversations")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Optional YAML config path (defaults to config/config.yaml)")
    parser.add_argument("--num_scenarios", type=int, default=None,
                        help="Number of scenarios to use")
    parser.add_argument("--turns", type=int, default=None,
                        help="Number of turns per conversation")
    parser.add_argument("--models", nargs="+", 
                        default=None,
                        help="Models to generate physician responses")
    parser.add_argument("--patient_model", type=str, default=None,
                        help="Model to use for patient simulation")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle dataset before taking first N (default: false)")
    parser.add_argument("--physician_temperature", type=float, default=None,
                        help="Temperature for physician turns (default from config or 0.3)")
    parser.add_argument("--patient_temperature", type=float, default=None,
                        help="Temperature for patient turns (default from config or 0.8)")
    parser.add_argument("--max_tokens_per_turn", type=int, default=None,
                        help="Max tokens per turn (default from config or 500)")
    parser.add_argument("--repair", action="store_true",
                        help="Enable 1-pass repair rewrite when a turn violates constraints")
    parser.add_argument("--max_concurrency", type=int, default=12,
                        help="Max conversations generated concurrently (cloud models fan out; "
                             "Qwen is bounded by OLLAMA_NUM_PARALLEL). Lower if you hit 429s.")

    args = parser.parse_args()

    # Enlarge the default thread pool so the blocking to_thread clients (Gemini REST,
    # Qwen/Ollama native) can actually run `max_concurrency` calls at once rather than
    # queueing behind the small default executor (~cpu+4 workers).
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=max(32, args.max_concurrency * 2), thread_name_prefix="gen"
        )
    )

    cfg = load_yaml_config(args.config)

    num_scenarios = args.num_scenarios if args.num_scenarios is not None else cfg_get(cfg, ["data", "num_scenarios"], 100)
    turns = args.turns if args.turns is not None else cfg_get(cfg, ["generation", "turns_per_conversation"], 8)
    models = args.models if args.models is not None else cfg_get(cfg, ["generation", "physician_models"], ["gpt-4"])
    patient_model = args.patient_model if args.patient_model is not None else cfg_get(cfg, ["generation", "patient_simulator"], "gpt-4")

    max_tokens_per_turn = (
        args.max_tokens_per_turn
        if args.max_tokens_per_turn is not None
        else cfg_get(cfg, ["generation", "max_tokens_per_turn"], 500)
    )
    physician_temperature = (
        args.physician_temperature
        if args.physician_temperature is not None
        else cfg_get(cfg, ["generation", "physician_temperature"], 0.3)
    )
    patient_temperature = (
        args.patient_temperature
        if args.patient_temperature is not None
        else cfg_get(cfg, ["generation", "patient_temperature"], 0.8)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_healthcaremagic_scenarios(
        num_scenarios=num_scenarios,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    scenarios = prepare_scenarios(raw_data)

    save_scenarios(scenarios, output_dir / "scenarios.json")

    print("\nScenario Distribution:")
    print(f"  Risk levels: {dict((r, sum(1 for s in scenarios if s.risk_level == r)) for r in ['low', 'medium', 'high'])}")
    print(f"  Task types: {dict((t, sum(1 for s in scenarios if s.task_type == t)) for t in ['diagnosis', 'treatment', 'explanation', 'followup'])}")
    results = await generate_all_conversations(
        scenarios=scenarios,
        physician_models=models,
        patient_simulator_model=patient_model,
        num_turns=turns,
        output_dir=output_dir,
        physician_temperature=physician_temperature,
        patient_temperature=patient_temperature,
        max_tokens_per_turn=max_tokens_per_turn,
        enable_repair=args.repair,
        max_concurrency=args.max_concurrency,
    )
    all_conversations = []
    for model, convs in results.items():
        all_conversations.extend(convs)

    save_conversations(all_conversations, output_dir / "all_conversations.json")

    print("\nGeneration complete")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Models used: {models}")
    print(f"Patient simulator: {patient_model}")
    print(f"Turns per conversation: {turns}")
    print(f"Physician temperature: {physician_temperature}")
    print(f"Patient temperature: {patient_temperature}")
    print(f"Max tokens/turn: {max_tokens_per_turn}")
    print(f"Repair enabled: {args.repair}")
    print(f"Total conversations generated: {len(all_conversations)}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
