#!/usr/bin/env python3
"""
Eval (with child spans) that runs your OpenAI TTS pipeline against a Braintrust Dataset.

- Uses hooks.span("tts_openai" | "transcribe_whisper" | "judge_llm") to create child spans
- Robust to older SDKs (falls back to no-op if hooks.span is missing)
- Reads dataset by NAME (preferred) or ID (server-side attach)
- Returns Input/Output + scores just like your direct script

Env (in .env or exported):
  BRAINTRUST_API_KEY=...                  # Braintrust key
  BRAINTRUST_PROJECT=...                  # project NAME (or use BRAINTRUST_PROJECT_ID)
  OPENAI_API_KEY=sk_...                   # OpenAI key
  # Choose ONE:
  BT_DATASET_NAME=mistral-prompt-data
  # or
  BT_DATASET_ID=33329ecc-856a-4b73-a414-a2b31f19da1c
"""

import os, io, json, base64, requests, contextlib
from typing import Dict, Any, Tuple, Optional

# --- Load .env early ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import braintrust as bt

# --- Import your existing helpers (unchanged) ---
import run_bt_audio_direct as r


# ---------------------- Env helpers ----------------------
def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    if v is None:
        v = default
    return v.strip().strip('"').strip("'")

BT_API_KEY     = _env("BRAINTRUST_API_KEY")
PROJECT_NAME   = _env("BRAINTRUST_PROJECT")
PROJECT_ID     = _env("BRAINTRUST_PROJECT_ID")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
#DATASET_ID     = _env("BT_DATASET_ID") or _env("DATASET_ID")
#DATASET_NAME   = _env("BT_DATASET_NAME") or _env("DATASET_NAME")
DATASET_ID   = os.getenv("BT_DATASET_ID") or os.getenv("DATASET_ID")
DATASET_NAME = os.getenv("BT_DATASET_NAME") or os.getenv("DATASET_NAME")

if not BT_API_KEY:    raise SystemExit("Missing env: BRAINTRUST_API_KEY")
if not OPENAI_API_KEY:raise SystemExit("Missing env: OPENAI_API_KEY")
PROJECT_REF = PROJECT_NAME or PROJECT_ID
if not PROJECT_REF:   raise SystemExit("Missing env: set BRAINTRUST_PROJECT (name) or BRAINTRUST_PROJECT_ID (id)")

# ---------------------- Span helper ----------------------
def _span(hooks, name: str):
    """
    Prefer hooks.span(name=...), else no-op so this runs on older SDKs too.
    """
    if hasattr(hooks, "span") and callable(getattr(hooks, "span")):
        return hooks.span(name=name)
    return contextlib.nullcontext()

# ---------------------- Row normalizer ----------------------
def _extract_prompt_expected(raw) -> Tuple[str, Optional[str]]:
    """
    Supports Datum objects (row.input/row.expected), dicts, and primitives.
    """
    prompt, expected = None, None

    if hasattr(raw, "input") or hasattr(raw, "expected"):
        prompt   = getattr(raw, "input", None)
        expected = getattr(raw, "expected", None)
        if isinstance(prompt, dict):
            prompt = prompt.get("user_prompt") or prompt.get("prompt") or prompt.get("text") or prompt.get("input")

    elif isinstance(raw, dict):
        if isinstance(raw.get("input"), dict):
            inner = raw["input"]
            prompt   = inner.get("user_prompt") or inner.get("prompt") or inner.get("text") or inner.get("input")
            expected = inner.get("expected", raw.get("expected"))
        else:
            prompt   = raw.get("input") or raw.get("prompt") or raw.get("user_prompt") or raw.get("text")
            expected = raw.get("expected") or raw.get("reference") or raw.get("label") or raw.get("target")

    else:
        prompt = str(raw)

    prompt = (str(prompt).strip() if prompt is not None else "")
    expected = (None if expected is None else str(expected))
    return prompt, expected

# --- stable datum wrapper so framework never sees primitives ---
# --- stable datum wrapper so framework never sees primitives ---
class _Datum:
    __slots__ = ("input", "expected", "metadata", "tags")
    def __init__(self, inp, expected=None, metadata=None, tags=None):
        self.input = inp
        self.expected = expected
        self.metadata = metadata or {}
        # Braintrust expects .tags to exist; default to list
        self.tags = list(tags or [])

def _dataset_as_datums():
    """
    Iterate the dataset and yield _Datum(input, expected, metadata, tags).
    Prefer ID for safety; fall back to NAME.
    """
    # If your SDK supports id= in init_dataset, you can switch to that; name works broadly.
    ds = bt.init_dataset(project=PROJECT_REF, name=os.getenv("BT_DATASET_NAME") or os.getenv("DATASET_NAME"))

    for row in ds:
        prompt, expected, meta, tags = None, None, {}, []

        # Case 1: Braintrust Datum object
        if hasattr(row, "input") or hasattr(row, "expected") or hasattr(row, "metadata"):
            prompt   = getattr(row, "input", None)
            expected = getattr(row, "expected", None)
            meta     = getattr(row, "metadata", {}) or {}
            tags     = getattr(row, "tags", []) or []

        # Case 2: mapping (dict-like)
        elif isinstance(row, dict):
            prompt   = row.get("input") or row.get("prompt") or row.get("user_prompt") or row.get("text")
            expected = row.get("expected") or row.get("reference") or row.get("label") or row.get("target")
            meta     = row.get("metadata", {}) or {}
            tags     = row.get("tags", []) or []

        # Case 3: primitive -> coerce
        else:
            prompt, expected, meta, tags = str(row), None, {}, []

        # Unwrap nested input payloads
        if isinstance(prompt, dict):
            prompt = (
                prompt.get("user_prompt")
                or prompt.get("prompt")
                or prompt.get("text")
                or prompt.get("input")
                or str(prompt)
            )

        if prompt is None:
            continue

        yield _Datum(
            inp=str(prompt).strip(),
            expected=(None if expected is None else str(expected)),
            metadata=meta if isinstance(meta, dict) else {},
            tags=tags if isinstance(tags, (list, tuple)) else []
        )


# ---------------------- Task (with child spans) ----------------------
def tts_task(input_row: Dict[str, Any], hooks) -> Dict[str, Any]:
    """
    input_row can be a Datum, a dict, or a primitive string depending on how the dataset is streamed.
    We create child spans per step using hooks.span(...) and still return a single row payload.
    """
    prompt, expected = _extract_prompt_expected(input_row)
    if not prompt:
        return {"input": str(input_row), "output": "", "expected": expected or "", "error": "empty prompt"}

    voice = r.resolve_voice(None)  # customize to pass through voice_profile if your dataset includes it

    # --- TTS (child span) ---
    with _span(hooks, "tts_openai") as sp_tts:
        try:
            audio_bytes, mime = r.openai_chat_audio(
                prompt, model="gpt-4o-mini-audio-preview", voice=voice, audio_format="mp3"
            )
            audio_url = r.data_url(mime, audio_bytes)

            if hasattr(sp_tts, "log"):
                sp_tts.log({
                    "input": {"prompt": prompt, "voice": voice},
                    "output": {"audio_format": mime},
                    "attachments": {"audio": audio_url},
                    "metadata": {"provider": "openai", "model_name": "gpt-4o-mini-audio-preview"},
                })
        except Exception as e:
            if hasattr(sp_tts, "log"): sp_tts.log({"error": f"TTS error: {type(e).__name__}: {e}"})
            return {"input": prompt, "expected": expected or "", "output": "", "error": f"TTS: {e}"}

    # --- Transcribe (child span) ---
    with _span(hooks, "transcribe_whisper") as sp_tr:
        try:
            transcript_raw    = r.openai_transcribe(audio_bytes, filename="audio.mp3")
            transcript_tagged = r.prosody_tag_stub(transcript_raw)
            if hasattr(sp_tr, "log"):
                sp_tr.log({
                    "input": {"audio_format": mime},
                    "output": {"transcript_len": len(transcript_raw)},
                    "attachments": {
                        "transcript_raw":    r.data_url("text/plain", transcript_raw.encode("utf-8")),
                        "transcript_tagged": r.data_url("text/plain", transcript_tagged.encode("utf-8")),
                    },
                })
        except Exception as e:
            if hasattr(sp_tr, "log"): sp_tr.log({"error": f"Transcribe error: {type(e).__name__}: {e}"})
            return {"input": prompt, "expected": expected or "", "output": "", "error": f"Transcribe: {e}"}

    # --- Judge (child span) ---
    with _span(hooks, "judge_llm") as sp_j:
        try:
            judge_raw = r.openai_judge(prompt, transcript_tagged)
            scores_01 = {k: r.to01(int(judge_raw[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}
            for k in ("content_accuracy","prosody","fluency","style","overall"):
                scores_01[k] = float(scores_01.get(k) or 0.0)
            if hasattr(sp_j, "log"):
                sp_j.log({
                    "input": {"prompt_preview": prompt[:200], "transcript_preview": transcript_tagged[:200]},
                    "output": {"scores_01": scores_01, "judge_raw_1to5": judge_raw},
                    "metadata": {"judge_model": r.JUDGE_MODEL},
                })
        except Exception as e:
            if hasattr(sp_j, "log"): sp_j.log({"error": f"Judge error: {type(e).__name__}: {e}"})
            scores_01 = {k: 0.0 for k in ["content_accuracy","prosody","fluency","style","overall"]}
            judge_raw = {"content_accuracy": 0, "prosody": 0, "fluency": 0, "style": 0, "overall": 0, "justification": ""}

    # Row payload (what the UI table + scorers consume)
    return {
        "input": prompt,
        "expected": expected or "",
        "output": transcript_tagged,
        "scores": {
            "openai.content_accuracy": scores_01["content_accuracy"],
            "openai.prosody":          scores_01["prosody"],
            "openai.fluency":          scores_01["fluency"],
            "openai.style":            scores_01["style"],
            "openai.overall":          scores_01["overall"],
        },
        "scores_v1": {  # unprefixed (same naming as your earlier script)
            "content_accuracy": scores_01["content_accuracy"],
            "prosody":          scores_01["prosody"],
            "fluency":          scores_01["fluency"],
            "style":            scores_01["style"],
            "overall":          scores_01["overall"],
        },
        "attachments": {
            "audio":             audio_url,
            "transcript_raw":    r.data_url("text/plain", transcript_raw.encode("utf-8")),
            "transcript_tagged": r.data_url("text/plain", transcript_tagged.encode("utf-8")),
        },
        "metadata": {
            "provider": "openai",
            "model_name": "gpt-4o-mini-audio-preview",
            "voice": voice,
            "audio_format": mime,
        },
        "output_json": {
            "judge_model": r.JUDGE_MODEL,
            "judge_raw_1to5": {
                "content_accuracy": int(judge_raw.get("content_accuracy", 0)),
                "prosody":          int(judge_raw.get("prosody", 0)),
                "fluency":          int(judge_raw.get("fluency", 0)),
                "style":            int(judge_raw.get("style", 0)),
                "overall":          int(judge_raw.get("overall", 0)),
            },
            "judge_justification": judge_raw.get("justification", ""),
        },
    }


# ---------------------- Scorers (keyword-arg style) ----------------------
def _scores_from_output(output) -> dict:
    if not isinstance(output, dict):
        return {}
    if isinstance(output.get("scores_v1"), dict):
        return output["scores_v1"]
    s = output.get("scores")
    if isinstance(s, dict):
        return {
            "content_accuracy": s.get("openai.content_accuracy", s.get("content_accuracy")),
            "prosody":          s.get("openai.prosody",          s.get("prosody")),
            "fluency":          s.get("openai.fluency",          s.get("fluency")),
            "style":            s.get("openai.style",            s.get("style")),
            "overall":          s.get("openai.overall",          s.get("overall")),
        }
    return {}

def _num(x, default=0.0) -> float:
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)): return float(x)
        v = str(x).strip()
        if v.lower() in ("", "none", "nan"): return float(default)
        return float(v)
    except Exception:
        return float(default)

def score_overall(*, output=None, **kwargs):
    s = _scores_from_output(output)
    val = s.get("overall")
    if val is None:
        parts = [s.get("content_accuracy"), s.get("prosody"), s.get("fluency"), s.get("style")]
        parts = [p for p in parts if isinstance(p, (int, float))]
        val = sum(parts)/len(parts) if parts else 0.0
    return {"name": "overall", "score": _num(val)}

def score_content(*, output=None, **kwargs):
    return {"name": "content_accuracy", "score": _num(_scores_from_output(output).get("content_accuracy"))}

def score_prosody(*, output=None, **kwargs):
    return {"name": "prosody", "score": _num(_scores_from_output(output).get("prosody"))}

def score_fluency(*, output=None, **kwargs):
    return {"name": "fluency", "score": _num(_scores_from_output(output).get("fluency"))}

def score_style(*, output=None, **kwargs):
    return {"name": "style", "score": _num(_scores_from_output(output).get("style"))}


# ---------------------- Define the Eval ----------------------
# Attach dataset server-side (recommended), so the eval is linked to a dataset version in the UI.
if DATASET_ID:
    DATA_SPEC = {"dataset": {"id": DATASET_ID}}
elif DATASET_NAME:
    DATA_SPEC = {"dataset": {"name": DATASET_NAME}}
else:
    raise SystemExit("Set BT_DATASET_NAME or BT_DATASET_ID to select a dataset.")

bt.Eval(
    PROJECT_REF,
    task=lambda row, hooks: tts_task(row, hooks),   # child spans live in your task
    scores=[score_overall, score_content, score_prosody, score_fluency, score_style],
    data=_dataset_as_datums,                        # <<< stable _Datum objects (never strings)
)

