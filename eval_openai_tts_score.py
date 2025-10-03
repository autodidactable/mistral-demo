#!/usr/bin/env python3
"""
Eval that runs your OpenAI TTS pipeline against a Braintrust Dataset.
- Iterates via SDK by dataset NAME and yields stable dict rows
- Compatible with older SDKs (no hooks.log, Eval(..., data=...))

Env:
  BRAINTRUST_API_KEY=...                 # Braintrust key
  BRAINTRUST_PROJECT=mistral-tts-eval    # project NAME (or BRAINTRUST_PROJECT_ID)
  OPENAI_API_KEY=sk_...                  # OpenAI key
  DATASET_NAME=mistral                   # dataset NAME (preferred)
"""

import os
from typing import Dict, Any, Generator, Tuple
import braintrust as bt

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Import your existing pipeline helpers ──────────────────────────────────────
import run_bt_audio_direct as r

# ── Env helpers ────────────────────────────────────────────────────────────────
def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    if v is None:
        v = default
    return v.strip().strip('"').strip("'")

BT_API_KEY    = _env("BRAINTRUST_API_KEY")
PROJECT_NAME  = _env("BRAINTRUST_PROJECT")
PROJECT_ID    = _env("BRAINTRUST_PROJECT_ID")
OPENAI_KEY    = _env("OPENAI_API_KEY")
DATASET_NAME  = _env("DATASET_NAME") or _env("BT_DATASET_NAME") or "mistral-prompt-data"

if not BT_API_KEY:
    raise SystemExit("Missing env: BRAINTRUST_API_KEY")
if not OPENAI_KEY:
    raise SystemExit("Missing env: OPENAI_API_KEY")

PROJECT_REF = PROJECT_NAME or PROJECT_ID
if not PROJECT_REF:
    raise SystemExit("Missing env: set BRAINTRUST_PROJECT (name) or BRAINTRUST_PROJECT_ID (id)")

# ── Dataset iterator: always return dicts with input/expected/metadata ─────────
def _dataset_rows_as_dicts() -> Generator[Dict[str, Any], None, None]:
    """
    Yields rows like: {"input": <prompt str>, "expected": <gold or "">, "metadata": {...}}
    """
    ds = bt.init_dataset(project=PROJECT_REF, name=DATASET_NAME)

    for row in ds:
        prompt, expected, meta = None, None, {}

        # 1) SDK datum object
        if hasattr(row, "input") or hasattr(row, "expected") or hasattr(row, "metadata"):
            prompt   = getattr(row, "input", None)
            expected = getattr(row, "expected", None)
            meta     = getattr(row, "metadata", {}) or {}

        # 2) Mapping
        elif isinstance(row, dict):
            prompt   = row.get("input") or row.get("prompt") or row.get("user_prompt") or row.get("text")
            expected = row.get("expected") or row.get("reference") or row.get("label") or row.get("target")
            meta     = row.get("metadata", {}) or {}

        # 3) Primitive
        else:
            prompt = str(row)

        # If input is a dict, dig out the string
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

        yield {
            "input": str(prompt).strip(),
            "expected": "" if expected is None else str(expected),
            "metadata": meta if isinstance(meta, dict) else {},
        }

# ── Extractor (also handles object/dict/primitive when task sees raw rows) ─────
def _extract_prompt_expected(raw: Any) -> Tuple[str, str | None]:
    if hasattr(raw, "input") or hasattr(raw, "expected"):
        prompt = getattr(raw, "input", None)
        expected = getattr(raw, "expected", None)
        if isinstance(prompt, dict):
            prompt = prompt.get("user_prompt") or prompt.get("prompt") or prompt.get("text") or prompt.get("input")
        return (str(prompt).strip() if prompt else ""), (None if expected is None else str(expected))

    if isinstance(raw, dict):
        if isinstance(raw.get("input"), dict):
            inner = raw["input"]
            prompt = inner.get("user_prompt") or inner.get("prompt") or inner.get("text") or inner.get("input")
            expected = inner.get("expected", raw.get("expected"))
        else:
            prompt = raw.get("input") or raw.get("prompt") or raw.get("user_prompt") or raw.get("text")
            expected = raw.get("expected") or raw.get("reference") or raw.get("label") or raw.get("target")
        return (str(prompt).strip() if prompt else ""), (None if expected is None else str(expected))

    return (str(raw).strip(), None)

# ── Task: OpenAI TTS -> Whisper -> Judge ───────────────────────────────────────
def tts_task(input_row, hooks) -> Dict[str, Any]:
    # If using our dict rows generator, this will just read the two strings.
    prompt, expected = _extract_prompt_expected(input_row)

    if not prompt:
        return {"input": str(input_row), "output": "", "expected": expected or "", "error": "empty prompt"}

    voice = r.resolve_voice(None)  # pass a voice_profile if you add it to rows later

    # 1) text->audio
    audio_bytes, mime = r.openai_chat_audio(
        prompt, model="gpt-4o-mini-audio-preview", voice=voice, audio_format="mp3"
    )
    audio_url = r.data_url(mime, audio_bytes)

    # 2) transcribe + tag
    transcript_raw    = r.openai_transcribe(audio_bytes, filename="audio.mp3")
    transcript_tagged = r.prosody_tag_stub(transcript_raw)

    # 3) judge (1–5 -> 0–1)
    judge_raw = r.openai_judge(prompt, transcript_tagged)
    scores_01 = {k: r.to01(int(judge_raw[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}
    # ensure numeric (no None)
    for k in ("content_accuracy","prosody","fluency","style","overall"):
        scores_01[k] = float(scores_01.get(k) or 0.0)

    scores_v1 = {
        "content_accuracy": scores_01["content_accuracy"],
        "prosody":          scores_01["prosody"],
        "fluency":          scores_01["fluency"],
        "style":            scores_01["style"],
        "overall":          scores_01["overall"],
    }

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
        "scores_v1": scores_v1,
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
            "judge_raw_1to5": {k: int(judge_raw[k]) for k in ["content_accuracy","prosody","fluency","style","overall"]},
            "judge_justification": judge_raw.get("justification", ""),
        },
    }

# ── Scorers (robust; safe numeric coercion) ────────────────────────────────────
def _dig(node, key):
    """Get key from dict or attribute from object; return None if missing."""
    if isinstance(node, dict):
        return node.get(key)
    try:
        return getattr(node, key)
    except Exception:
        return None

def _to_dictish(node):
    """If node looks like an object with a dict payload, unwrap common shapes."""
    # Some SDKs hand an object with .output/.result/...
    for k in ("output", "result", "data", "value"):
        sub = _dig(node, k)
        if isinstance(sub, (dict,)) or hasattr(sub, "__dict__"):
            return sub
    # Last resort, try __dict__
    try:
        return node.__dict__
    except Exception:
        return node

# ---- Scorers (keyword-arg style: the framework passes output=<task return>) ----

def _scores_from_output(output) -> dict:
    """Extract a unified score map from the task's output object."""
    if not isinstance(output, dict):
        return {}
    # prefer the unprefixed map (identical to your original script)
    if isinstance(output.get("scores_v1"), dict):
        return output["scores_v1"]
    # fall back to namespaced map
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
    # use judge-provided overall (already 0..1 via to01)
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


# ── Define the Eval (older SDK signature: data=...) ────────────────────────────
bt.Eval(
    PROJECT_REF,
    task=lambda row, hooks: tts_task(row, hooks),
    scores=[score_overall, score_content, score_prosody, score_fluency, score_style],
    data=_dataset_rows_as_dicts,  # stable dict rows; prevents .metadata errors
)
