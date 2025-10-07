#!/usr/bin/env python3
"""
Braintrust SDK + OpenAI SDK (wrapped with braintrust.wrap_openai)
Reads test cases from a Braintrust Dataset (default name: 'mistral').

- Child spans via @traced(type=..., name=..., notrace_io=True)
- Structured logging with current_span().log (allowed keys only)
- Original function names preserved:
    openai_chat_audio, openai_transcribe, openai_judge, prosody_tag_stub
"""

import os, io, json, base64
from typing import Dict, Any, Optional, Tuple

# ── .env ───────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv("mistral-tts-eval/.env")   # optional extra
except Exception:
    pass

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
PROJECT_NAME       = (os.getenv("BRAINTRUST_PROJECT") or "").strip().strip('"').strip("'")
PROJECT_ID         = (os.getenv("BRAINTRUST_PROJECT_ID") or "").strip().strip('"').strip("'")
DATASET_NAME       = (os.getenv("BT_DATASET_NAME") or os.getenv("DATASET_NAME") or "mistral").strip()

if not OPENAI_API_KEY or not (PROJECT_NAME or PROJECT_ID):
    raise SystemExit("Missing env: OPENAI_API_KEY and BRAINTRUST_PROJECT (name) or BRAINTRUST_PROJECT_ID")
PROJECT_REF = PROJECT_NAME or PROJECT_ID

# ── Braintrust SDK + OpenAI SDK (wrapped) ─────────────────────────────────────
from braintrust import init_logger, current_span, start_span, traced, wrap_openai, init_dataset
from openai import OpenAI

logger = init_logger(project=PROJECT_REF)
client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))  # all calls go through Braintrust proxy

# ── Helpers ────────────────────────────────────────────────────────────────────
def data_url(mime: str, b: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

OPENAI_VOICE_MAP = {
    "female_warm_neutral": "alloy",
    "male_calm_inquisitive": "verse",
}
def resolve_voice(profile: Optional[str]) -> str:
    return OPENAI_VOICE_MAP.get(profile or "", "alloy")

def to01(x: int) -> float:
    return max(0.0, min(1.0, (float(x) - 1.0) / 4.0))  # 1..5 -> 0..1

def prosody_tag_stub(text: str) -> str:
    t = (text or "").strip()
    t = t.replace(". ", ". [pause_350ms] ")
    t = t.replace("? ", "? [pause_350ms] ")
    t = t.replace("! ", "! [raised_voice] [pause_350ms] ")
    return t

# ── Child spans via @traced (per docs) ─────────────────────────────────────────
@traced(type="llm", name="TTS: OpenAI Chat-Audio", notrace_io=True)
def openai_chat_audio(prompt: str,
                      model: str = "gpt-4o-mini-audio-preview",
                      voice: str = "alloy",
                      audio_format: str = "mp3") -> Tuple[bytes, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful expert. Answer concisely (3-6 sentences), in a warm explanatory tone."},
            {"role": "user", "content": prompt},
        ],
        modalities=["text", "audio"],
        audio={"voice": voice, "format": audio_format},
        temperature=0.3,
    )

    audio_b64 = None
    try:
        audio_b64 = resp.choices[0].message.audio.data  # preferred shape
    except Exception:
        try:
            parts = getattr(resp.choices[0].message, "content", None) or []
            for p in parts:
                tp = getattr(p, "type", None) or (isinstance(p, dict) and p.get("type"))
                if tp in ("audio", "output_audio"):
                    audio_b64 = (getattr(p, "data", None)
                                 or (getattr(p, "audio", None) and getattr(p.audio, "data", None))
                                 or (isinstance(p, dict) and (p.get("data") or (p.get("audio") or {}).get("data"))))
                    if audio_b64: break
        except Exception:
            pass

    if not audio_b64:
        raise RuntimeError("Chat audio response missing base64 audio.")

    audio_bytes = base64.b64decode(audio_b64)
    mime = f"audio/{audio_format}"
    audio_url = data_url(mime, audio_bytes)

    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(audio_bytes) // 320)

    current_span().log(
        input=[{"role": "user", "content": prompt}],
        output={"audio_format": mime, "voice": voice},
        metrics=dict(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens=prompt_tokens + completion_tokens,
        ),
        metadata={
            "provider": "openai",
            "model": model,
            "voice": voice,
            "format": audio_format,
            "artifacts": {"audio": audio_url},  # data-URL stored in metadata
        },
    )
    return audio_bytes, mime

@traced(type="tool", name="ASR: OpenAI Whisper", notrace_io=True)
def openai_transcribe(audio_bytes: bytes,
                      filename: str = "audio.mp3",
                      model: str = "whisper-1") -> str:
    bio = io.BytesIO(audio_bytes)
    bio.name = filename
    resp = client.audio.transcriptions.create(
        model=model,
        file=bio,
        response_format="json",
    )
    text = getattr(resp, "text", None)
    if text is None and isinstance(resp, dict):
        text = resp.get("text", "")
    if text is None:
        text = ""

    tagged = prosody_tag_stub(text)
    current_span().log(
        input={"filename": filename, "bytes": len(audio_bytes)},
        output={"transcript_len": len(text)},
        metadata={
            "provider": "openai",
            "model": model,
            "artifacts": {
                "transcript_raw":    data_url("text/plain", text.encode("utf-8")),
                "transcript_tagged": data_url("text/plain", tagged.encode("utf-8")),
            },
        },
    )
    return text

# Judge constants preserved
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_SYS = """You are evaluating a TTS assistant's response.
Return ONLY JSON with integer scores 1-5 and a short justification."""
JUDGE_USER_TMPL = """
User prompt:
«{prompt}»

Transcript of assistant audio:
«{transcript}»

Score 1–5 (integers only):
- content_accuracy
- prosody
- fluency
- style
Return ONLY JSON:
{{"content_accuracy":N,"prosody":N,"fluency":N,"style":N,"overall":N,"justification":"..."}}"""

@traced(type="llm", name="Judge: LLM Scoring", notrace_io=True)
def openai_judge(prompt: str, transcript: str) -> dict:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role":"system","content": JUDGE_SYS},
            {"role":"user","content": JUDGE_USER_TMPL.format(prompt=prompt, transcript=transcript)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw_json = resp.choices[0].message.content
    raw = raw_json if isinstance(raw_json, dict) else json.loads(raw_json)

    scores_01 = {k: to01(int(raw[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}
    current_span().log(
        input={"prompt_preview": prompt[:200], "transcript_preview": transcript[:200]},
        output={"scores_01": scores_01, "judge_raw_1to5": raw},
        metadata={"judge_model": JUDGE_MODEL},
    )
    return raw

# ── Runner (root span + child spans via the traced functions above) ────────────
def run_example(prompt: str, *, voice_profile: Optional[str] = None):
    voice = resolve_voice(voice_profile)

    with start_span(name="eval_case") as root:
        # TTS
        audio_bytes, mime = openai_chat_audio(
            prompt, model="gpt-4o-mini-audio-preview", voice=voice, audio_format="mp3"
        )
        audio_url = data_url(mime, audio_bytes)

        # ASR
        transcript_raw = openai_transcribe(audio_bytes, filename="audio.mp3")
        transcript_tagged = prosody_tag_stub(transcript_raw)

        # Judge
        raw_1to5 = openai_judge(prompt, transcript_tagged)
        scores_01 = {k: to01(int(raw_1to5[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}

        # Root summary (allowed keys only)
        root.log(
            input=prompt,
            output=transcript_tagged,
            metrics=scores_01,
            metadata={
                "voice": voice,
                "audio_format": mime,
                "tts_model": "gpt-4o-mini-audio-preview",
                "judge_model": JUDGE_MODEL,
                "judge_raw_1to5": raw_1to5,
                "artifacts": {  # data URLs kept here
                    "audio":             audio_url,
                    "transcript_raw":    data_url("text/plain", transcript_raw.encode("utf-8")),
                    "transcript_tagged": data_url("text/plain", transcript_tagged.encode("utf-8")),
                },
            },
        )
        print(f"[BT] traced via SDK: '{prompt[:60]}...'  mime={mime}")

# ── Main: iterate Braintrust Dataset instead of prompts.jsonl ──────────────────
def _prompt_from_dataset_row(row: Any) -> Optional[str]:
    """
    Accepts Braintrust Datum objects or dicts/strings and extracts the prompt string.
    Dataset 'mistral' rows are assumed to have top-level 'input' (string).
    """
    # Datum object
    if hasattr(row, "input"):
        v = getattr(row, "input")
        return v if isinstance(v, str) else (v.get("user_prompt") or v.get("prompt") if isinstance(v, dict) else None)
    # Mapping
    if isinstance(row, dict):
        v = row.get("input") or row.get("prompt") or row.get("user_prompt") or row.get("text")
        return v if isinstance(v, str) else (v.get("user_prompt") or v.get("prompt") if isinstance(v, dict) else None)
    # Primitive
    if isinstance(row, str):
        return row
    return None

def main():
    # Attach dataset by NAME (minimal change); set BT_DATASET_NAME in .env to override
    ds = init_dataset(project=PROJECT_REF, name=DATASET_NAME)

    n = 0
    for row in ds:
        prompt = _prompt_from_dataset_row(row)
        if not prompt:
            continue
        # (optional) pull a voice profile if present
        voice_profile = None
        if hasattr(row, "metadata") and isinstance(getattr(row, "metadata"), dict):
            voice_profile = row.metadata.get("voice_profile")
        elif isinstance(row, dict):
            voice_profile = (row.get("metadata") or {}).get("voice_profile")

        run_example(prompt, voice_profile=voice_profile)
        n += 1

    print(f"Completed {n} prompts.")

if __name__ == "__main__":
    main()
