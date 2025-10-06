#!/usr/bin/env python3
"""
Braintrust SDK tracing (doc-style) with your original function names preserved.

Per example (row), this creates:
  root:   eval_case
    ├─ TTS: OpenAI Chat-Audio   (type="llm")         -> openai_chat_audio(...)
    ├─ ASR: OpenAI Whisper      (type="tool")        -> openai_transcribe(...)
    └─ Judge: LLM Scoring       (type="llm")         -> openai_judge(...)

Each child span logs input/output, metrics, metadata, and attachments.
The root span logs final output + scores for clean table columns.

Env needed:
  BRAINTRUST_API_KEY
  BRAINTRUST_PROJECT  (or BRAINTRUST_PROJECT_ID)
  OPENAI_API_KEY
"""

import os, io, json, base64, requests
from typing import Dict, Any, Optional, Tuple

# ── .env ───────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv("mistral-tts-eval/.env")  # optional extra location
except Exception:
    pass

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
PROJECT_NAME       = (os.getenv("BRAINTRUST_PROJECT") or "").strip().strip('"').strip("'")
PROJECT_ID         = (os.getenv("BRAINTRUST_PROJECT_ID") or "").strip().strip('"').strip("'")
if not OPENAI_API_KEY or not (PROJECT_NAME or PROJECT_ID):
    raise SystemExit("Missing env: OPENAI_API_KEY and BRAINTRUST_PROJECT (name) or BRAINTRUST_PROJECT_ID")

PROJECT_REF = PROJECT_NAME or PROJECT_ID

# ── Braintrust SDK  ─────────────────────────────────────────────────
from braintrust import current_span, init_logger, start_span, traced
logger = init_logger(project=PROJECT_REF)

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
    # 1..5 -> 0..1
    return max(0.0, min(1.0, (float(x) - 1.0) / 4.0))


def prosody_tag_stub(text: str) -> str:
    t = (text or "").strip()
    t = t.replace(". ", ". [pause_350ms] ")
    t = t.replace("? ", "? [pause_350ms] ")
    t = t.replace("! ", "! [raised_voice] [pause_350ms] ")
    return t

# We *wrap* these with @traced, and log inside via current_span().log

@traced(type="llm", name="TTS: OpenAI Chat-Audio", notrace_io=True)
def openai_chat_audio(prompt: str,
                      model: str = "gpt-4o-mini-audio-preview",
                      voice: str = "alloy",
                      audio_format: str = "mp3") -> Tuple[bytes, str]:
    """
    Returns (audio_bytes, mime). Child span logs input/output/metrics/attachments.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": audio_format},
        "temperature": 0.3,
        "messages": [
            {"role": "system",
             "content": "You are a helpful expert. Answer concisely (3-6 sentences), in a warm explanatory tone."},
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if not r.ok:
        try: err = r.json()
        except Exception: err = r.text
        raise RuntimeError(f"OpenAI chat-audio {r.status_code}: {err}")
    data = r.json()

    #  parsing across preview shapes
    audio_b64 = None
    try:
        audio_b64 = data["choices"][0]["message"]["audio"]["data"]
    except Exception:
        try:
            parts = data["choices"][0]["message"]["content"] or []
            for p in parts:
                if isinstance(p, dict) and p.get("type") in ("audio", "output_audio"):
                    audio_b64 = p.get("data") or (p.get("audio") or {}).get("data")
                    if audio_b64: break
        except Exception:
            pass
    if not audio_b64:
        raise RuntimeError("Chat audio response missing base64 audio.")

    audio_bytes = base64.b64decode(audio_b64)
    mime = f"audio/{audio_format}"
    audio_url = data_url(mime, audio_bytes)

    # Basic metrics (approx) for annotations
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
            "artifacts": {"audio": audio_url},
        },
    )


    return audio_bytes, mime

@traced(type="tool", name="ASR: OpenAI Whisper", notrace_io=True)
def openai_transcribe(audio_bytes: bytes,
                      filename: str = "audio.mp3",
                      model: str = "whisper-1") -> str:
    """
    Returns transcript (string). Child span logs transcript length and attachments.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "file": (filename, io.BytesIO(audio_bytes), "audio/mpeg"),
        "model": (None, model),
        "response_format": (None, "json"),
    }
    r = requests.post(url, headers=headers, files=files, timeout=120)
    if not r.ok:
        try: err = r.json()
        except Exception: err = r.text
        raise RuntimeError(f"OpenAI Transcribe {r.status_code}: {err}")
    text = r.json().get("text", "")
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
    """
    Returns raw 1–5 score dict. Child span logs normalized + raw 1–5.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "messages": [
            {"role":"system","content": JUDGE_SYS},
            {"role":"user","content": JUDGE_USER_TMPL.format(prompt=prompt, transcript=transcript)}
        ],
        "response_format": {"type":"json_object"}
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if not r.ok:
        try: err = r.json()
        except Exception: err = r.text
        raise RuntimeError(f"OpenAI Judge {r.status_code}: {err}")
    raw = json.loads(r.json()["choices"][0]["message"]["content"])

    scores_01 = {k: to01(int(raw[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}
    current_span().log(
        input={"prompt_preview": prompt[:200], "transcript_preview": transcript[:200]},
        output={"scores_01": scores_01, "judge_raw_1to5": raw},
        metadata={"judge_model": JUDGE_MODEL},
    )
    return raw

# ── Runner (root span + child spans via the traced functions above) ────────────
def run_example(row: Dict[str,Any]):
    inputs = row.get("input", {}) or {}
    prompt = (inputs.get("user_prompt") or "").strip()
    if not prompt:
        print("skip (empty prompt)"); return

    voice = resolve_voice(inputs.get("voice_profile"))

    with start_span(name="eval_case") as root:
        # TTS (child span)
        audio_bytes, mime = openai_chat_audio(prompt, model="gpt-4o-mini-audio-preview", voice=voice, audio_format="mp3")
        audio_url = data_url(mime, audio_bytes)

        # ASR (child span)
        transcript_raw = openai_transcribe(audio_bytes, filename="audio.mp3")
        transcript_tagged = prosody_tag_stub(transcript_raw)

        # Judge (child span)
        raw_1to5 = openai_judge(prompt, transcript_tagged)
        scores_01 = {k: to01(int(raw_1to5[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}

        # Root summary for table columns
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
                "artifacts": {  
                    "audio":             audio_url,
                    "transcript_raw":    data_url("text/plain", transcript_raw.encode("utf-8")),
                    "transcript_tagged": data_url("text/plain", transcript_tagged.encode("utf-8")),
                },
            },
        )



        print(f"[BT] traced: '{prompt[:60]}...'  mime={mime}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/prompts.jsonl")
    args = ap.parse_args()

    n = 0
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            run_example(row)
            n += 1
    print(f"Completed {n} prompts.")

if __name__ == "__main__":
    main()
