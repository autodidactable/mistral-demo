#!/usr/bin/env python3
import os, io, json, time, base64, requests
from uuid import uuid4
from typing import Dict, Any

# ── .env ───────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv("mistral-tts-eval/.env")  # optional extra location
except Exception:
    pass

BT_API_KEY     = os.getenv("BRAINTRUST_API_KEY")
BT_PROJECT_ID  = os.getenv("BRAINTRUST_PROJECT_ID")
BT_BASE        = (os.getenv("BRAINTRUST_BASE_URL") or "https://api.braintrust.dev").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (BT_API_KEY and BT_PROJECT_ID and OPENAI_API_KEY):
    raise SystemExit("Missing env: BRAINTRUST_API_KEY, BRAINTRUST_PROJECT_ID, OPENAI_API_KEY")

# ── Braintrust helpers (root-only: 1 row per prompt) ───────────────────────────
def _bt_headers():
    return {"Authorization": f"Bearer {BT_API_KEY}", "Content-Type": "application/json"}

def _now_s(): return time.time()

INSERT_URL = f"{BT_BASE}/v1/project_logs/{BT_PROJECT_ID}/insert"

def _bt_post(payload: dict):
    r = requests.post(INSERT_URL, headers=_bt_headers(), json=payload, timeout=60)
    if not r.ok:
        try: err = r.json()
        except Exception: err = r.text
        raise SystemExit(f"BT insert failed: {r.status_code} {err}")
    return r.json()

def bt_start_root(name: str, input_obj: Any) -> str:
    root_id = uuid4().hex
    _bt_post({
        "events": [{
            "id": root_id,
            "span_attributes": {"name": name, "span_type": "root"},
            "input": input_obj,                 # keep as plain string (prompt)
            "metrics": {"start": _now_s()},
        }]
    })
    return root_id

def bt_merge_end_root(root_id: str, *, outputs=None, scores=None, artifacts=None, top_output=None, top_input=None):
    event = {"id": root_id, "is_merge": True, "metrics": {"end": _now_s()}}
    if top_input  is not None: event["input"]  = top_input       # keep Input column populated
    if top_output is not None: event["output"] = top_output      # human-readable Output column
    if outputs:   event["outputs"]   = outputs
    if scores:    event["scores"]    = scores
    if artifacts: event["artifacts"] = artifacts
    _bt_post({"events": [event]})

# ── helpers ────────────────────────────────────────────────────────────────────
def data_url(mime: str, b: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

OPENAI_VOICE_MAP = {
    "female_warm_neutral": "alloy",
    "male_calm_inquisitive": "verse",
}
def resolve_voice(profile: str) -> str:
    return OPENAI_VOICE_MAP.get(profile or "", "alloy")

# ── Chat Completions (audio out) ───────────────────────────────────────────────
def openai_chat_audio(prompt: str,
                      model: str = "gpt-4o-mini-audio-preview",
                      voice: str = "alloy",
                      audio_format: str = "mp3") -> tuple[bytes, str]:
    """
    One-shot: text → audio via Chat Completions (audio preview models).
    Returns (audio_bytes, mime).
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

    # Robust parsing across preview shapes
    audio_b64 = None
    try:
        audio_b64 = data["choices"][0]["message"]["audio"]["data"]
    except Exception:
        pass
    if audio_b64 is None:
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
    return base64.b64decode(audio_b64), f"audio/{audio_format}"

# ── OpenAI Whisper transcription ───────────────────────────────────────────────
def openai_transcribe(audio_bytes: bytes, filename: str = "audio.mp3", model: str = "whisper-1") -> str:
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
    return r.json().get("text", "")

# ── Judge (1–5 → 0–1) ─────────────────────────────────────────────────────────
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

def openai_judge(prompt: str, transcript: str) -> dict:
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
    return json.loads(r.json()["choices"][0]["message"]["content"])

def to01(x: int) -> float:
    return max(0.0, min(1.0, (float(x)-1.0)/4.0))  # 1..5 -> 0..1

# ── Prosody-tagging stub ───────────────────────────────────────────────────────
def prosody_tag_stub(text: str) -> str:
    t = text.strip()
    t = t.replace(". ", ". [pause_350ms] ")
    t = t.replace("? ", "? [pause_350ms] ")
    t = t.replace("! ", "! [raised_voice] [pause_350ms] ")
    return t

# ── Runner (root-only) ────────────────────────────────────────────────────────
def run_example(row: Dict[str,Any]):
    inputs = row.get("input", {}) or {}
    prompt = (inputs.get("user_prompt") or "").strip()
    if not prompt:
        print("skip (empty prompt)"); return

    root_id = bt_start_root("tts_eval_with_transcription", input_obj=prompt)

    # Prompt → audio (single call)
    voice = resolve_voice(inputs.get("voice_profile"))
    audio_bytes, mime = openai_chat_audio(prompt, model="gpt-4o-mini-audio-preview", voice=voice, audio_format="mp3")
    audio_url = data_url(mime, audio_bytes)

    # Transcribe + tag
    transcript_raw = openai_transcribe(audio_bytes, filename="audio.mp3")
    transcript_tagged = prosody_tag_stub(transcript_raw)

    # Judge (1–5 → 0–1)
    raw = openai_judge(prompt, transcript_tagged)
    scores_01 = {k: to01(int(raw[k])) for k in ["content_accuracy","prosody","fluency","style","overall"]}
    judge_extras = {
        "judge_model": JUDGE_MODEL,
        "judge_raw_1to5": {k:int(raw[k]) for k in ["content_accuracy","prosody","fluency","style","overall"]},
        "judge_justification": raw.get("justification",""),
    }

    # Log (keep default Input/Output useful)
    outputs = {
        "user_prompt": prompt,
        "model_name": "gpt-4o-mini-audio-preview",
        "voice": voice,
        "audio_url": audio_url,
        "audio_format": mime,
        "transcript_raw": transcript_raw,
        "transcript_tagged": transcript_tagged,
        **judge_extras,
    }
    artifacts = {
        "audio": audio_url,
        "transcript_raw":    data_url("text/plain", transcript_raw.encode("utf-8")),
        "transcript_tagged": data_url("text/plain", transcript_tagged.encode("utf-8")),
    }

    bt_merge_end_root(
        root_id,
        outputs=outputs,
        scores=scores_01,
        artifacts=artifacts,
        top_input=prompt,                 # Default "Input" column shows the prompt
        top_output=transcript_tagged      # Default "Output" column shows readable answer
    )
    print(f"[BT] logged: '{prompt[:60]}...'  mime={mime}")

# ── CLI ───────────────────────────────────────────────────────────────────────
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
