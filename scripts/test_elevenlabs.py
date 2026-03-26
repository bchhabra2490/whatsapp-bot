"""
Quick sanity test for ElevenLabs TTS configuration.

Usage:
  source .venv/bin/activate
  python scripts/test_elevenlabs.py

Requires env vars:
  - ELEVENLABS_API_KEY
  - ELEVENLABS_VOICE_ID
Optional:
  - ELEVENLABS_MODEL_ID (default: eleven_v3)
  - ELEVENLABS_OUTPUT_FORMAT (default: mp3_44100_128)
  - CALL_AUDIO_DIR (default: generated_audio)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _write_audio_bytes(audio, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        if isinstance(audio, (bytes, bytearray)):
            f.write(audio)
            return
        for chunk in audio:
            if chunk:
                f.write(chunk)


def main() -> int:
    load_dotenv()

    api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
    model_id = (os.getenv("ELEVENLABS_MODEL_ID") or "eleven_v3").strip()
    output_format = (os.getenv("ELEVENLABS_OUTPUT_FORMAT") or "mp3_44100_128").strip()
    out_dir = Path(os.getenv("CALL_AUDIO_DIR") or "generated_audio")
    out_path = out_dir / "elevenlabs_test.mp3"

    if not api_key:
        print("FAIL: ELEVENLABS_API_KEY is missing.")
        return 2
    if not voice_id:
        print("FAIL: ELEVENLABS_VOICE_ID is missing.")
        return 2

    try:
        # ElevenLabs v2 SDK import
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=api_key)
        print("Calling ElevenLabs TTS...")
        audio = client.text_to_speech.convert(
            text="Hello! This is a test of the ElevenLabs text to speech setup.",
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        _write_audio_bytes(audio, out_path)

        size = out_path.stat().st_size if out_path.exists() else 0
        if size <= 0:
            print("FAIL: generated file is empty.")
            return 1

        print("OK: ElevenLabs TTS succeeded.")
        print(f"Wrote: {out_path} ({size} bytes)")
        return 0
    except Exception as e:
        print("FAIL: ElevenLabs TTS request failed.")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

