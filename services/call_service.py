"""
Orchestrates outbound voice calls:
- Twilio outbound call + TwiML webhooks
- Deepgram transcription of recorded response
- LLM response generation
- ElevenLabs speech synthesis for playback
"""

import os
import re
import uuid
from urllib.parse import urlencode
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import requests
from twilio.rest import Client as TwilioClient

from services.openai_client import OpenAIClient


class CallService:
    """Coordinates phone-call workflows from text instructions."""

    _pending_calls: Dict[str, Dict[str, Any]] = {}
    _lock: Lock = Lock()

    def __init__(self, twilio_client: TwilioClient, openai_client: OpenAIClient):
        self.twilio_client = twilio_client
        self.openai = openai_client
        self.twilio_voice_from = (os.getenv("TWILIO_VOICE_NUMBER") or "").strip()
        self.base_url = (os.getenv("APP_BASE_URL") or "").strip().rstrip("/")
        self.deepgram_api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        self.elevenlabs_api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        self.elevenlabs_voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
        self.audio_dir = Path(os.getenv("CALL_AUDIO_DIR") or "generated_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_phone_number(raw: str) -> str:
        digits = re.sub(r"[^\d+]", "", (raw or "").strip())
        if digits.startswith("+"):
            return digits
        if digits:
            return f"+{digits}"
        return ""

    def start_outbound_call(self, requested_by: str, to_number: str, prompt_question: str) -> Dict[str, Any]:
        if not self.twilio_voice_from:
            return {"success": False, "error": "TWILIO_VOICE_NUMBER is not configured"}
        if not self.base_url:
            return {"success": False, "error": "APP_BASE_URL is not configured"}

        normalized_to = self._normalize_phone_number(to_number)
        if not normalized_to:
            return {"success": False, "error": "Could not parse a valid target phone number"}

        call_id = str(uuid.uuid4())
        prompt_question = (prompt_question or "").strip()
        with self._lock:
            self._pending_calls[call_id] = {
                "requested_by": requested_by,
                "to_number": normalized_to,
                "prompt_question": prompt_question,
            }

        query = urlencode(
            {
                "call_id": call_id,
                "prompt_question": prompt_question,
                "requested_by": requested_by,
                "to_number": normalized_to,
            }
        )
        call = self.twilio_client.calls.create(
            to=normalized_to,
            from_=self.twilio_voice_from,
            url=f"{self.base_url}/voice/call?{query}",
            method="POST",
        )
        return {"success": True, "call_id": call_id, "call_sid": call.sid, "to_number": normalized_to}

    def get_call_context(self, call_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._pending_calls.get(call_id)

    def _transcribe_with_deepgram(self, audio_bytes: bytes, content_type: str) -> str:
        if not self.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY is not configured")
        resp = requests.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&punctuate=true",
            headers={
                "Authorization": f"Token {self.deepgram_api_key}",
                "Content-Type": content_type or "audio/wav",
            },
            data=audio_bytes,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "").strip()
        )

    def _synthesize_with_elevenlabs(self, text: str, output_path: Path) -> None:
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is not configured")
        if not self.elevenlabs_voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is not configured")

        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}",
            headers={
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.75},
            },
            timeout=60,
        )
        resp.raise_for_status()
        output_path.write_bytes(resp.content)

    def process_recording(self, call_id: str, recording_url: str, prompt_question: str = "") -> Dict[str, Any]:
        context = self.get_call_context(call_id)
        prompt_to_use = (prompt_question or "").strip() or ((context or {}).get("prompt_question") or "").strip()
        if not prompt_to_use:
            return {"success": False, "error": "Unknown call context"}

        if not recording_url:
            return {"success": False, "error": "Missing recording URL"}

        rec_resp = requests.get(f"{recording_url}.wav", timeout=60)
        rec_resp.raise_for_status()
        content_type = rec_resp.headers.get("Content-Type", "audio/wav")
        transcript = self._transcribe_with_deepgram(rec_resp.content, content_type=content_type)
        if not transcript:
            transcript = "No speech recognized."

        llm_response = self.openai.chat(
            system=(
                "You are assisting in a live phone call. "
                "Reply naturally and briefly in 1-3 sentences, suitable for speaking aloud."
            ),
            user=(f"Original question to ask on call: {prompt_to_use}\n" f"Call recipient response: {transcript}"),
            temperature=0.2,
            max_tokens=180,
        )
        spoken_text = llm_response or "Thanks, I could not generate a response."

        output_name = f"{call_id}.mp3"
        output_path = self.audio_dir / output_name
        audio_url = None
        tts_error = ""
        try:
            self._synthesize_with_elevenlabs(spoken_text, output_path=output_path)
            if not self.base_url:
                return {"success": False, "error": "APP_BASE_URL is not configured"}
            audio_url = f"{self.base_url}/audio/{output_name}"
        except Exception as e:
            # Graceful fallback: caller can still use Twilio <Say> with llm_response.
            tts_error = str(e)

        return {
            "success": True,
            "transcript": transcript,
            "llm_response": spoken_text,
            "audio_url": audio_url,
            "tts_error": tts_error,
        }
