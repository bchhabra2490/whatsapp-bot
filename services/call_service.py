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
import json
from urllib.parse import urlencode
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import requests
from twilio.rest import Client as TwilioClient
from elevenlabs.client import ElevenLabs

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

        elevenlabs = ElevenLabs(api_key=self.elevenlabs_api_key)
        audio = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id=self.elevenlabs_voice_id,
            model_id=os.getenv("ELEVENLABS_MODEL_ID") or "eleven_v3",
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT") or "mp3_44100_128",
        )

        # The SDK returns an iterator/stream of bytes chunks.
        with output_path.open("wb") as f:
            if isinstance(audio, (bytes, bytearray)):
                f.write(audio)
            else:
                for chunk in audio:
                    if chunk:
                        f.write(chunk)

    def synthesize_for_call(self, call_id: str, text: str, tag: str) -> Dict[str, Any]:
        """
        Generate an MP3 file for Twilio <Play>.
        Returns: { success, audio_url, filename, error? }
        """
        if not self.base_url:
            return {"success": False, "error": "APP_BASE_URL is not configured"}
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", tag or "audio")
        filename = f"{call_id}_{safe_tag}.mp3"
        output_path = self.audio_dir / filename
        self._synthesize_with_elevenlabs((text or "").strip(), output_path=output_path)
        return {"success": True, "audio_url": f"{self.base_url}/audio/{filename}", "filename": filename}

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

        raw = self.openai.chat(
            system=(
                "You are controlling a phone-call assistant.\n"
                "Return STRICT JSON with keys:\n"
                "- response_text (string): what to say to the callee now.\n"
                "- hangup (boolean): true if the call should end now.\n"
                "- followup_prompt (string): if NOT hanging up, what question to ask next.\n"
                "Guidelines:\n"
                "- Keep response_text concise and natural for speech.\n"
                "- Only set hangup=true when the conversation is complete.\n"
            ),
            user=(f"Current question asked: {prompt_to_use}\n" f"Callee response (transcript): {transcript}\n"),
            temperature=0.2,
            max_tokens=220,
        )
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}

        response_text = str(parsed.get("response_text") or "").strip() or "Thanks — got it."
        hangup = bool(parsed.get("hangup")) if "hangup" in parsed else True
        followup_prompt = str(parsed.get("followup_prompt") or "").strip()
        if not hangup and not followup_prompt:
            followup_prompt = "Anything else you'd like to add?"

        tts = self.synthesize_for_call(call_id=call_id, text=response_text, tag="response")
        if not tts.get("success"):
            return {"success": False, "error": tts.get("error") or "tts_failed"}

        return {
            "success": True,
            "transcript": transcript,
            "response_text": response_text,
            "audio_url": tts.get("audio_url"),
            "hangup": hangup,
            "followup_prompt": followup_prompt,
        }
