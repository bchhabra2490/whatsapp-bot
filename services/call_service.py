"""
Orchestrates outbound voice calls:
- Twilio outbound call + Connect/Stream TwiML
- Redis call state shared with the Pipecat voice service
"""

import os
import re
import uuid
import json
import logging
from urllib.parse import urlencode, urlparse
from threading import Lock
from typing import Any, Dict, Optional

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse
import redis

from services.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class CallService:
    """Coordinates phone-call workflows from text instructions."""

    _pending_calls: Dict[str, Dict[str, Any]] = {}
    _lock: Lock = Lock()

    def __init__(self, twilio_client: TwilioClient, openai_client: OpenAIClient):
        self.twilio_client = twilio_client
        self.openai = openai_client
        self.twilio_voice_from = (os.getenv("TWILIO_VOICE_NUMBER") or "").strip()
        self.base_url = (os.getenv("APP_BASE_URL") or "").strip().rstrip("/")
        self.redis_url = (
            os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL") or "redis://localhost:6379/0"
        ).strip()
        self.call_state_ttl_seconds = int(os.getenv("CALL_STATE_TTL_SECONDS") or "86400")
        self._redis_client: Optional[redis.Redis] = None

    def _redis_target(self) -> str:
        try:
            parsed = urlparse(self.redis_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379
            db = (parsed.path or "/0").lstrip("/") or "0"
            return f"{host}:{port}/{db}"
        except Exception:
            return "unknown"

    def ping_redis(self) -> bool:
        """Ping Redis and log whether the connection works."""
        target = self._redis_target()
        try:
            self._redis().ping()
            logger.info(f"[CallService] Redis connected at {target}")
            return True
        except Exception as e:
            logger.error(f"[CallService] Redis not connected at {target}: {e}")
            return False

    @property
    def voice_base_url(self) -> str:
        return (os.getenv("VOICE_BASE_URL") or self.base_url).strip().rstrip("/")

    def media_stream_ws_url(self) -> str:
        base = self.voice_base_url
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/voice/stream"

    def _redis(self) -> redis.Redis:
        if self._redis_client is None:
            # decode_responses=False because we store raw JSON bytes
            self._redis_client = redis.Redis.from_url(self.redis_url, decode_responses=False)
        return self._redis_client

    @staticmethod
    def _state_key(call_id: str) -> str:
        return f"wbot:call:{call_id}"

    def _load_state(self, call_id: str) -> Dict[str, Any]:
        try:
            raw = self._redis().get(self._state_key(call_id))
            if not raw:
                return {}
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _save_state(self, call_id: str, state: Dict[str, Any]) -> None:
        payload = json.dumps(state, ensure_ascii=False).encode("utf-8")
        self._redis().setex(self._state_key(call_id), self.call_state_ttl_seconds, payload)

    def update_state(self, call_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge updates into Redis state and persist.
        Returns the merged state.
        """
        state = self._load_state(call_id) or {"call_id": call_id, "history": []}
        state.update(updates or {})
        self._save_state(call_id, state)
        if updates:
            logger.info(f"[CallService] state updated call_id={call_id} keys={list(updates.keys())}")
        return state

    def append_transcript_turn(self, call_id: str, role: str, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not call_id or not text:
            return self.get_call_context(call_id) or {}
        state = self._load_state(call_id) or {"call_id": call_id, "history": []}
        history = state.get("history") if isinstance(state.get("history"), list) else []
        if history:
            last = history[-1] or {}
            if last.get("role") == role and last.get("text") == text:
                return state
        history.append({"role": role, "text": text})
        state["history"] = history
        self._save_state(call_id, state)
        return state

    def persist_call_end(self, call_id: str, history: Optional[list] = None) -> Dict[str, Any]:
        """Write hangup + transcript to Redis. Does not call OpenAI or Twilio."""
        updates: Dict[str, Any] = {"hangup": True}
        if history:
            updates["history"] = history
        state = self.get_call_context(call_id) or {}
        if not (state.get("final_summary") or "").strip():
            reason = (state.get("hangup_reason") or "").strip()
            if reason == "idle_timeout":
                updates["final_summary"] = "The callee did not respond after a follow-up prompt, so the call ended."
            elif reason == "max_duration":
                updates["final_summary"] = "The call reached the maximum duration and was ended."
        return self.update_state(call_id, updates)

    @staticmethod
    def _normalize_phone_number(raw: str) -> str:
        digits = re.sub(r"[^\d+]", "", (raw or "").strip())
        if digits.startswith("+"):
            return digits
        if digits:
            return f"+{digits}"
        return ""

    @staticmethod
    def _normalize_whatsapp_number(raw: str) -> str:
        to_number = (raw or "").strip()
        if not to_number:
            return ""
        if not to_number.lower().startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}" if to_number.startswith("+") else f"whatsapp:+{to_number}"
        return to_number

    @staticmethod
    def _twilio_whatsapp_from() -> str:
        twilio_from = (os.getenv("TWILIO_WHATSAPP_NUMBER") or "").strip()
        if not twilio_from:
            return ""
        if not twilio_from.lower().startswith("whatsapp:"):
            twilio_from = f"whatsapp:{twilio_from}" if twilio_from.startswith("+") else f"whatsapp:+{twilio_from}"
        return twilio_from

    def connect_stream_twiml(self, call_id: str) -> str:
        """Return bidirectional Media Stream TwiML for the Pipecat WebSocket."""
        response = VoiceResponse()
        connect = Connect()
        stream = Stream(url=self.media_stream_ws_url())
        stream.parameter(name="call_id", value=call_id or "")
        connect.append(stream)
        response.append(connect)
        return str(response)

    def start_outbound_call(
        self, requested_by: str, to_number: str, prompt_question: str, purpose_of_call: str
    ) -> Dict[str, Any]:
        if not self.twilio_voice_from:
            return {"success": False, "error": "TWILIO_VOICE_NUMBER is not configured"}
        if not self.voice_base_url:
            return {"success": False, "error": "VOICE_BASE_URL or APP_BASE_URL is not configured"}

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
                "purpose_of_call": purpose_of_call,
            }
        try:
            self._save_state(
                call_id,
                {
                    "call_id": call_id,
                    "requested_by": requested_by,
                    "to_number": normalized_to,
                    "prompt_question": prompt_question,
                    "purpose_of_call": purpose_of_call,
                    "history": [
                        {"role": "assistant", "text": prompt_question},
                    ],
                    "hangup": False,
                    "final_summary": "",
                },
            )
        except Exception:
            pass

        query = urlencode({"call_id": call_id})
        call = self.twilio_client.calls.create(
            to=normalized_to,
            from_=self.twilio_voice_from,
            url=f"{self.voice_base_url}/voice/call?{query}",
            method="POST",
        )
        try:
            self.update_state(call_id, {"call_sid": call.sid})
        except Exception:
            pass
        logger.info(f"[CallService] outbound call started call_id={call_id} call_sid={call.sid} to={normalized_to}")
        return {"success": True, "call_id": call_id, "call_sid": call.sid, "to_number": normalized_to}

    def get_call_context(self, call_id: str) -> Optional[Dict[str, Any]]:
        state = self._load_state(call_id) if call_id else {}
        if state:
            return state
        with self._lock:
            return self._pending_calls.get(call_id)

    def hangup_twilio_call(self, call_sid: str, reason: str = "") -> bool:
        """Force-complete a Twilio call. Last resort if the Pipecat pipeline does not end."""
        call_sid = (call_sid or "").strip()
        if not call_sid:
            return False
        try:
            self.twilio_client.calls(call_sid).update(status="completed")
            logger.info(f"[CallService] twilio hangup call_sid={call_sid} reason={reason or 'unspecified'}")
            return True
        except Exception as e:
            logger.error(f"[CallService] twilio hangup failed call_sid={call_sid}: {e}", exc_info=True)
            return False

    def send_call_summary_to_whatsapp(self, call_id: str, *, raise_on_error: bool = False) -> bool:
        """
        Send a call summary to the WhatsApp requester.
        Safe to call more than once; Redis `summary_sent` makes it idempotent.
        """
        try:
            state = self.get_call_context(call_id) or {}
            if state.get("summary_sent"):
                return True
            requested_by = self._normalize_whatsapp_number(state.get("requested_by") or "")
            twilio_from = self._twilio_whatsapp_from()
            if not (requested_by and twilio_from):
                logger.error(
                    f"[call_summary] missing numbers call_id={call_id} requested_by={requested_by} twilio_from={twilio_from}"
                )
                return False

            purpose = (state.get("purpose_of_call") or state.get("purpose") or "").strip()
            existing_summary = (state.get("final_summary") or "").strip()
            history = state.get("history") if isinstance(state.get("history"), list) else []
            lines = []
            for turn in history[-20:]:
                role = (turn.get("role") or "user").strip()
                text = (turn.get("text") or "").strip()
                if text:
                    lines.append(f"{role}: {text}")
            transcript = "\n".join(lines).strip()

            summary = existing_summary
            if not summary:
                summary = self.openai.chat(
                    system=(
                        "Summarize a phone call for the person who requested the call.\n"
                        "Be concise and actionable. Do not mention internal tools.\n"
                        "Return plain text.\n"
                    ),
                    user=(
                        f"Purpose of call:\n{purpose or '(not provided)'}\n\n"
                        f"Conversation (most recent last):\n{transcript or '(no transcript)'}\n\n"
                        "Write:\n"
                        "- 3-6 bullet summary\n"
                        "- Any commitments / next steps\n"
                    ),
                    temperature=0.2,
                    max_tokens=250,
                ).strip()

            body = f"📞 Call summary\n\n{summary}".strip()
            self.twilio_client.messages.create(from_=twilio_from, to=requested_by, body=body)
            self.update_state(call_id, {"summary_sent": True, "final_summary": summary})
            logger.info(f"[call_summary] sent to={requested_by} call_id={call_id}")
            return True
        except Exception as e:
            logger.error(f"[call_summary] failed call_id={call_id}: {e}", exc_info=True)
            if raise_on_error:
                raise
            return False
