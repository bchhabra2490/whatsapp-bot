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
import logging
from urllib.parse import urlencode
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import requests
from twilio.rest import Client as TwilioClient
from elevenlabs.client import ElevenLabs
import redis
import threading
import time

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
        self.deepgram_api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        self.elevenlabs_api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        self.elevenlabs_voice_id = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
        self.audio_dir = Path(os.getenv("CALL_AUDIO_DIR") or "generated_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.redis_url = (
            os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL") or "redis://localhost:6379/0"
        ).strip()
        self.call_state_ttl_seconds = int(os.getenv("CALL_STATE_TTL_SECONDS") or "86400")
        self._redis_client: Optional[redis.Redis] = None

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

    @staticmethod
    def _drop_recent_assistant_turns(history: list, max_drop: int = 2) -> list:
        """
        Remove the most recent assistant turns from history.
        Used for barge-in: if user interrupts playback, assume they did not hear
        the latest assistant response/follow-up.
        """
        if not isinstance(history, list) or not history:
            return []
        dropped = 0
        out = list(history)
        while out and dropped < max_drop:
            if (out[-1] or {}).get("role") == "assistant":
                out.pop()
                dropped += 1
            else:
                break
        return out

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
        # Persist initial call state in Redis so voice webhooks (separate process) can retrieve it.
        try:
            self._save_state(
                call_id,
                {
                    "call_id": call_id,
                    "requested_by": requested_by,
                    "to_number": normalized_to,
                    "prompt_question": prompt_question,
                    "history": [
                        {"role": "assistant", "text": prompt_question},
                    ],
                },
            )
        except Exception:
            # If Redis is unavailable, fall back to stateless query params + in-memory.
            pass

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
        logger.info(f"[CallService] outbound call started call_id={call_id} call_sid={call.sid} to={normalized_to}")
        return {"success": True, "call_id": call_id, "call_sid": call.sid, "to_number": normalized_to}

    def get_call_context(self, call_id: str) -> Optional[Dict[str, Any]]:
        # Prefer Redis state (works across processes); fall back to in-memory.
        state = self._load_state(call_id) if call_id else {}
        if state:
            return state
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

    def process_transcript(self, call_id: str, transcript: str, prompt_question: str = "") -> Dict[str, Any]:
        """
        Process a transcript (from realtime Deepgram or fallback recording):
        - append transcript to Redis history
        - run LLM to produce {response_text, hangup, followup_prompt}
        - synthesize ElevenLabs audio for response and followup (if needed)
        - persist decision + audio URLs in Redis
        """
        context = self.get_call_context(call_id) or {}
        prompt_to_use = (prompt_question or "").strip() or (str(context.get("prompt_question") or "").strip())
        if not prompt_to_use:
            return {"success": False, "error": "Unknown call context"}

        transcript = (transcript or "").strip() or "No speech recognized."
        logger.info(f"[CallService] process_transcript call_id={call_id} chars={len(transcript)}")

        # Update Redis history with the user's turn
        try:
            state = self._load_state(call_id) or {
                "call_id": call_id,
                "prompt_question": prompt_to_use,
                "history": [],
            }
            history = state.get("history") if isinstance(state.get("history"), list) else []
            history.append({"role": "user", "text": transcript})
            state["history"] = history
            state["prompt_question"] = prompt_to_use
            self._save_state(call_id, state)
            context = state
        except Exception:
            pass

        history_lines = []
        for turn in (context.get("history") or [])[:60]:
            role = turn.get("role") or "user"
            text = (turn.get("text") or "").strip()
            if text:
                history_lines.append(f"{role}: {text}")
        history_text = "\n".join(history_lines).strip() or "(none)"

        raw = self.openai.chat(
            system=(
                "You are controlling a phone-call assistant.\n"
                "Return STRICT JSON with keys:\n"
                "- response_text (string): what to say to the callee now.\n"
                "- hangup (boolean): true if the call should end now.\n"
                "- followup_prompt (string): if NOT hanging up, what question to ask next.\n"
                "Guidelines:\n"
                "- Keep response_text concise (ideally 1 short sentence).\n"
                "- Only set hangup=true when the conversation is complete.\n"
            ),
            user=(
                f"Conversation so far:\n{history_text}\n\n" f"Most recent callee response (transcript): {transcript}\n"
            ),
            temperature=0.2,
            max_tokens=140,
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
        logger.info(
            f"[CallService] llm decision call_id={call_id} hangup={hangup} response_chars={len(response_text)} followup_chars={len(followup_prompt)}"
        )

        response_tts = self.synthesize_for_call(call_id=call_id, text=response_text, tag="response")
        if not response_tts.get("success"):
            return {"success": False, "error": response_tts.get("error") or "tts_failed"}

        followup_audio_url = ""
        if not hangup:
            followup_tts = self.synthesize_for_call(call_id=call_id, text=followup_prompt, tag="followup")
            if followup_tts.get("success"):
                followup_audio_url = followup_tts.get("audio_url") or ""

        # Persist assistant turn + decision so /voice/play can serve it
        try:
            state = self._load_state(call_id) or context or {"call_id": call_id, "history": []}
            history = state.get("history") if isinstance(state.get("history"), list) else []
            history.append({"role": "assistant", "text": response_text})
            if not hangup:
                history.append({"role": "assistant", "text": followup_prompt})
            state.update(
                {
                    "history": history,
                    "hangup": hangup,
                    "response_text": response_text,
                    "followup_prompt": followup_prompt,
                    "response_audio_url": response_tts.get("audio_url"),
                    "followup_audio_url": followup_audio_url,
                    "prompt_question": followup_prompt if not hangup else prompt_to_use,
                }
            )
            self._save_state(call_id, state)
        except Exception:
            pass

        return {
            "success": True,
            "transcript": transcript,
            "response_text": response_text,
            "hangup": hangup,
            "followup_prompt": followup_prompt,
            "response_audio_url": response_tts.get("audio_url"),
            "followup_audio_url": followup_audio_url,
        }

    def start_streaming_response(self, call_id: str, transcript: str, prompt_question: str = "") -> Dict[str, Any]:
        """
        Start streaming LLM -> chunked ElevenLabs audio generation in a background thread.
        Stores chunk URLs in Redis for /voice/play to consume.
        """
        context = self.get_call_context(call_id) or {}
        prompt_to_use = (prompt_question or "").strip() or (str(context.get("prompt_question") or "").strip())
        if not prompt_to_use:
            return {"success": False, "error": "Unknown call context"}

        transcript = (transcript or "").strip() or "No speech recognized."
        logger.info(f"[CallService] start_streaming_response call_id={call_id} transcript_chars={len(transcript)}")

        # Append user turn to history
        try:
            state = self._load_state(call_id) or {"call_id": call_id, "history": []}
            history = state.get("history") if isinstance(state.get("history"), list) else []
            # If caller barged in during playback, discard the last assistant output from context.
            if bool(state.get("discard_last_assistant")):
                history = self._drop_recent_assistant_turns(history, max_drop=2)
                state["discard_last_assistant"] = False
            history.append({"role": "user", "text": transcript})
            state["history"] = history
            state["prompt_question"] = prompt_to_use
            self._save_state(call_id, state)
            context = state
        except Exception:
            pass

        history_lines = []
        for turn in (context.get("history") or [])[:60]:
            role = turn.get("role") or "user"
            text = (turn.get("text") or "").strip()
            if text:
                history_lines.append(f"{role}: {text}")
        history_text = "\n".join(history_lines).strip() or "(none)"

        # Fast control decision (hangup + followup) non-streaming
        raw = self.openai.chat(
            system=(
                "You are controlling a phone-call assistant.\n"
                "Return STRICT JSON with keys:\n"
                "- hangup (boolean)\n"
                "- followup_prompt (string)\n"
                "Guidelines:\n"
                "- If you need more info, set hangup=false and ask a concise followup.\n"
            ),
            user=(
                f"Conversation so far:\n{history_text}\n\n" f"Most recent callee response (transcript): {transcript}\n"
            ),
            temperature=0.2,
            max_tokens=80,
        )
        try:
            ctrl = json.loads(raw)
        except Exception:
            ctrl = {}
        hangup = bool(ctrl.get("hangup")) if "hangup" in ctrl else True
        followup_prompt = str(ctrl.get("followup_prompt") or "").strip()
        if not hangup and not followup_prompt:
            followup_prompt = "Anything else you'd like to add?"
        logger.info(f"[CallService] stream control call_id={call_id} hangup={hangup} followup_chars={len(followup_prompt)}")

        # Initialize streaming state
        try:
            self.update_state(
                call_id,
                {
                    "stream_mode": True,
                    "stream_audio_urls": [],
                    "stream_done": False,
                    "stream_error": "",
                    "hangup": hangup,
                    "followup_prompt": followup_prompt,
                    "followup_audio_url": "",
                    "playback_active": False,
                },
            )
        except Exception:
            pass

        def _should_flush(buf: str) -> bool:
            s = buf.strip()
            if len(s) >= 140:
                return True
            if s.endswith((".", "!", "?", "…")) and len(s) >= 40:
                return True
            return False

        def worker():
            chunk_idx = 0
            buf = ""
            full_text = ""
            try:
                # Stream response text tokens
                for delta in self.openai.chat_stream(
                    system=(
                        "You are speaking in a live phone call.\n"
                        "Respond naturally and succinctly.\n"
                        "Output only the words to say (no JSON).\n"
                    ),
                    user=(
                        f"Conversation so far:\n{history_text}\n\n"
                        f"Most recent callee response (transcript): {transcript}\n\n"
                        "Speak your response now:"
                    ),
                    temperature=0.2,
                    max_tokens=180,
                ):
                    buf += delta
                    full_text += delta
                    if _should_flush(buf):
                        text_chunk = buf.strip()
                        buf = ""
                        if text_chunk:
                            tts = self.synthesize_for_call(call_id=call_id, text=text_chunk, tag=f"chunk_{chunk_idx}")
                            if tts.get("success"):
                                # Append chunk URL
                                for _ in range(3):
                                    state = self._load_state(call_id) or {}
                                    urls = (
                                        state.get("stream_audio_urls")
                                        if isinstance(state.get("stream_audio_urls"), list)
                                        else []
                                    )
                                    urls.append(tts.get("audio_url"))
                                    state["stream_audio_urls"] = urls
                                    self._save_state(call_id, state)
                                    break
                                logger.info(f"[CallService] stream tts chunk ready call_id={call_id} chunk_idx={chunk_idx}")
                            chunk_idx += 1
                    # Small yield to avoid hogging CPU
                    time.sleep(0.01)

                # Flush remainder
                tail = buf.strip()
                if tail:
                    tts = self.synthesize_for_call(call_id=call_id, text=tail, tag=f"chunk_{chunk_idx}")
                    if tts.get("success"):
                        state = self._load_state(call_id) or {}
                        urls = (
                            state.get("stream_audio_urls") if isinstance(state.get("stream_audio_urls"), list) else []
                        )
                        urls.append(tts.get("audio_url"))
                        state["stream_audio_urls"] = urls
                        self._save_state(call_id, state)
                        logger.info(f"[CallService] stream tts tail ready call_id={call_id} chunk_idx={chunk_idx}")

                # Pre-generate followup prompt audio if continuing
                followup_audio_url = ""
                if not hangup:
                    try:
                        ttsf = self.synthesize_for_call(call_id=call_id, text=followup_prompt, tag="followup")
                        if ttsf.get("success"):
                            followup_audio_url = ttsf.get("audio_url") or ""
                    except Exception:
                        followup_audio_url = ""

                # Persist completion
                state = self._load_state(call_id) or {}
                state.update(
                    {
                        "stream_done": True,
                        "stream_error": "",
                        "response_text": full_text.strip(),
                        "followup_audio_url": followup_audio_url,
                    }
                )
                self._save_state(call_id, state)
                logger.info(
                    f"[CallService] stream complete call_id={call_id} total_chars={len(full_text.strip())} total_chunks={chunk_idx + (1 if tail else 0)}"
                )
            except Exception as e:
                state = self._load_state(call_id) or {}
                state.update({"stream_done": True, "stream_error": str(e)})
                self._save_state(call_id, state)
                logger.error(f"[CallService] stream worker failed call_id={call_id}: {e}", exc_info=True)

        threading.Thread(target=worker, daemon=True).start()
        return {"success": True, "hangup": hangup, "followup_prompt": followup_prompt}

    def process_recording(self, call_id: str, recording_url: str, prompt_question: str = "") -> Dict[str, Any]:
        context = self.get_call_context(call_id) or {}
        prompt_to_use = (prompt_question or "").strip() or (str(context.get("prompt_question") or "").strip())
        if not prompt_to_use:
            return {"success": False, "error": "Unknown call context"}

        if not recording_url:
            return {"success": False, "error": "Missing recording URL"}

        # Prefer MP3 to reduce latency and payload size; fall back to WAV.
        rec_resp = None
        content_type = ""
        try:
            rec_resp = requests.get(f"{recording_url}.mp3", timeout=45)
            rec_resp.raise_for_status()
            content_type = rec_resp.headers.get("Content-Type", "audio/mpeg")
        except Exception:
            rec_resp = requests.get(f"{recording_url}.wav", timeout=60)
            rec_resp.raise_for_status()
            content_type = rec_resp.headers.get("Content-Type", "audio/wav")

        transcript = self._transcribe_with_deepgram(rec_resp.content, content_type=content_type)
        return self.process_transcript(call_id=call_id, transcript=transcript, prompt_question=prompt_to_use)
