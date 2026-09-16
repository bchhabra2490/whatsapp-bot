"""Pipecat bidirectional voice pipeline for Twilio Media Streams."""

from __future__ import annotations

import os
from typing import Any, Dict

from loguru import logger

from services.call_service import CallService

VOICE_SYSTEM_PROMPT = """You are speaking on a live phone call.

Speak in at most 1-2 short sentences.
Use no lists, preambles, repetition, or explanations.
Ask only one question at a time.
Stop as soon as the call objective is achieved.
Never ask if there is anything else.
When the objective is achieved, say a brief one-sentence closing and immediately call complete_call.
Do not mention tools or that you are an AI.
Output only the words to say.
If you are not sure about some information, do not make it up. Use harmless defaults.
"""


def _call_data_value(call_data: Any, *names: str, default: str = "") -> str:
    for name in names:
        if call_data is None:
            break
        if isinstance(call_data, dict):
            value = call_data.get(name)
        else:
            value = getattr(call_data, name, None)
            if value is None and hasattr(call_data, "get"):
                value = call_data.get(name)
        if value:
            return str(value)
    return default


def _call_data_body(call_data: Any) -> Dict[str, Any]:
    if call_data is None:
        return {}
    if isinstance(call_data, dict):
        raw = call_data.get("body") or {}
        return raw if isinstance(raw, dict) else {}
    raw = getattr(call_data, "body", None)
    if raw is None and hasattr(call_data, "get"):
        raw = call_data.get("body")
    return raw if isinstance(raw, dict) else {}


def _end_frame_cls():
    try:
        from pipecat.frames.frames import EndWorkerFrame

        return EndWorkerFrame
    except ImportError:
        from pipecat.frames.frames import EndTaskFrame

        return EndTaskFrame


def _message_role_and_text(msg: Any) -> tuple[str, str]:
    if isinstance(msg, dict):
        role = msg.get("role") or ""
        content = msg.get("content")
    else:
        role = getattr(msg, "role", "") or ""
        content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
            elif isinstance(item, str):
                parts.append(item)
        content = " ".join(p for p in parts if p).strip()
    text = content.strip() if isinstance(content, str) else ""
    return str(role), text


async def run_voice_pipeline(
    *,
    websocket,
    call_service: CallService,
    call_data: Any = None,
) -> None:
    from pipecat.adapters.schemas.direct_function import tool_options
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.llm_service import FunctionCallParams
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.tts_service import TextAggregationMode
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
    from pipecat.workers.runner import WorkerRunner

    EndFrameCls = _end_frame_cls()

    stream_sid = _call_data_value(call_data, "stream_id", "stream_sid")
    twilio_call_sid = _call_data_value(call_data, "call_id", "call_sid")
    body = _call_data_body(call_data)
    call_id = str(body.get("call_id") or "").strip()

    context_state = call_service.get_call_context(call_id) if call_id else {}
    purpose = str((context_state or {}).get("purpose_of_call") or "").strip()
    opening = str((context_state or {}).get("prompt_question") or "").strip() or "I have a quick question for you."

    if call_id and twilio_call_sid:
        try:
            call_service.update_state(call_id, {"call_sid": twilio_call_sid})
        except Exception:
            logger.exception("Failed to persist Twilio call SID")

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=twilio_call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID") or "",
        auth_token=os.getenv("TWILIO_AUTH_TOKEN") or "",
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            session_timeout=None,
        ),
    )

    endpointing_ms = int(os.getenv("DEEPGRAM_ENDPOINTING_MS") or "250")
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model=os.getenv("DEEPGRAM_MODEL") or "nova-3-general",
            language="en",
            interim_results=True,
            punctuate=True,
            smart_format=True,
            endpointing=endpointing_ms,
        ),
    )

    llm_max_tokens = int(os.getenv("VOICE_LLM_MAX_TOKENS") or "50")
    system_instruction = f"{VOICE_SYSTEM_PROMPT}\nThe purpose of this call is: {purpose or opening}\n"
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
            system_instruction=system_instruction,
            temperature=0.2,
            max_tokens=llm_max_tokens,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model=os.getenv("ELEVENLABS_MODEL_ID") or "eleven_flash_v2_5",
        ),
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )

    @tool_options(cancel_on_interruption=False)
    async def complete_call(params: FunctionCallParams, summary: str):
        """End the call now because the objective is achieved. Never ask another question.

        Args:
            summary: One-sentence outcome for the person who requested this call.
        """
        if call_id:
            call_service.update_state(
                call_id,
                {
                    "hangup": True,
                    "final_summary": (summary or "").strip(),
                },
            )
        await params.result_callback({"status": "ended"})
        await params.llm.push_frame(EndFrameCls(), FrameDirection.DOWNSTREAM)

    context = LLMContext(tools=[complete_call])
    vad_stop_secs = float(os.getenv("VOICE_VAD_STOP_SECS") or "0.25")
    vad_start_secs = float(os.getenv("VOICE_VAD_START_SECS") or "0.15")
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    start_secs=vad_start_secs,
                    stop_secs=vad_stop_secs,
                )
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = WorkerRunner(handle_sigint=False, force_gc=True)
    await runner.add_workers(worker)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        content = getattr(message, "content", None)
        if call_id and content:
            call_service.append_transcript_turn(call_id, "user", str(content))

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        content = getattr(message, "content", None)
        if call_id and content:
            call_service.append_transcript_turn(call_id, "assistant", str(content))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"[voice] client connected call_id={call_id} call_sid={twilio_call_sid}")
        await worker.queue_frames([TTSSpeakFrame(opening)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[voice] client disconnected call_id={call_id}")
        await runner.cancel()

    summarized = {"done": False}

    async def _persist_and_summarize() -> None:
        if not call_id or summarized["done"]:
            return
        summarized["done"] = True
        try:
            messages = context.get_messages() if hasattr(context, "get_messages") else []
            history = []
            for msg in messages or []:
                role, text = _message_role_and_text(msg)
                if role in {"user", "assistant"} and text:
                    history.append({"role": role, "text": text})
            updates: Dict[str, Any] = {"hangup": True}
            if history:
                updates["history"] = history
            call_service.update_state(call_id, updates)
        except Exception:
            logger.exception(f"[voice] failed to persist final history call_id={call_id}")
        try:
            call_service.send_call_summary_to_whatsapp(call_id)
        except Exception:
            logger.exception(f"[voice] failed to send WhatsApp summary call_id={call_id}")

    @worker.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(worker, frame):
        await _persist_and_summarize()

    try:
        await runner.run()
    finally:
        await _persist_and_summarize()
