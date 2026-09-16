"""ASGI voice service: FastAPI + Pipecat bidirectional Twilio Media Streams."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from loguru import logger
from twilio.rest import Client as TwilioClient

from services.call_service import CallService
from services.openai_client import OpenAIClient
from services.twilio_security import candidate_urls, request_is_valid
from services.voice_pipeline import run_voice_pipeline

load_dotenv()

app = FastAPI(title="WhatsApp Bot Voice Service")

_call_service: Optional[CallService] = None


def get_call_service() -> CallService:
    global _call_service
    if _call_service is None:
        twilio_client = TwilioClient(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        _call_service = CallService(twilio_client=twilio_client, openai_client=OpenAIClient())
    return _call_service


def _forwarded_host(request_headers) -> str:
    return request_headers.get("x-forwarded-host") or request_headers.get("host") or ""


def _forwarded_proto(request_headers, fallback: str = "https") -> str:
    return request_headers.get("x-forwarded-proto") or fallback


def _http_signature_ok(request: Request, form: dict) -> bool:
    signature = request.headers.get("x-twilio-signature")
    urls = candidate_urls(
        forwarded_proto=_forwarded_proto(request.headers, request.url.scheme),
        forwarded_host=_forwarded_host(request.headers),
        path=request.url.path,
        query=request.url.query,
    )
    return request_is_valid(signature, urls, form)


def _ws_signature_ok(websocket: WebSocket) -> bool:
    signature = websocket.headers.get("x-twilio-signature")
    urls = candidate_urls(
        forwarded_proto=_forwarded_proto(websocket.headers, websocket.url.scheme or "https"),
        forwarded_host=_forwarded_host(websocket.headers),
        path=websocket.url.path,
        query=websocket.url.query,
    )
    params = dict(websocket.query_params)
    return request_is_valid(signature, urls, params)


@app.get("/voice/health")
async def voice_health():
    return {"status": "healthy", "service": "voice"}


@app.post("/voice/call")
async def voice_call(request: Request):
    """
    Twilio voice webhook. Returns <Connect><Stream> so the call stays on a
    persistent bidirectional WebSocket to Pipecat.
    """
    form = dict(await request.form())
    if not _http_signature_ok(request, form):
        logger.warning("[voice_call] invalid Twilio signature")
        return Response(content="Forbidden", status_code=403)

    call_service = get_call_service()
    call_id = (request.query_params.get("call_id") or form.get("call_id") or "").strip()
    context = call_service.get_call_context(call_id) if call_id else None
    if not context:
        logger.warning(f"[voice_call] unknown call_id={call_id}")
        return Response(content="<Response><Hangup/></Response>", media_type="application/xml")

    call_sid = (form.get("CallSid") or "").strip()
    if call_sid:
        try:
            call_service.update_state(call_id, {"call_sid": call_sid})
        except Exception:
            logger.exception("Failed to store call_sid")

    logger.info(f"[voice_call] connect stream call_id={call_id} call_sid={call_sid}")
    xml = call_service.connect_stream_twiml(call_id)
    return Response(content=xml, media_type="application/xml")


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Twilio Media Streams WebSocket — persistent bidirectional audio."""
    if not _ws_signature_ok(websocket):
        logger.warning("[voice_stream] invalid Twilio signature")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("[voice_stream] websocket accepted")

    try:
        from pipecat.runner.utils import parse_telephony_websocket

        _transport_type, call_data = await parse_telephony_websocket(websocket)
    except Exception:
        logger.exception("[voice_stream] failed to parse Twilio websocket")
        await websocket.close(code=1011)
        return

    try:
        await run_voice_pipeline(
            websocket=websocket,
            call_service=get_call_service(),
            call_data=call_data,
        )
    except WebSocketDisconnect:
        logger.info("[voice_stream] client disconnected")
    except Exception:
        logger.exception("[voice_stream] pipeline failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def _mount_flask_if_enabled() -> None:
    raw = (os.getenv("VOICE_MOUNT_FLASK") or "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return
    try:
        from fastapi.middleware.wsgi import WSGIMiddleware
        from app import app as flask_app

        app.mount("/", WSGIMiddleware(flask_app))
        logger.info("Mounted Flask WhatsApp app at /")
    except Exception:
        logger.exception("Failed to mount Flask app")


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy", "service": "voice"})


_mount_flask_if_enabled()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("VOICE_PORT", 7860)))
    uvicorn.run(
        "voice_service:app",
        host="0.0.0.0",
        port=port,
        ws="websockets",
        timeout_keep_alive=75,
        proxy_headers=True,
        forwarded_allow_ips="*",
        workers=1,
    )
