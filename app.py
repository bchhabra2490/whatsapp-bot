"""
WhatsApp Receipt Capture Bot
Main Flask application entry point
"""

import os
import logging
import base64
import json
import threading
from urllib.parse import urlencode
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from xml.sax.saxutils import escape
from flask_sock import Sock
import websocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from services.supabase_client import SupabaseClient
from services.tasks import process_whatsapp_job
from services.openai_client import OpenAIClient
from services.call_service import CallService

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

# Initialize services
supabase_client = SupabaseClient()
twilio_client = TwilioClient(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
openai_client = OpenAIClient()
call_service = CallService(twilio_client=twilio_client, openai_client=openai_client)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    """Twilio WhatsApp webhook handler"""
    try:
        # Get incoming message data
        incoming_message = request.form.get("Body", "")
        print(f"Incoming message: {incoming_message}")
        # Twilio sends media as MediaUrl0, MediaUrl1, etc. and MediaContentType0 for the first.
        media_urls = [
            url
            for url in [
                request.form.get("MediaUrl0"),
                request.form.get("MediaUrl1"),
                request.form.get("MediaUrl2"),
            ]
            if url
        ]
        media_content_type0 = (request.form.get("MediaContentType0") or "").strip().lower()
        # Twilio sends Latitude, Longitude, Address, Label for shared location
        latitude = request.form.get("Latitude", "").strip()
        longitude = request.form.get("Longitude", "").strip()
        address = (request.form.get("Address") or "").strip()
        label = (request.form.get("Label") or "").strip()
        has_location = latitude and longitude
        print(f"Media URLs: {media_urls}, MediaContentType0: {media_content_type0}, Location: {latitude},{longitude}")
        from_number = request.form.get("From", "")
        print(f"From number: {from_number}")
        message_sid = request.form.get("MessageSid", "")
        print(f"Message SID: {message_sid}")

        # Create Twilio response
        resp = MessagingResponse()

        # Build a background job (location | media | audio | text)
        job_type = "text"
        payload: dict = {}
        if has_location:
            job_type = "location"
            payload = {
                "latitude": latitude,
                "longitude": longitude,
                "address": address or None,
                "label": label or None,
            }
        elif media_urls:
            if media_content_type0.startswith("audio/"):
                job_type = "audio"
                payload = {"media_urls": media_urls}
            else:
                job_type = "media"
                payload = {"media_urls": media_urls, "incoming_text": incoming_message}
        elif incoming_message:
            payload = {"text": incoming_message}

        if not payload:
            resp.message("Please send an image, PDF, voice note, location, or text message.")
            return str(resp), 200

        # Persist the incoming message immediately
        try:
            supabase_client.save_message(
                {
                    "phone_number": from_number,
                    "direction": "in",
                    "role": "user",
                    "message_sid": message_sid,
                    "content": incoming_message
                    or (
                        f"[location] {latitude},{longitude}"
                        if has_location
                        else (f"[media] {', '.join(media_urls)}" if media_urls else "")
                    ),
                    "metadata": (
                        {"media_urls": media_urls}
                        if media_urls
                        else ({"latitude": latitude, "longitude": longitude} if has_location else {})
                    ),
                }
            )
        except Exception as e:
            logger.error(f"Failed to save incoming message: {e}", exc_info=True)

        # Create job in DB
        job = supabase_client.create_job(
            {
                "phone_number": from_number,
                "message_sid": message_sid,
                "job_type": job_type,
                "payload": payload,
            }
        )

        # Enqueue Celery task
        try:
            process_whatsapp_job.delay(str(job.get("id")))
            # resp.message("✅ Got your message. I'm processing it in the background and will reply shortly.")
        except Exception as e:
            logger.error(f"Failed to enqueue background job: {e}", exc_info=True)
            resp.message("❌ Sorry, I couldn't start processing your message. Please try again later.")

        return str(resp), 200

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        resp = MessagingResponse()
        resp.message("Sorry, an error occurred processing your request. Please try again.")
        return str(resp), 500


@app.route("/voice/call", methods=["POST"])
def voice_call():
    """
    Twilio voice webhook for active outbound calls.
    Prompts recipient and records their answer.
    """
    call_id = (request.args.get("call_id") or "").strip()
    prompt_from_query = (request.args.get("prompt_question") or "").strip()
    context = call_service.get_call_context(call_id) if call_id else None
    print(f"Context: {context}")

    if not context and not prompt_from_query:
        # Return valid TwiML even on error.
        xml = "<Response><Hangup/></Response>"
        return xml, 200, {"Content-Type": "application/xml"}

    prompt = prompt_from_query or context.get("prompt_question") or "I have a quick question for you."
    # Avoid Twilio <Say> (robotic). Use ElevenLabs audio + <Play>.
    prompt_tts = call_service.synthesize_for_call(call_id=call_id, text=prompt, tag="prompt")
    if not prompt_tts.get("success"):
        xml = "<Response><Hangup/></Response>"
        return xml, 200, {"Content-Type": "application/xml"}

    # Keep prompt question in Redis so stream processing has the correct context.
    try:
        call_service.update_state(call_id, {"prompt_question": prompt})
    except Exception:
        pass

    # Start Twilio Media Stream (realtime) to /voice/stream
    ws_base = call_service.base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_query = urlencode({"call_id": call_id})
    stream_url = f"{ws_base}/voice/stream?{stream_query}"
    xml = (
        "<Response>"
        f"<Play>{escape(prompt_tts.get('audio_url') or '')}</Play>"
        "<Start>"
        f'<Stream url="{escape(stream_url)}" />'
        "</Start>"
        # Keep call alive while stream runs; call control will redirect to /voice/play when ready.
        '<Pause length="600" />'
        "</Response>"
    )
    return xml, 200, {"Content-Type": "application/xml"}


@app.route("/voice/play", methods=["POST", "GET"])
def voice_play():
    """
    TwiML endpoint used after a transcript is finalized.
    Plays the latest response (and followup prompt if continuing), then either hangs up
    or redirects back to /voice/stream-start to listen again.
    """
    call_id = (request.args.get("call_id") or "").strip()
    idx = int(request.args.get("idx") or "0")
    state = call_service.get_call_context(call_id) or {}
    stream_mode = bool(state.get("stream_mode"))
    stream_urls = state.get("stream_audio_urls") if isinstance(state.get("stream_audio_urls"), list) else []
    stream_done = bool(state.get("stream_done"))
    response_audio_url = (state.get("response_audio_url") or "").strip()
    followup_audio_url = (state.get("followup_audio_url") or "").strip()
    hangup = bool(state.get("hangup"))

    parts = ["<Response>"]
    if stream_mode:
        # Play next available chunk; if not ready, poll briefly.
        if idx < len(stream_urls) and stream_urls[idx]:
            parts.append(f"<Play>{escape(stream_urls[idx])}</Play>")
            next_url = f"{call_service.base_url}/voice/play?{urlencode({'call_id': call_id, 'idx': idx + 1})}"
            parts.append(f'<Redirect method="POST">{escape(next_url)}</Redirect>')
            parts.append("</Response>")
            return "".join(parts), 200, {"Content-Type": "application/xml"}

        if not stream_done:
            # Wait a moment for next chunk to be generated
            parts.append('<Pause length="1" />')
            retry_url = f"{call_service.base_url}/voice/play?{urlencode({'call_id': call_id, 'idx': idx})}"
            parts.append(f'<Redirect method="POST">{escape(retry_url)}</Redirect>')
            parts.append("</Response>")
            return "".join(parts), 200, {"Content-Type": "application/xml"}

        # Stream finished but no more chunks to play
        if not hangup and followup_audio_url:
            parts.append(f"<Play>{escape(followup_audio_url)}</Play>")
        if hangup:
            parts.append("<Hangup/>")
            parts.append("</Response>")
            return "".join(parts), 200, {"Content-Type": "application/xml"}

        redirect_url = f"{call_service.base_url}/voice/stream-start?{urlencode({'call_id': call_id})}"
        parts.append(f'<Redirect method="POST">{escape(redirect_url)}</Redirect>')
        parts.append("</Response>")
        return "".join(parts), 200, {"Content-Type": "application/xml"}

    # Non-stream legacy mode (single audio)
    if response_audio_url:
        parts.append(f"<Play>{escape(response_audio_url)}</Play>")
    if not hangup and followup_audio_url:
        parts.append(f"<Play>{escape(followup_audio_url)}</Play>")

    if hangup:
        parts.append("<Hangup/>")
        parts.append("</Response>")
        return "".join(parts), 200, {"Content-Type": "application/xml"}

    # Resume streaming for next user response
    redirect_url = f"{call_service.base_url}/voice/stream-start?{urlencode({'call_id': call_id})}"
    parts.append(f'<Redirect method="POST">{escape(redirect_url)}</Redirect>')
    parts.append("</Response>")
    return "".join(parts), 200, {"Content-Type": "application/xml"}


@app.route("/voice/stream-start", methods=["POST", "GET"])
def voice_stream_start():
    """
    Starts / restarts Twilio Media Stream for the call.
    """
    call_id = (request.args.get("call_id") or "").strip()
    ws_base = call_service.base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_base}/voice/stream?{urlencode({'call_id': call_id})}"
    xml = (
        "<Response>"
        "<Start>"
        f'<Stream url="{escape(stream_url)}" />'
        "</Start>"
        '<Pause length="600" />'
        "</Response>"
    )
    return xml, 200, {"Content-Type": "application/xml"}


@sock.route("/voice/stream")
def voice_stream(ws):
    """
    Twilio Media Stream WebSocket:
    - Receives Twilio audio frames (mulaw/8khz) over websocket
    - Forwards them to Deepgram realtime websocket
    - On final transcript, runs LLM->ElevenLabs and redirects the live call to /voice/play
    """
    call_id = (request.args.get("call_id") or "").strip()
    if not call_id:
        return

    deepgram_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
    if not deepgram_key:
        logger.error("DEEPGRAM_API_KEY missing; cannot stream")
        return

    # Deepgram realtime expects raw audio bytes. Twilio sends base64 mulaw (8khz mono).
    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?encoding=mulaw&sample_rate=8000&channels=1"
        "&model=nova-2&smart_format=true&punctuate=true"
        "&interim_results=true&endpointing=200"
    )

    final_transcript = {"text": ""}
    call_sid_box = {"sid": ""}
    done = threading.Event()

    def on_dg_message(_ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return

        # Deepgram JSON shape: results.channels[0].alternatives[0].transcript
        transcript = (
            data.get("channel", {}).get("alternatives", [{}])[0].get("transcript")
            if isinstance(data.get("channel"), dict)
            else data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript")
        )
        if not transcript:
            return

        is_final = bool(data.get("is_final") or data.get("speech_final") or data.get("final"))
        if is_final:
            final_transcript["text"] = transcript.strip()
            done.set()

    def on_dg_error(_ws, error):
        logger.error(f"Deepgram ws error: {error}")
        done.set()

    def on_dg_close(_ws, status_code, msg):
        done.set()

    dg_ws = websocket.WebSocketApp(
        dg_url,
        header=[f"Authorization: Token {deepgram_key}"],
        on_message=on_dg_message,
        on_error=on_dg_error,
        on_close=on_dg_close,
    )

    dg_thread = threading.Thread(target=lambda: dg_ws.run_forever(ping_interval=20, ping_timeout=10), daemon=True)
    dg_thread.start()

    # Wait briefly for Deepgram to connect
    for _ in range(50):
        if dg_ws.sock and dg_ws.sock.connected:
            break
        if done.is_set():
            break
        threading.Event().wait(0.02)

    try:
        while not done.is_set():
            raw = ws.receive()
            if not raw:
                break
            try:
                evt = json.loads(raw)
            except Exception:
                continue

            etype = evt.get("event")
            if etype == "start":
                start = evt.get("start") or {}
                call_sid = start.get("callSid") or start.get("call_sid") or ""
                if call_sid:
                    call_sid_box["sid"] = call_sid
                    try:
                        call_service.update_state(call_id, {"call_sid": call_sid})
                    except Exception:
                        pass
                continue

            if etype == "media":
                media = evt.get("media") or {}
                payload_b64 = media.get("payload") or ""
                if not payload_b64:
                    continue
                try:
                    audio_bytes = base64.b64decode(payload_b64)
                except Exception:
                    continue
                try:
                    if dg_ws.sock and dg_ws.sock.connected:
                        dg_ws.send(audio_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    # if DG send fails, stop to avoid spinning
                    done.set()
                continue

            if etype == "stop":
                done.set()
                break
    finally:
        try:
            dg_ws.close()
        except Exception:
            pass

    # If we got a final transcript, start streaming response generation and redirect to /voice/play
    transcript_text = (final_transcript.get("text") or "").strip()
    if not transcript_text:
        return

    started = call_service.start_streaming_response(call_id=call_id, transcript=transcript_text)
    if not started.get("success"):
        logger.error(f"start_streaming_response failed: {started}")
        return

    # Redirect call to /voice/play (Twilio will fetch TwiML and play audio)
    call_sid = call_sid_box.get("sid") or (call_service.get_call_context(call_id) or {}).get("call_sid") or ""
    if not call_sid:
        logger.error("Missing call_sid; cannot redirect to /voice/play")
        return

    try:
        play_url = f"{call_service.base_url}/voice/play?{urlencode({'call_id': call_id, 'idx': 0})}"
        call_service.twilio_client.calls(call_sid).update(url=play_url, method="POST")
    except Exception as e:
        logger.error(f"Twilio redirect failed: {e}", exc_info=True)


@app.route("/audio/<path:filename>", methods=["GET"])
def serve_generated_audio(filename: str):
    """
    Serves generated ElevenLabs MP3 files back to Twilio <Play>.
    """
    directory = os.getenv("CALL_AUDIO_DIR") or "generated_audio"
    return send_from_directory(directory, filename, mimetype="audio/mpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "False") == "True")
