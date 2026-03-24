"""
WhatsApp Receipt Capture Bot
Main Flask application entry point
"""

import os
import logging
from urllib.parse import urlencode
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from services.supabase_client import SupabaseClient
from services.tasks import process_whatsapp_job
from services.openai_client import OpenAIClient
from services.call_service import CallService

load_dotenv()

app = Flask(__name__)

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

    vr = VoiceResponse()
    if not context and not prompt_from_query:
        vr.say("Sorry, the call session could not be found. Goodbye.")
        vr.hangup()
        return str(vr), 200, {"Content-Type": "application/xml"}

    prompt = prompt_from_query or context.get("prompt_question") or "I have a quick question for you."
    vr.say(f"Hello. This is an automated assistant call. {prompt}")
    action_query = urlencode({"call_id": call_id, "prompt_question": prompt})
    vr.record(
        max_length=20,
        play_beep=True,
        action=f"/voice/recording-complete?{action_query}",
        method="POST",
        timeout=4,
    )
    vr.say("No recording was received. Goodbye.")
    vr.hangup()
    return str(vr), 200, {"Content-Type": "application/xml"}


@app.route("/voice/recording-complete", methods=["POST"])
def voice_recording_complete():
    """
    Receives Twilio recording callback, transcribes audio with Deepgram,
    generates spoken reply with LLM + ElevenLabs, and plays it to callee.
    """
    call_id = (request.args.get("call_id") or "").strip()
    prompt_question = (request.args.get("prompt_question") or "").strip()
    recording_url = (request.form.get("RecordingUrl") or "").strip()
    vr = VoiceResponse()
    try:
        processed = call_service.process_recording(
            call_id=call_id,
            recording_url=recording_url,
            prompt_question=prompt_question,
        )
        if not processed.get("success"):
            vr.say("Sorry, I could not process your response right now. Goodbye.")
            vr.hangup()
            return str(vr), 200, {"Content-Type": "application/xml"}

        audio_url = (processed.get("audio_url") or "").strip()
        if audio_url:
            vr.play(audio_url)
        else:
            if processed.get("tts_error"):
                logger.error(f"ElevenLabs TTS failed, falling back to Twilio Say: {processed.get('tts_error')}")
            vr.say(processed.get("llm_response") or "Thank you for your response.")
        vr.say("Thank you. Goodbye.")
        vr.hangup()
        return str(vr), 200, {"Content-Type": "application/xml"}
    except Exception as e:
        logger.error(f"Voice recording handler failed: {e}", exc_info=True)
        vr.say("Sorry, something went wrong while processing this call. Goodbye.")
        vr.hangup()
        return str(vr), 200, {"Content-Type": "application/xml"}


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
