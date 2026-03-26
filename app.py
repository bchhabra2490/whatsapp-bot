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
from twilio.rest import Client as TwilioClient
from xml.sax.saxutils import escape

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

    action_query = urlencode({"call_id": call_id, "prompt_question": prompt})
    record_action = f"{call_service.base_url}/voice/recording-complete?{action_query}"
    xml = (
        "<Response>"
        f"<Play>{escape(prompt_tts.get('audio_url') or '')}</Play>"
        f'<Record maxLength="20" playBeep="true" timeout="4" method="POST" action="{escape(record_action)}"/>'
        "</Response>"
    )
    return xml, 200, {"Content-Type": "application/xml"}


@app.route("/voice/recording-complete", methods=["POST"])
def voice_recording_complete():
    """
    Receives Twilio recording callback, transcribes audio with Deepgram,
    generates spoken reply with LLM + ElevenLabs, and plays it to callee.
    """
    call_id = (request.args.get("call_id") or "").strip()
    prompt_question = (request.args.get("prompt_question") or "").strip()
    recording_url = (request.form.get("RecordingUrl") or "").strip()
    print(f"Recording URL: {recording_url}")
    print(f"Prompt Question: {prompt_question}")
    print(f"Call ID: {call_id}")
    try:
        processed = call_service.process_recording(
            call_id=call_id,
            recording_url=recording_url,
            prompt_question=prompt_question,
        )
        print(f"Processed: {processed}")
        if not processed.get("success"):
            xml = "<Response><Hangup/></Response>"
            return xml, 200, {"Content-Type": "application/xml"}

        audio_url = (processed.get("audio_url") or "").strip()
        print(f"Audio URL: {audio_url}")
        hangup = bool(processed.get("hangup"))
        followup_prompt = (processed.get("followup_prompt") or "").strip()

        # Play LLM response via ElevenLabs
        parts = ["<Response>"]
        if audio_url:
            parts.append(f"<Play>{escape(audio_url)}</Play>")

        if hangup:
            parts.append("<Hangup/>")
            parts.append("</Response>")
            return "".join(parts), 200, {"Content-Type": "application/xml"}

        # Not hanging up: ask a follow-up and record again (all via ElevenLabs audio)
        followup_tts = call_service.synthesize_for_call(call_id=call_id, text=followup_prompt, tag="followup")
        if followup_tts.get("success") and followup_tts.get("audio_url"):
            parts.append(f"<Play>{escape(followup_tts.get('audio_url'))}</Play>")

        next_query = urlencode({"call_id": call_id, "prompt_question": followup_prompt})
        next_action = f"{call_service.base_url}/voice/recording-complete?{next_query}"
        parts.append(
            f'<Record maxLength="20" playBeep="true" timeout="4" method="POST" action="{escape(next_action)}"/>'
        )
        parts.append("</Response>")
        return "".join(parts), 200, {"Content-Type": "application/xml"}
    except Exception as e:
        logger.error(f"Voice recording handler failed: {e}", exc_info=True)
        xml = "<Response><Hangup/></Response>"
        return xml, 200, {"Content-Type": "application/xml"}


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
