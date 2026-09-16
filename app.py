"""
WhatsApp Receipt Capture Bot
Main Flask application entry point
"""

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

from services.supabase_client import SupabaseClient
from services.tasks import process_whatsapp_job
from services.twilio_security import candidate_urls, request_is_valid

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

supabase_client = SupabaseClient()


def _twilio_signature_ok() -> bool:
    signature = request.headers.get("X-Twilio-Signature")
    urls = candidate_urls(
        forwarded_proto=request.headers.get("X-Forwarded-Proto") or request.scheme,
        forwarded_host=request.headers.get("X-Forwarded-Host") or request.host,
        path=request.path,
        query=request.query_string.decode("utf-8") if request.query_string else "",
    )
    return request_is_valid(signature, urls, request.form)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    """Twilio WhatsApp webhook handler"""
    try:
        if not _twilio_signature_ok():
            logger.warning("[webhook] invalid Twilio signature")
            return "Forbidden", 403

        incoming_message = request.form.get("Body", "")
        logger.info(f"[webhook] incoming message body='{incoming_message[:120]}'")
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
        latitude = request.form.get("Latitude", "").strip()
        longitude = request.form.get("Longitude", "").strip()
        address = (request.form.get("Address") or "").strip()
        label = (request.form.get("Label") or "").strip()
        has_location = latitude and longitude
        logger.info(
            f"[webhook] media_count={len(media_urls)} media_type={media_content_type0} location=({latitude},{longitude})"
        )
        from_number = request.form.get("From", "")
        logger.info(f"[webhook] from={from_number}")
        message_sid = request.form.get("MessageSid", "")
        logger.info(f"[webhook] message_sid={message_sid}")

        resp = MessagingResponse()

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

        job = supabase_client.create_job(
            {
                "phone_number": from_number,
                "message_sid": message_sid,
                "job_type": job_type,
                "payload": payload,
            }
        )

        try:
            process_whatsapp_job.delay(str(job.get("id")))
        except Exception as e:
            logger.error(f"Failed to enqueue background job: {e}", exc_info=True)
            resp.message("❌ Sorry, I couldn't start processing your message. Please try again later.")

        return str(resp), 200

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        resp = MessagingResponse()
        resp.message("Sorry, an error occurred processing your request. Please try again.")
        return str(resp), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "False") == "True")
