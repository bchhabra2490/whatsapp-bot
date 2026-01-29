"""
WhatsApp Receipt Capture Bot
Main Flask application entry point
"""

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from services.supabase_client import SupabaseClient
from services.tasks import process_whatsapp_job

load_dotenv()

app = Flask(__name__)

# Initialize services
supabase_client = SupabaseClient()


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
        # Twilio sends media as MediaUrl0, MediaUrl1, etc.
        media_urls = [
            url
            for url in [
                request.form.get("MediaUrl0"),
                request.form.get("MediaUrl1"),
                request.form.get("MediaUrl2"),
            ]
            if url
        ]
        print(f"Media URLs: {media_urls}")
        from_number = request.form.get("From", "")
        print(f"From number: {from_number}")
        message_sid = request.form.get("MessageSid", "")
        print(f"Message SID: {message_sid}")

        # Create Twilio response
        resp = MessagingResponse()

        # Build a background job and enqueue it
        job_type = "media" if media_urls else "text"
        payload: dict = {}
        if media_urls:
            payload = {"media_urls": media_urls, "incoming_text": incoming_message}
        elif incoming_message:
            payload = {"text": incoming_message}

        if not payload:
            resp.message("Please send a receipt image or PDF, or ask a question about your receipts.")
            return str(resp), 200

        # Persist the incoming message immediately
        try:
            supabase_client.save_message(
                {
                    "phone_number": from_number,
                    "direction": "in",
                    "role": "user",
                    "message_sid": message_sid,
                    "content": incoming_message or (f"[media] {', '.join(media_urls)}" if media_urls else ""),
                    "metadata": {"media_urls": media_urls} if media_urls else {},
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "False") == "True")
