"""
Celery tasks for background processing of WhatsApp messages.

IMPORTANT: Celery runs in a separate process, so we explicitly load `.env`
here to make sure SUPABASE_*, TWILIO_*, OPENAI_* etc. are available.
"""

import os
from typing import Any, Dict

from celery import Celery
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

from services.supabase_client import SupabaseClient
from services.mistral_ocr import MistralOCR
from services.openai_client import OpenAIClient
from services.receipt_processor import RecordProcessor
from services.whatsapp_handler import WhatsAppHandler


def make_celery() -> Celery:
    # Ensure .env is loaded when the worker process starts
    # Assumes worker is started from project root, e.g.:
    #   cd whatsapp-bot && celery -A services.tasks.celery_app worker --loglevel=info
    load_dotenv()
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend_url = os.getenv("CELERY_RESULT_BACKEND", broker_url)
    app = Celery("whatsapp_bot", broker=broker_url, backend=backend_url)
    return app


celery_app = make_celery()


def _build_services():
    supabase = SupabaseClient()
    mistral = MistralOCR()
    openai_client = OpenAIClient()
    processor = RecordProcessor(supabase, mistral, openai_client)
    handler = WhatsAppHandler(processor)

    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = (os.getenv("TWILIO_WHATSAPP_NUMBER") or "").strip()
    if not (twilio_account_sid and twilio_auth_token and twilio_from):
        raise ValueError("Twilio env vars (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER) must be set")
    # WhatsApp channel requires "whatsapp:+1234567890" format (Twilio sandbox e.g. whatsapp:+14155238886)
    if not twilio_from.lower().startswith("whatsapp:"):
        raw = twilio_from.lstrip()
        twilio_from = f"whatsapp:{raw}" if raw.startswith("+") else f"whatsapp:+{raw}"

    twilio_client = TwilioClient(twilio_account_sid, twilio_auth_token)

    return supabase, handler, twilio_client, twilio_from


@celery_app.task(name="process_whatsapp_job")
def process_whatsapp_job(job_id: str) -> Dict[str, Any]:
    """
    Background job: process a WhatsApp message (media or text) and send a reply.
    """
    supabase, handler, twilio_client, twilio_from = _build_services()

    # Load job
    job = supabase.get_job(job_id)
    if not job:
        return {"success": False, "error": "job_not_found", "job_id": job_id}

    phone_number = job.get("phone_number")
    message_sid = job.get("message_sid") or ""
    job_type = job.get("job_type")
    payload = job.get("payload") or {}

    print(f"[tasks.py] Processing job: {job_id} for {phone_number} with type {job_type}")
    try:
        supabase.update_job(job_id, {"status": "processing"})

        if job_type == "media":
            media_urls = payload.get("media_urls") or []
            response_text = handler.handle_media(
                media_urls=media_urls, from_number=phone_number, message_sid=message_sid
            )
        elif job_type == "text":
            text = payload.get("text") or ""
            response_text = handler.handle_text(message=text, from_number=phone_number, message_sid=message_sid)
        else:
            raise ValueError(f"Unsupported job_type: {job_type}")

        # Update job result
        supabase.update_job(job_id, {"status": "completed", "result": {"response": response_text}})

        # Send WhatsApp reply (From and To must both be whatsapp: channel)
        to_number = (phone_number or "").strip()
        if not to_number.lower().startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}" if to_number.startswith("+") else f"whatsapp:+{to_number}"

        print(f"[tasks.py] Sending WhatsApp reply to {to_number} from {twilio_from}")
        twilio_client.messages.create(
            from_=twilio_from,
            to=to_number,
            body=response_text,
        )

        ## Save the response to the database
        supabase.save_message(
            {
                "phone_number": phone_number,
                "direction": "out",
                "role": "assistant",
                "message_sid": message_sid,
                "content": response_text,
            }
        )

        return {"success": True, "job_id": job_id}
    except Exception as e:
        supabase.update_job(job_id, {"status": "failed", "error": str(e)})
        # Try to notify user about failure
        try:
            to_number = (phone_number or "").strip()
            if not to_number.lower().startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}" if to_number.startswith("+") else f"whatsapp:+{to_number}"
            twilio_client.messages.create(
                from_=twilio_from,
                to=to_number,
                body="Sorry, your request could not be processed. Please try again later.",
            )
        except Exception:
            # swallow any Twilio error here; main failure is already recorded
            pass
        return {"success": False, "job_id": job_id, "error": str(e)}
