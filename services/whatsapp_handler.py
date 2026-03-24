"""
WhatsApp message handler
"""

from typing import List

from services.receipt_processor import RecordProcessor
from services.call_service import CallService


class WhatsAppHandler:
    """Handles incoming WhatsApp messages"""

    def __init__(self, record_processor: RecordProcessor, call_service: CallService | None = None):
        self.processor = record_processor
        self.call_service = call_service

    def handle_media(self, media_urls: List[str], from_number: str, message_sid: str) -> str:
        """
        Handle media messages (images/PDFs)

        Args:
            media_urls: List of media URLs from Twilio
            from_number: Sender's WhatsApp number
            message_sid: Twilio message SID

        Returns:
            Response message to send back
        """
        print(f"[WhatsAppHandler] handle_media from={from_number} sid={message_sid} urls={media_urls}")
        if not media_urls:
            return "Please send one or more images/PDFs."

        result = self.processor.process_media_urls(
            media_urls=media_urls, phone_number=from_number, message_sid=message_sid
        )

        print(f"[WhatsAppHandler] handle_media result: {result}")
        if result.get("success"):
            return (
                f"✅ Saved {result.get('media_count', 1)} file(s) as a record.\nRecord ID: {result.get('record_id')}"
            )

        error = result.get("error", "Unknown error")
        return f"❌ Failed to process media: {error}"

    def handle_text(self, message: str, from_number: str, message_sid: str = "") -> str:
        """
        Handle text messages (queries)

        Args:
            message: Text message content
            from_number: Sender's WhatsApp number

        Returns:
            Response message to send back
        """
        print(f"[WhatsAppHandler] handle_text from={from_number} sid={message_sid} message='{message[:120]}'")
        # Fetch recent conversation history (without current message) for better intent + answer
        history = self.processor.supabase.get_messages_by_phone(phone_number=from_number, limit=10)
        intent = self.processor.detect_intent(message, history=history)
        print(f"[WhatsAppHandler] detected intent={intent}")

        if intent == "save_record":
            # For text-only notes, we don't have Twilio MessageSid in text handler currently.
            # We'll save with empty message_sid; app.py can be updated to pass it if desired.
            saved = self.processor.save_note(phone_number=from_number, message_sid=message_sid, user_text=message)
            print(f"[WhatsAppHandler] save_note result: {saved}")
            if saved.get("success"):
                return f"✅ Saved your note.\nRecord ID: {saved.get('record_id')}"
            return f"❌ Failed to save your note: {saved.get('error')}"

        if intent == "call":
            if not self.call_service:
                return "❌ Calling is not configured yet."

            extracted = self.processor.extract_call_request(message=message)
            target_number = extracted.get("target_number") or ""
            question_to_ask = extracted.get("question_to_ask") or ""
            if not target_number or not question_to_ask:
                return (
                    "❌ I couldn't parse the call details.\n"
                    "Try: call +1234567890 and ask if they can share the shipment ETA."
                )

            started = self.call_service.start_outbound_call(
                requested_by=from_number,
                to_number=target_number,
                prompt_question=question_to_ask,
            )
            print(f"[WhatsAppHandler] start_outbound_call result: {started}")
            if started.get("success"):
                return f"📞 Calling {started.get('to_number')} now.\n" f'I\'ll ask: "{question_to_ask}"'
            return f"❌ Failed to start call: {started.get('error')}"

        answered = self.processor.answer_question(phone_number=from_number, question=message, history=history)
        print(f"[WhatsAppHandler] answer_question result: {answered}")
        if answered.get("success"):
            return answered.get("answer", "I couldn't generate an answer.")
        return f"❌ Failed to answer: {answered.get('error')}"
