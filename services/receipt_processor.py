"""
Record processing pipeline:
- Media: download -> store -> OCR -> embed -> save
- Note: embed -> save
- Question: embed -> match -> answer
"""

import requests
from typing import Dict, Any, List

from services.supabase_client import SupabaseClient
from services.mistral_ocr import MistralOCR
from services.openai_client import OpenAIClient


class RecordProcessor:
    """Handles the complete record workflow (media + text notes + Q&A)."""

    def __init__(self, supabase_client: SupabaseClient, mistral_ocr: MistralOCR, openai: OpenAIClient):
        self.supabase = supabase_client
        self.ocr = mistral_ocr
        self.openai = openai

    def process_media_urls(self, media_urls: List[str], phone_number: str, message_sid: str) -> Dict[str, Any]:
        """
        Process one or more media URLs: download, upload, OCR each, embed combined text, save to DB.
        """
        try:
            if not media_urls:
                return {"success": False, "error": "No media URLs provided"}

            print(f"[RecordProcessor] process_media_urls: {len(media_urls)} URL(s) for {phone_number}")

            storage_urls: List[str] = []
            ocr_texts: List[str] = []

            for media_url in media_urls:
                print(f"[RecordProcessor] Downloading media: {media_url}")
                media_response = requests.get(media_url, timeout=30)
                media_response.raise_for_status()
                file_content = media_response.content

                content_type = media_response.headers.get("Content-Type", "image/jpeg")
                file_name = media_url.split("/")[-1] or "upload"

                print(f"[RecordProcessor] Uploading to storage: {file_name} ({content_type})")
                storage_url = self.supabase.upload_file(
                    file_content=file_content, file_name=file_name, content_type=content_type
                )
                storage_urls.append(storage_url)

                print(f"[RecordProcessor] Running OCR via Mistral on: {storage_url}")
                text = self.ocr.extract_text(storage_url, content_type=content_type)
                if text:
                    ocr_texts.append(text)

            combined_text = "\n\n---\n\n".join(ocr_texts).strip()
            print(f"[RecordProcessor] Combined OCR text length: {len(combined_text)}")
            embedding = self.openai.create_embedding(combined_text) if combined_text else []
            print(f"[RecordProcessor] Embedding generated: dim={len(embedding) if embedding else 0}")

            record = {
                "phone_number": phone_number,
                "message_sid": message_sid,
                "record_type": "media",
                "storage_urls": storage_urls,
                "ocr_text": combined_text,
                "embedding": embedding if embedding else None,
                "metadata": {"source": "whatsapp", "media_count": len(storage_urls)},
            }

            print("[RecordProcessor] Saving media record to Supabase")
            saved = self.supabase.save_record(record)
            return {"success": True, "record_id": saved.get("id"), "media_count": len(storage_urls)}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Failed to download media: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Processing error: {str(e)}"}

    def save_note(self, phone_number: str, message_sid: str, user_text: str) -> Dict[str, Any]:
        try:
            print(f"[RecordProcessor] save_note for {phone_number}, text length={len(user_text)}")
            embedding = self.openai.create_embedding(user_text)
            print(f"[RecordProcessor] Note embedding dim={len(embedding) if embedding else 0}")
            record = {
                "phone_number": phone_number,
                "message_sid": message_sid,
                "record_type": "note",
                "user_text": user_text,
                "embedding": embedding if embedding else None,
                "metadata": {"source": "whatsapp"},
            }
            print("[RecordProcessor] Saving note record to Supabase")
            saved = self.supabase.save_record(record)
            return {"success": True, "record_id": saved.get("id")}
        except Exception as e:
            return {"success": False, "error": f"Failed to save note: {str(e)}"}

    def detect_intent(self, message: str, history: List[Dict[str, Any]] | None = None) -> str:
        """
        Returns: 'question' | 'save_record'
        """
        # Build short conversation context from recent messages if provided
        history_text = ""
        if history:
            # history is most-recent-first; reverse to chronological
            lines: List[str] = []
            for m in reversed(history):
                role = m.get("role", "user")
                direction = m.get("direction", "in")
                txt = (m.get("content") or "")[:120].replace("\n", " ")
                lines.append(f"{role}({direction}): {txt}")
            history_text = "\n".join(lines)

        system = (
            "You classify user WhatsApp messages for a personal capture bot.\n"
            "You will be given the recent conversation and the latest user message.\n"
            "Return exactly one token: question OR save_record.\n"
            "- If the user asks anything, requests info, or wants to find something: question.\n"
            "- If the user is stating something to remember, logging info, or saving a note: save_record.\n"
        )
        user = f"Recent conversation:\n{history_text or '(none)'}\n\nLatest user message:\n{message}"

        print(f"[RecordProcessor] detect_intent: message='{message[:80]}'")
        out = self.openai.chat(system=system, user=user, temperature=0.0, max_tokens=5).lower()
        print(f"[RecordProcessor] detect_intent raw output: '{out}'")
        if "save_record" in out:
            return "save_record"
        return "question"

    def answer_question(self, phone_number: str, question: str, history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        try:
            print(f"[RecordProcessor] answer_question for {phone_number}: '{question[:120]}'")
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_records",
                        "description": "Semantic search over the user's saved records (OCR text + notes).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_recent_records",
                        "description": "Fetch the user's most recent saved records.",
                        "parameters": {
                            "type": "object",
                            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
                        },
                    },
                },
            ]

            def tool_executor(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                print(f"[RecordProcessor] tool_executor called: {name} args={args}")
                if name == "search_records":
                    q = (args.get("query") or "").strip()
                    top_k = int(args.get("top_k") or 5)
                    print(f"[RecordProcessor] search_records: query='{q[:80]}', top_k={top_k}")
                    emb = self.openai.create_embedding(q)
                    matches = self.supabase.match_records(
                        phone_number=phone_number, query_embedding=emb, match_count=top_k
                    )
                    # Keep payload small + safe
                    out = []
                    for m in matches:
                        out.append(
                            {
                                "id": m.get("id"),
                                "record_type": m.get("record_type"),
                                "created_at": m.get("created_at"),
                                "similarity": m.get("similarity"),
                                "text": (m.get("ocr_text") or m.get("user_text") or "")[:3000],
                            }
                        )
                    print(f"[RecordProcessor] search_records: {len(out)} match(es)")
                    return {"matches": out}

                if name == "get_recent_records":
                    limit = int(args.get("limit") or 5)
                    print(f"[RecordProcessor] get_recent_records: limit={limit}")
                    recs = self.supabase.get_records_by_phone(phone_number=phone_number, limit=limit)
                    out = []
                    for r in recs:
                        out.append(
                            {
                                "id": r.get("id"),
                                "record_type": r.get("record_type"),
                                "created_at": r.get("created_at"),
                                "text": ((r.get("ocr_text") or r.get("user_text") or "")[:1500]),
                            }
                        )
                    print(f"[RecordProcessor] get_recent_records: {len(out)} record(s)")
                    return {"records": out}

                return {"error": f"unknown_tool:{name}"}

            # Build short conversation context from recent messages if provided
            history_text = ""
            if history:
                lines: List[str] = []
                for m in reversed(history):
                    role = m.get("role", "user")
                    direction = m.get("direction", "in")
                    txt = (m.get("content") or "")[:200].replace("\n", " ")
                    lines.append(f"{role}({direction}): {txt}")
                history_text = "\n".join(lines)

            system = (
                "You are a WhatsApp capture-bot assistant.\n"
                "You have tools to search the user's saved records.\n"
                "You are also given recent conversation messages as context.\n"
                "Use tools and conversation context when needed to answer.\n"
                "Answer concisely.\n"
                "If the answer is not in the records, say you don't know and ask what to save.\n"
                "Do not mention embeddings, vectors, Supabase, or internal tooling."
            )

            answer = self.openai.agent_chat(
                system=system,
                user=f"Recent conversation:\n{history_text or '(none)'}\n\nUser question:\n{question}",
                tools=tools,
                tool_executor=tool_executor,
                max_steps=4,
                temperature=0.2,
                max_tokens=500,
            )

            print("[RecordProcessor] answer_question complete")
            return {"success": True, "answer": answer}
        except Exception as e:
            return {"success": False, "error": f"Failed to answer: {str(e)}"}
