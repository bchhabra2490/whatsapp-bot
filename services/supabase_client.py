"""
Supabase client for database and storage operations
"""

import os
from supabase import create_client, Client
from typing import Optional, Dict, Any
import uuid
import requests
from urllib.parse import urlparse, unquote
import time


class SupabaseClient:
    """Handles all Supabase operations"""

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

        self.client: Client = create_client(url, key)
        self.storage_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "whatsapp")
        self.records_table = os.getenv("SUPABASE_RECORDS_TABLE", "wbot_records")
        self.messages_table = os.getenv("SUPABASE_MESSAGES_TABLE", "wbot_messages")
        self.jobs_table = os.getenv("SUPABASE_JOBS_TABLE", "wbot_jobs")

    def upload_file(self, file_content: bytes, file_name: str, content_type: str) -> str:
        """
        Upload file to Supabase Storage

        Args:
            file_content: File content as bytes
            file_name: Original file name
            content_type: MIME type of the file

        Returns:
            Public URL of the uploaded file
        """
        # Generate unique file path
        file_id = str(uuid.uuid4())
        file_ext = file_name.split(".")[-1] if "." in file_name else "jpg"
        storage_path = f"{file_id}.{file_ext}"

        # Upload to storage
        try:
            print(f"[SupabaseClient] upload_file: path={storage_path}, content_type={content_type}")
            self.client.storage.from_(self.storage_bucket).upload(
                path=storage_path, file=file_content, file_options={"content-type": content_type}
            )
        except Exception as e:
            raise Exception(f"Failed to upload file to Supabase Storage: {str(e)}")

        def _looks_like_invalid_jwt(resp_text: str) -> bool:
            t = (resp_text or "").lower()
            return "invalidjwt" in t and ("exp" in t or "timestamp" in t)

        def _validate_url(url: str) -> bool:
            """
            Best-effort fetch to ensure Supabase signed URL isn't immediately invalid/expired.
            Uses Range to avoid downloading full content.
            """
            if not url or not isinstance(url, str):
                return False
            headers = {"Range": "bytes=0-1"}
            try:
                r = requests.get(url, headers=headers, timeout=5)
                # Supabase storage returns 200 (or 206 for range) for valid links.
                if r.status_code in (200, 206):
                    return True
                # If it's an invalid/expired token, Supabase returns a 400/401 with body mentioning InvalidJWT.
                if _looks_like_invalid_jwt(r.text):
                    return False
                return r.status_code < 400
            except Exception:
                return False

        # Generate a time-limited signed URL for external access
        try:
            expires_in_seconds = int(
                os.getenv("SUPABASE_SIGNED_URL_EXPIRES_IN_SECONDS", str(60 * 60 * 24))
            )  # default 24 hours
            signed = self.client.storage.from_(self.storage_bucket).create_signed_url(
                storage_path, expires_in=expires_in_seconds
            )
            # supabase-py returns a dict with 'signedURL' or 'signed_url' depending on version
            url = signed.get("signed_url") or signed.get("signedURL") or signed

            # Some environments interpret `expires_in` in milliseconds.
            # If the returned URL fails immediately (InvalidJWT), retry once with ms-style.
            if not _validate_url(url):
                ms_expires_in = expires_in_seconds * 1000
                print(
                    f"[SupabaseClient] upload_file: signed url validation failed; retrying with ms expires_in={ms_expires_in}"
                )
                signed_retry = self.client.storage.from_(self.storage_bucket).create_signed_url(
                    storage_path, expires_in=ms_expires_in
                )
                url_retry = (
                    signed_retry.get("signed_url") or signed_retry.get("signedURL") or signed_retry
                    if isinstance(signed_retry, dict)
                    else signed_retry
                )
                print(f"[SupabaseClient] upload_file: signed_url_retry={url_retry}")
                return str(url_retry)

            print(f"[SupabaseClient] upload_file: signed_url={url}")
            return str(url)
        except Exception as e:
            raise Exception(f"Failed to create signed URL from Supabase Storage: {str(e)}")

    def save_record(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a record (media OCR text or user note) to Postgres.

        Args:
            record_data: Record payload for insertion

        Returns:
            Saved record
        """
        try:
            print(f"[SupabaseClient] save_record into {self.records_table}")
            result = self.client.table(self.records_table).insert(record_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise Exception(f"Failed to save record to database: {str(e)}")

    def save_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a WhatsApp message (incoming or outgoing) for conversation context.

        Args:
            message_data: { phone_number, direction, role, content, message_sid?, metadata? }
        """
        try:
            print(f"[SupabaseClient] save_message into {self.messages_table}")
            result = self.client.table(self.messages_table).insert(message_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise Exception(f"Failed to save message to database: {str(e)}")

    def get_records_by_phone(self, phone_number: str, limit: int = 10) -> list:
        """
        Get recent records for a phone number.

        Args:
            phone_number: WhatsApp phone number

        Returns:
            List of receipt records
        """
        print(f"[SupabaseClient] get_records_by_phone: phone={phone_number}, limit={limit}")
        result = (
            self.client.table(self.records_table)
            .select("*")
            .eq("phone_number", phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        data = result.data if result.data else []
        print(f"[SupabaseClient] get_records_by_phone: found={len(data)}")
        return data

    def get_record_count(self, phone_number: str) -> int:
        """
        Get count of records for a phone number.

        Args:
            phone_number: WhatsApp phone number

        Returns:
            Count of receipts
        """
        print(f"[SupabaseClient] get_record_count: phone={phone_number}")
        result = (
            self.client.table(self.records_table)
            .select("id", count="exact")
            .eq("phone_number", phone_number)
            .execute()
        )
        count = result.count if result.count else 0
        print(f"[SupabaseClient] get_record_count: count={count}")
        return count

    def match_records(self, phone_number: str, query_embedding: list, match_count: int = 5) -> list:
        """
        Semantic search via RPC `wbot_match_records`.
        """
        if not query_embedding:
            return []

        print(f"[SupabaseClient] match_records: phone={phone_number}, k={match_count}")
        result = self.client.rpc(
            "wbot_match_records",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "p_phone_number": phone_number,
            },
        ).execute()

        data = result.data if result.data else []
        print(f"[SupabaseClient] match_records: found={len(data)}")
        return data

    def get_messages_by_phone(self, phone_number: str, limit: int = 10) -> list:
        """
        Get recent conversation messages for a phone number (most recent first).
        """
        print(f"[SupabaseClient] get_messages_by_phone: phone={phone_number}, limit={limit}")
        result = (
            self.client.table(self.messages_table)
            .select("*")
            .eq("phone_number", phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        data = result.data if result.data else []
        print(f"[SupabaseClient] get_messages_by_phone: found={len(data)}")
        return data

    # Job helpers

    def create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a background job record and return it.
        """
        try:
            print(f"[SupabaseClient] create_job into {self.jobs_table}")
            result = self.client.table(self.jobs_table).insert(job_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise Exception(f"Failed to create job: {str(e)}")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get a job by ID.
        """
        try:
            print(f"[SupabaseClient] get_job: id={job_id}")
            result = self.client.table(self.jobs_table).select("*").eq("id", job_id).single().execute()
            return result.data or {}
        except Exception as e:
            raise Exception(f"Failed to get job: {str(e)}")

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update job fields (status, error, result, etc.).
        """
        try:
            print(f"[SupabaseClient] update_job: id={job_id}, updates={updates}")
            result = self.client.table(self.jobs_table).update(updates).eq("id", job_id).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise Exception(f"Failed to update job: {str(e)}")

    def resign_url(self, url: str, expires_in_seconds: Optional[int] = None) -> str:
        """
        Given a previously-signed Supabase Storage URL, drop the querystring and
        create a fresh signed URL for the same object.
        """
        if not url:
            return url
        if not isinstance(url, str):
            return str(url)

        # Per your request: split at '?' and ignore the existing token.
        base = url.split("?", 1)[0].strip()
        parsed = urlparse(base)
        parts = [p for p in parsed.path.split("/") if p]

        # Expected: /storage/v1/object/<sign|public>/<bucket>/<object_path...>
        try:
            object_idx = parts.index("object")
        except ValueError:
            return url

        if len(parts) < object_idx + 4:
            return url

        bucket = parts[object_idx + 2]
        object_path_parts = parts[object_idx + 3 :]
        object_path = "/".join(unquote(p) for p in object_path_parts).strip("/")
        if not object_path:
            return url

        if expires_in_seconds is None:
            expires_in_seconds = int(os.getenv("SUPABASE_SIGNED_URL_EXPIRES_IN_SECONDS", str(60 * 60 * 24)))

        try:
            signed = self.client.storage.from_(bucket).create_signed_url(
                object_path, expires_in=expires_in_seconds
            )
            new_url = signed.get("signed_url") or signed.get("signedURL") or signed
            return str(new_url)
        except Exception:
            # If resigning fails for any reason, return the original URL.
            return url
