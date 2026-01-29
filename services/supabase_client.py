"""
Supabase client for database and storage operations
"""

import os
from supabase import create_client, Client
from typing import Optional, Dict, Any
import uuid


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

        # Generate a time-limited signed URL for external OCR access
        try:
            signed = self.client.storage.from_(self.storage_bucket).create_signed_url(
                storage_path, expires_in=60 * 60  # 1 hour
            )
            # supabase-py returns a dict with 'signedURL' or 'signed_url' depending on version
            url = signed.get("signed_url") or signed.get("signedURL") or signed
            print(f"[SupabaseClient] upload_file: signed_url={url}")
            return url
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
