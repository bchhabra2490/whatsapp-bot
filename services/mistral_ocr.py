"""
Mistral OCR service (generic) for extracting text from images/PDFs.
"""

import os
import requests
from typing import Dict, Any, Optional
import base64
from io import BytesIO


class MistralOCR:
    """Handles OCR operations using Mistral API"""

    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("MISTRAL_MODEL", "pixtral-12b-2409")
        self.api_url = "https://api.mistral.ai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY must be set in environment variables")

    def extract_text(self, file_url: str, content_type: Optional[str] = None) -> str:
        """
        Extract raw text from image/PDF using Mistral vision model.

        Args:
            file_url: URL of the image or PDF
            content_type: MIME type of the file (optional)

        Returns:
            Extracted text
        """
        # Download file
        response = requests.get(file_url)
        response.raise_for_status()
        file_data = response.content

        # Handle PDF files - convert first page to image
        if content_type == "application/pdf" or file_url.lower().endswith(".pdf"):
            try:
                # Try to use pdf2image if available
                try:
                    from pdf2image import convert_from_bytes

                    images = convert_from_bytes(file_data, first_page=1, last_page=1)
                    if images:
                        img_byte_arr = BytesIO()
                        images[0].save(img_byte_arr, format="JPEG")
                        file_data = img_byte_arr.getvalue()
                        content_type = "image/jpeg"
                except ImportError:
                    raise Exception("PDF support requires pdf2image library. Install with: pip install pdf2image")
            except Exception as e:
                raise Exception(f"Failed to process PDF: {str(e)}")

        # Convert to base64
        image_base64 = base64.b64encode(file_data).decode("utf-8")

        # Determine MIME type for API
        mime_type = content_type or "image/jpeg"
        if mime_type.startswith("image/"):
            data_url_prefix = f"data:{mime_type};base64,"
        else:
            data_url_prefix = "data:image/jpeg;base64,"

        prompt = (
            "Extract ALL visible text from this image. Preserve line breaks where helpful. "
            "Return only the extracted text, with no commentary."
        )

        # Call Mistral API
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"{data_url_prefix}{image_base64}"}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return (content or "").strip()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Mistral API error: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {str(e)}")
