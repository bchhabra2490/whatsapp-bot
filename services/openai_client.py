"""
OpenAI client using the official `openai` package for embeddings, chat (with tools), and Whisper transcription.
"""

import io
import base64
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


class OpenAIClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        # Defaults can be overridden via env
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.whisper_model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.ogg") -> str:
        """
        Transcribe audio using OpenAI Whisper. Returns transcribed text.
        """
        if not audio_bytes:
            return ""
        print(f"[OpenAIClient] transcribe_audio: size={len(audio_bytes)}, filename={filename}")
        file_like = io.BytesIO(audio_bytes)
        file_like.name = filename
        resp = self.client.audio.transcriptions.create(
            model=self.whisper_model,
            file=file_like,
        )
        text = (resp.text or "").strip()
        print(f"[OpenAIClient] transcribe_audio: got {len(text)} chars")
        return text

    def create_embedding(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        print(f"[OpenAIClient] create_embedding: len={len(text)}")
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        emb = resp.data[0].embedding
        print("[OpenAIClient] create_embedding: success")
        return emb

    def generate_image(self, prompt: str, size: str = "1024x1024") -> bytes:
        """
        Generate a PNG image from text using OpenAI Images.

        Returns:
            PNG bytes.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required to generate image")

        # Keep model configurable in case you want to switch later.
        image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        resp = self.client.images.generate(
            model=image_model,
            prompt=prompt,
            size=size,
            response_format="b64_json",
        )
        b64 = (resp.data[0].b64_json or "").strip()
        if not b64:
            raise RuntimeError("OpenAI image generation returned empty image data")
        return base64.b64decode(b64)

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        msg = self._chat_raw(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out = (msg.get("content") or "").strip()
        print("[OpenAIClient] chat: got response")
        return out

    def chat_stream(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 300) -> Iterable[str]:
        """
        Stream chat completion tokens (text deltas).
        """
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for event in resp:
            try:
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
            except Exception:
                text = None
            if text:
                yield text

    def _chat_raw(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 600,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
        }
        print(f"[OpenAIClient] _chat_raw: kwargs: {kwargs}")
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        resp = self.client.chat.completions.create(**kwargs)
        print(f"[OpenAIClient] _chat_raw: response: {resp}")
        choice = resp.choices[0]
        print("[OpenAIClient] _chat_raw: completion received")
        # `choice.message` is an object; we convert to a dict-like for downstream code.
        msg: Dict[str, Any] = {
            "role": choice.message.role,
            "content": choice.message.content,
        }
        if choice.message.tool_calls:
            # Convert tool_calls to plain dicts
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            msg["tool_calls"] = tool_calls
        return msg

    def agent_chat(
        self,
        system: str,
        user: str,
        tools: List[Dict[str, Any]],
        tool_executor,
        max_steps: int = 4,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """
        Minimal tool-using agent loop for chat/completions.
        `tool_executor(name, arguments_dict) -> dict` must be provided by caller.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for _ in range(max_steps):
            print("[OpenAIClient] agent_chat: requesting step with tools")
            msg = self._chat_raw(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",
            )

            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                print(f"[OpenAIClient] agent_chat: model requested {len(tool_calls)} tool call(s)")
            if not tool_calls:
                out = (msg.get("content") or "").strip()
                print("[OpenAIClient] agent_chat: no tool calls, returning answer")
                return out

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                fn = (tc.get("function") or {}).get("name")
                raw_args = (tc.get("function") or {}).get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}

                print(f"[OpenAIClient] agent_chat: executing tool '{fn}'")
                result = tool_executor(fn, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # If we hit step limit, force a final response without tools.
        msg = self._chat_raw(
            messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice="none"
        )
        return (msg.get("content") or "").strip()
