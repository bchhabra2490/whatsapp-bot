"""
OpenAI client using the official `openai` package for embeddings + chat (with tools).
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class OpenAIClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        # Defaults can be overridden via env
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def create_embedding(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        print(f"[OpenAIClient] create_embedding: len={len(text)}")
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        emb = resp.data[0].embedding
        print("[OpenAIClient] create_embedding: success")
        return emb

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        msg = self._chat_raw(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out = (msg.get("content") or "").strip()
        print("[OpenAIClient] chat: got response")
        return out

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
