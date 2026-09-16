"""Twilio request signature validation for HTTP webhooks and Media Stream WebSockets."""

import os
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import parse_qsl, urlparse

from twilio.request_validator import RequestValidator


def validation_enabled() -> bool:
    raw = (os.getenv("TWILIO_VALIDATE_SIGNATURE") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _auth_token() -> str:
    return (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()


def _configured_public_bases() -> list[str]:
    bases = []
    for key in ("VOICE_BASE_URL", "APP_BASE_URL"):
        value = (os.getenv(key) or "").strip().rstrip("/")
        if value:
            bases.append(value)
    return bases


def _with_slash_variants(url: str) -> list[str]:
    parsed = urlparse(url)
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    base = f"{parsed.scheme}://{parsed.netloc}"
    trimmed = path.rstrip("/") or "/"
    slashed = trimmed if trimmed.endswith("/") else f"{trimmed}/"
    return [
        f"{base}{trimmed}{query}",
        f"{base}{slashed}{query}",
    ]


def _scheme_variants(url: str) -> list[str]:
    variants = [url]
    if url.startswith("https://"):
        variants.append("wss://" + url[len("https://") :])
        variants.append("http://" + url[len("https://") :])
        variants.append("ws://" + url[len("https://") :])
    elif url.startswith("http://"):
        variants.append("ws://" + url[len("http://") :])
        variants.append("https://" + url[len("http://") :])
        variants.append("wss://" + url[len("http://") :])
    elif url.startswith("wss://"):
        variants.append("https://" + url[len("wss://") :])
        variants.append("ws://" + url[len("wss://") :])
    elif url.startswith("ws://"):
        variants.append("http://" + url[len("ws://") :])
        variants.append("wss://" + url[len("ws://") :])
    return variants


def candidate_urls(
    *,
    forwarded_proto: str,
    forwarded_host: str,
    path: str,
    query: str = "",
) -> list[str]:
    proto = (forwarded_proto or "https").split(",")[0].strip() or "https"
    host = (forwarded_host or "").split(",")[0].strip()
    path = path or "/"
    if not path.startswith("/"):
        path = "/" + path
    qs = f"?{query}" if query else ""
    urls: list[str] = []
    if host:
        urls.extend(_scheme_variants(f"{proto}://{host}{path}{qs}"))
    for base in _configured_public_bases():
        urls.extend(_scheme_variants(f"{base}{path}{qs}"))
    expanded: list[str] = []
    for url in urls:
        expanded.extend(_with_slash_variants(url))
    deduped = []
    seen = set()
    for url in expanded:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _form_params(params: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not params:
        return {}
    out: Dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        out[str(key)] = str(value)
    return out


def request_is_valid(
    signature: Optional[str],
    urls: Iterable[str],
    params: Optional[Mapping[str, Any]] = None,
) -> bool:
    if not validation_enabled():
        return True
    token = _auth_token()
    if not token or not signature:
        return False
    validator = RequestValidator(token)
    form = _form_params(params)
    for url in urls:
        try:
            if validator.validate(url, form, signature):
                return True
        except Exception:
            continue
    return False


def query_params_from_url(url: str) -> Dict[str, str]:
    parsed = urlparse(url)
    return dict(parse_qsl(parsed.query, keep_blank_values=True))
