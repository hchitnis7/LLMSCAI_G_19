"""
Client for the model service (fine-tuned Qwen 3 models, exposed over HTTPS through ngrok).

Service interface:
    POST {B_API_URL}/answer   request:  {"text": str, "image_b64": str | None, "state": ...}
                              response: {"answer": str, "state": ...}
    GET  {B_API_URL}/health

Configuration (read from .env locally and from Streamlit Secrets when deployed):
    B_API_URL        base address of the service, without a trailing path
    B_API_KEY        optional; sent as "Authorization: Bearer <key>" when set
    B_TIMEOUT        maximum time in seconds to wait for an answer (default 120)
    B_INITIAL_STATE  JSON value sent as "state" with the first message (default: null)
"""
from __future__ import annotations

import base64
import json
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BUnavailable(Exception):
    """The service could not be reached, did not respond in time, or returned a server error."""


class BBadResponse(Exception):
    """The service rejected the request or returned a response in an unexpected format."""


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def config() -> dict:
    return {
        "url": _setting("B_API_URL").rstrip("/"),
        "key": _setting("B_API_KEY"),
        "timeout": float(_setting("B_TIMEOUT", "120")),
        "initial_state": _setting("B_INITIAL_STATE", "null"),
    }


def initial_state():
    """Return the state value to send with the first message of a conversation."""
    raw = config()["initial_state"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _headers() -> dict:
    h = {"Content-Type": "application/json",
         "ngrok-skip-browser-warning": "true"}  # prevents ngrok's interstitial warning page
    key = config()["key"]
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def is_configured() -> bool:
    return bool(config()["url"])


def health() -> bool:
    """Return True if the service responds to /health. Does not raise exceptions."""
    import requests
    cfg = config()
    if not cfg["url"]:
        return False
    try:
        r = requests.get(f"{cfg['url']}/health", headers=_headers(), timeout=10)
        return r.ok
    except requests.RequestException:
        return False


def ask(text: str, state=None, image_bytes: bytes | None = None) -> dict:
    """Send one message to /answer and return {"answer": str, "state": ..., "raw": dict, "ms": int}."""
    import requests
    cfg = config()
    if not cfg["url"]:
        raise BUnavailable("B_API_URL is not set")

    payload = {
        "text": text,
        "image_b64": base64.b64encode(image_bytes).decode() if image_bytes else None,
        "state": state,
    }
    t = time.time()
    try:
        resp = requests.post(f"{cfg['url']}/answer", json=payload, headers=_headers(),
                             timeout=cfg["timeout"])
    except requests.Timeout as e:
        raise BUnavailable(f"timed out after {cfg['timeout']:.0f}s") from e
    except requests.RequestException as e:
        raise BUnavailable(f"connection failed: {e}") from e

    if resp.status_code >= 500 or resp.status_code in (502, 503, 504):
        raise BUnavailable(f"HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise BBadResponse(f"HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except ValueError as e:
        # An HTML page indicates that ngrok is offline or the service is not running
        raise BUnavailable(f"response is not JSON (service or tunnel may be offline): {resp.text[:200]}") from e

    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise BBadResponse(f"expected {{'answer': str, 'state': ...}}, got: {str(data)[:300]}")

    # "raw" keeps every field the service returned (severity, drugs, citations, safety_flag, gap, ...)
    return {"answer": data["answer"], "state": data.get("state"), "raw": data,
            "ms": int((time.time() - t) * 1000)}