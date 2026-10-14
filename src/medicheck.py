"""
Orchestrator: processes each user message and returns the result shown in the interface.

Public functions:
    chat(message, lang, state, image_bytes, mime)  main entry point used by the chat interface
    check(names, question, lang)                   structured check on a list of medicine names
    read_image(image_bytes, mime)                  extracts medicine names from a photo

The answer source is selected with the BACKEND setting (.env or Streamlit Secrets):
    stub     placeholder answers (default; no network access required)
    b_api    model service via POST {B_API_URL}/answer (see src/b_client.py)
    modules  optional direct import of Python modules (src/normalizer.py, rag.py, llm.py)
The earlier USE_STUB setting is still accepted when BACKEND is not set.

If the model service cannot be reached, the user receives a fixed notice instead of an answer.

Safety measures applied for every backend:
    before the model  crisis, dosage and instruction-override messages receive fixed replies
    after the model   fixed disclaimer, removal of dosage statements, pharmacist notice,
                      and a pregnancy notice when pregnancy is mentioned
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from . import safety
from .contracts import (DISCLAIMERS, LANGUAGE_INSTRUCTION, LANGUAGES, MODEL_LABELS, PHARMACIST_LINE,
                        PREGNANCY_LINE, CheckResult, Extraction, validate_result)


def _flag(name: str, default: str) -> str:
    return os.getenv(name, default)


# ---------------- Backend selection ----------------
def backend_name() -> str:
    b = _flag("BACKEND", "").strip().lower()
    if b in ("stub", "b_api", "modules"):
        return b
    return "stub" if _flag("USE_STUB", "true").lower() in ("1", "true", "yes") else "modules"


def use_stub() -> bool:
    return backend_name() == "stub"


def _backend():
    """Return the backend object providing normalize, retrieve, extract_drugs and answer."""
    if backend_name() == "modules":
        return _RealBackend
    from . import stub_backend
    return stub_backend


class _RealBackend:
    """Module-based backend; each module is imported only when first used."""

    @staticmethod
    def normalize(*a, **kw):
        return importlib.import_module("src.normalizer").normalize(*a, **kw)

    @staticmethod
    def retrieve(*a, **kw):
        return importlib.import_module("src.rag").retrieve(*a, **kw)

    @staticmethod
    def extract_drugs(*a, **kw):
        return importlib.import_module("src.vision").extract_drugs(*a, **kw)

    @staticmethod
    def answer(*a, **kw):
        return importlib.import_module("src.llm").answer(*a, **kw)


def model_tag() -> str:
    return {"stub": "stub", "b_api": "b-api"}.get(backend_name(), os.getenv("TEXT_MODEL", "unknown"))


# ---------------- Answer cache ----------------
def _cache_dir() -> Path:
    p = Path(_flag("CACHE_DIR", "./app_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(drugs: list[str], question: str, lang: str) -> str:
    payload = json.dumps([sorted(d.lower() for d in drugs), question.strip().lower(), lang, model_tag()])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def chat_cache_key(message: str, lang: str) -> str:
    payload = json.dumps(["chat", " ".join(message.lower().split()), lang, model_tag()])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_get(key: str) -> dict | None:
    path = _cache_dir() / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cache_put(key: str, result: dict) -> None:
    (_cache_dir() / f"{key}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                              encoding="utf-8")


# ---------------- Request log (traceability; contains no user text or personal data) ----------------
def _log(event: dict) -> None:
    log_dir = Path(_flag("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with open(log_dir / "requests.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------------- Helper functions ----------------
def _canned(kind: str, lang: str, drugs: list[str] | None = None) -> CheckResult:
    return {
        "drugs": drugs or [], "atc_codes": [], "severity": "INSUFFICIENT_DATA",
        "explanation": safety.MESSAGES[kind][lang], "citations": [], "confidence": 0.0,
        "disclaimer": DISCLAIMERS[lang], "language": lang, "notes": [],
        "from_cache": False, "db_last_updated": "", "model": "guardrail", "safety_flag": kind,
    }


def clean_name(name: str) -> str:
    """Remove strength information from a name, e.g. "Alvedon 500 mg" becomes "Alvedon"."""
    name = re.sub(r"\d+([.,]\d+)?\s*(mg|g|ml|mikrogram|µg|mcg|%)\b", " ", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip()


def parse_names(text: str) -> list[str]:
    """Split a list of names, e.g. "Waran + Ipren" or "Waran och Ipren" becomes ["Waran", "Ipren"]."""
    parts = re.split(r"\s*(?:\+|,|;|/|\band\b|\boch\b|\bو\b|\bwith\b|\bmed\b)\s*", text, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def find_names_in_text(text: str, vocabulary) -> list[str]:
    """Return the known medicine names found in free text, in order of appearance, without duplicates."""
    found = []
    for name in vocabulary:
        m = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.I)
        if m:
            found.append((m.start(), name))
    seen, out = set(), []
    for _, name in sorted(found):
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def _finalize(result: dict, message: str, lang: str) -> dict:
    """Apply the post-model safety measures; safety-critical texts are never generated by the model."""
    result["disclaimer"] = DISCLAIMERS[lang]
    if result.get("safety_flag") in ("crisis", "dose", "injection", "backend_down"):
        result["notes"] = []
        return result
    if result.get("model") not in ("stub",) and not result.get("grounded") and safety.contains_dose_advice(result.get("explanation", "")):
        result["explanation"] = safety.MESSAGES["dose_in_answer"][lang]
        result["safety_flag"] = "dose_in_answer"
    notes = [PHARMACIST_LINE[lang]]
    if safety.mentions_pregnancy(message):
        notes.append(PREGNANCY_LINE[lang])
    result["notes"] = notes
    return result


# ---------------- Public functions ----------------
def read_image(image_bytes: bytes, mime: str = "image/jpeg") -> Extraction:
    t = time.time()
    result = _backend().extract_drugs(image_bytes, mime)
    _log({"event": "ocr", "n_names": len(result.get("names", [])),
          "confidence": result.get("confidence"), "ms": int((time.time() - t) * 1000)})
    return result


def check(names: list[str], question: str = "", lang: str = "sv") -> CheckResult:
    """Check a list of medicine names using the placeholder or module-based backend."""
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}")
    t = time.time()

    # 1. Safety rules. Names are checked only for crisis and instruction-override patterns,
    #    so that strength information such as "500 mg" is not treated as a dosage question.
    flag = safety.classify(question)
    if not flag:
        name_flag = safety.classify(" ".join(names))
        flag = name_flag if name_flag in ("crisis", "injection") else None
    if flag:
        _log({"event": "guardrail", "flag": flag, "lang": lang})
        return _canned(flag, lang, names)

    backend = _backend()

    # 2. Normalise names
    names = [clean_name(n) for n in names if clean_name(n)]
    normalized = backend.normalize(names)
    drugs = [n["generic"] for n in normalized if n["generic"]]
    unknown = [n["input"] for n in normalized if not n["generic"]]

    # 3. Cache lookup
    key = cache_key(drugs + unknown, question, lang)
    cached = _cache_get(key)
    if cached:
        cached["from_cache"] = True
        _log({"event": "check", "drugs": drugs, "severity": cached["severity"], "cache": True,
              "lang": lang, "model": cached.get("model"), "ms": int((time.time() - t) * 1000)})
        return _finalize(cached, question, lang)

    # 4. Retrieve and answer; unknown names result in INSUFFICIENT_DATA
    if unknown or len(drugs) < 1:
        chunks = []
        result = backend.answer(drugs + unknown, [], question, lang)
    else:
        chunks = backend.retrieve(drugs, question, k=5)
        result = backend.answer(drugs, chunks, question, lang)

    # 5. Validate the result before it is shown
    errors = validate_result(result)
    if errors:
        _log({"event": "contract_error", "errors": errors})
        raise ValueError(f"backend returned an invalid CheckResult: {errors}")

    result["from_cache"] = False
    result["model"] = model_tag()
    result["safety_flag"] = None
    result["db_last_updated"] = max((c.get("db_last_updated", "") for c in chunks), default="")
    result = _finalize(result, question, lang)

    _cache_put(key, result)
    _log({"event": "check", "drugs": drugs, "unknown": len(unknown), "severity": result["severity"],
          "cache": False, "lang": lang, "model": result["model"], "ms": int((time.time() - t) * 1000)})
    return result


_API_DISCLAIMER_RE = re.compile(r"\s*This is not medical advice\. Please confirm with a pharmacist or doctor\.\s*$")


def _from_api_response(raw: dict, lang: str) -> CheckResult:
    """Map the service's JSON to a display result. Falls back to label parsing for old services."""
    if "severity" not in raw and "drugs" not in raw:
        return _from_model_text(raw["answer"], lang, model="b-api")
    text = _API_DISCLAIMER_RE.sub("", raw["answer"]).strip()   # the app shows its own disclaimer
    sev = raw.get("severity")
    if sev not in MODEL_LABELS:
        # no severity: gap / no-drug / confirmation prompts are "unknown"; other answers are information,
        # never a green light ("no documented interaction" is not "safe")
        unknown = raw.get("gap") or raw.get("status") in ("no_drug", "needs_confirmation")
        sev = "INSUFFICIENT_DATA" if unknown else "INFO"
    cites = []
    for c in raw.get("citations") or []:
        is_url = str(c).startswith("http")
        cites.append({"source": "FASS", "section": "SmPC" if is_url else str(c),
                      "url": str(c) if is_url else "https://fass.se"})
    flag = raw.get("safety_flag")
    return {
        "drugs": [d.get("name", "") if isinstance(d, dict) else str(d) for d in raw.get("drugs") or []],
        "atc_codes": [], "severity": sev, "explanation": text, "citations": cites,
        "confidence": 0.0, "disclaimer": DISCLAIMERS[lang], "language": lang,
        "from_cache": False, "db_last_updated": "", "model": "b-api", "safety_flag": flag,
        "grounded": bool(cites), "gap": bool(raw.get("gap")),
    }


def _from_model_text(answer: str, lang: str, model: str) -> CheckResult:
    """Convert a free-text answer from the model service into a display result."""
    label = safety.parse_label(answer)
    return {
        "drugs": [], "atc_codes": [],
        "severity": label or "INSUFFICIENT_DATA",
        "explanation": safety.strip_label_prefix(answer) if label else answer.strip(),
        "citations": [], "confidence": 0.0,
        "disclaimer": DISCLAIMERS[lang], "language": lang,
        "from_cache": False, "db_last_updated": "", "model": model, "safety_flag": None,
        "label_found": bool(label),
    }


def chat(message: str, lang: str = "sv", state=None, image_bytes: bytes | None = None,
         mime: str = "image/jpeg") -> dict:
    """Process one chat message and return the result, including the conversation state."""
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}")
    message = (message or "").strip()
    t = time.time()
    backend = backend_name()

    # 1. Safety rules before the model is called. Strength information is removed first,
    #    so that a message such as "Alvedon 500 mg + Ipren" is not treated as a dosage question.
    flag = safety.classify(clean_name(message))
    if flag:
        _log({"event": "guardrail", "flag": flag, "lang": lang, "backend": backend})
        return {**_canned(flag, lang), "state": state}

    # 2. Cache lookup, only for the first message without a photo; later messages depend on context
    first_turn = state in (None, {}, []) and not image_bytes and backend != "b_api"
    key = chat_cache_key(message, lang) if first_turn else None
    if key:
        cached = _cache_get(key)
        if cached:
            cached["from_cache"] = True
            _log({"event": "chat", "severity": cached["severity"], "cache": True, "lang": lang,
                  "model": cached.get("model"), "ms": int((time.time() - t) * 1000)})
            return cached

    # 3. Request an answer from the selected backend
    ocr = None
    if backend == "b_api":
        from . import b_client
        try:
            # Send the raw message: the service detects the language itself, and appending an
            # instruction here broke short replies such as "yes" (confirming an unsure drug name).
            reply = b_client.ask(message,
                                 state=state if state is not None else b_client.initial_state(),
                                 image_bytes=image_bytes)
            result = _from_api_response(reply.get("raw") or {"answer": reply["answer"]}, lang)
            new_state = reply["state"]
        except (b_client.BUnavailable, b_client.BBadResponse) as e:
            _log({"event": "backend_error", "backend": "b_api", "error": type(e).__name__,
                  "detail": str(e)[:200]})
            return {**_canned("backend_down", lang), "state": state}
    else:
        # Placeholder or module backend: identify medicine names in the text and photo, then check them
        b = _backend()
        if image_bytes:
            ocr = read_image(image_bytes, mime)
        vocabulary = getattr(b, "_NAMES", {}).keys()
        names = find_names_in_text(message, vocabulary) if vocabulary else parse_names(message)
        if ocr:
            names = list(dict.fromkeys([*ocr.get("names", []), *names]))
        result = dict(check(names, "", lang))
        result["from_cache"] = False
        new_state = {"turns": (state or {}).get("turns", 0) + 1} if isinstance(state, (dict, type(None))) else state

    result = _finalize(result, message, lang)
    if ocr is not None:
        result["ocr"] = ocr
    errors = validate_result(result)
    if errors:
        _log({"event": "contract_error", "errors": errors})
        raise ValueError(f"invalid result: {errors}")

    result["state"] = new_state
    if key:
        _cache_put(key, result)
    _log({"event": "chat", "severity": result["severity"], "cache": False, "lang": lang,
          "model": result["model"], "label_found": result.get("label_found"),
          "safety_flag": result.get("safety_flag"), "ms": int((time.time() - t) * 1000)})
    return result