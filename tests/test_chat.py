"""
Tests for the chat flow, the model service client and the post-model safety measures.

The model service is simulated, so no network access is required.
Run:  python -m pytest -q
"""
import json

import pytest
import requests

from src import medicheck, safety
from src.contracts import DISCLAIMERS, LANGUAGE_INSTRUCTION, LANGUAGES, PREGNANCY_LINE, validate_result

B_URL = "https://example-b.ngrok-free.dev"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BACKEND", "stub")
    monkeypatch.setenv("B_API_URL", B_URL)
    monkeypatch.setenv("B_API_KEY", "")
    yield tmp_path


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)
        self.ok = status < 400

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeB:
    """Simulated model service that records each request and returns a fixed answer."""

    def __init__(self, answer="DANGEROUS: Higher risk of bleeding.", state=None, status=200, raw=None, exc=None):
        self.calls = []
        self.answer, self.state, self.status, self.raw, self.exc = answer, state, status, raw, exc

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers or {}})
        if self.exc:
            raise self.exc
        if self.raw is not None:
            return FakeResponse(self.status, None, self.raw)
        return FakeResponse(self.status, {"answer": self.answer,
                                          "state": self.state if self.state is not None else {"turn": len(self.calls)}})


def use_fake_b(monkeypatch, **kw):
    fake = FakeB(**kw)
    monkeypatch.setenv("BACKEND", "b_api")
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


# ---------------- Label and dosage detection ----------------
@pytest.mark.parametrize("text, label", [
    ("DANGEROUS: Higher risk of bleeding.", "DANGEROUS"),
    ("**Severity: AVOID** - do not combine", "AVOID"),
    ("[CAUTION] Alcohol and paracetamol…", "CAUTION"),
    ("INFO\nParacetamol is a painkiller.", "INFO"),
    ("FARLIGT: ökad blödningsrisk", "DANGEROUS"),
    ("This is not dangerous to combine.", None),   # lower-case wording is not a label
    ("No label at all.", None),
])
def test_parse_label(text, label):
    assert safety.parse_label(text) == label


def test_label_prefix_is_removed_from_text():
    assert safety.strip_label_prefix("DANGEROUS: Higher risk of bleeding.") == "Higher risk of bleeding."


@pytest.mark.parametrize("text, has_dose", [
    ("Take 2 tablets three times a day.", True),
    ("Ta 2 tabletter tre gånger om dagen.", True),
    ("The usual amount is 1000 mg.", True),
    ("Avoid combining these medicines.", False),
    ("Ask your pharmacist before combining them.", False),
])
def test_dose_detection_in_answers(text, has_dose):
    assert safety.contains_dose_advice(text) is has_dose


# ---------------- Chat with the placeholder backend ----------------
def test_free_text_sentence_finds_medicines():
    res = medicheck.chat("Can I take Waran with Ipren?", "en")
    assert res["severity"] == "DANGEROUS"
    assert validate_result(res) == []


def test_strength_in_message_is_not_a_dose_question():
    res = medicheck.chat("Kan jag ta Alvedon 500 mg och Ipren?", "sv")
    assert res.get("safety_flag") is None


def test_pregnancy_adds_fixed_line():
    res = medicheck.chat("I am preganat and had alcohol, can I take Alvedon?", "en")
    assert PREGNANCY_LINE["en"] in res["notes"]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_every_chat_answer_has_fixed_disclaimer_and_pharmacist_line(lang):
    res = medicheck.chat("Waran + Ipren", lang)
    assert res["disclaimer"] == DISCLAIMERS[lang]
    assert len(res["notes"]) >= 1


# ---------------- Model service client ----------------
def test_b_answer_is_shown_with_badge_and_state(monkeypatch):
    fake = use_fake_b(monkeypatch, answer="DANGEROUS: Higher risk of bleeding.", state={"meds": ["warfarin"]})
    res = medicheck.chat("Can I take Waran with Ipren?", "en")
    assert res["severity"] == "DANGEROUS"
    assert res["explanation"] == "Higher risk of bleeding."
    assert res["model"] == "b-api"
    assert res["state"] == {"meds": ["warfarin"]}
    assert validate_result(res) == []
    sent = fake.calls[0]
    assert sent["url"] == f"{B_URL}/answer"
    assert set(sent["json"]) == {"text", "image_b64", "state"}
    assert sent["json"]["image_b64"] is None
    assert sent["headers"]["ngrok-skip-browser-warning"] == "true"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_language_instruction_is_sent_to_b(monkeypatch, lang):
    fake = use_fake_b(monkeypatch)
    medicheck.chat("Waran + Ipren", lang)
    assert LANGUAGE_INSTRUCTION[lang] in fake.calls[0]["json"]["text"]


def test_state_is_sent_back_on_the_next_turn(monkeypatch):
    fake = use_fake_b(monkeypatch, state={"meds": ["warfarin"]})
    first = medicheck.chat("I take Waran", "en")
    medicheck.chat("Can I also take Ipren?", "en", state=first["state"])
    assert fake.calls[1]["json"]["state"] == {"meds": ["warfarin"]}


def test_photo_is_sent_as_base64(monkeypatch):
    fake = use_fake_b(monkeypatch)
    medicheck.chat("What is this?", "en", image_bytes=b"\x89PNG fake image")
    assert fake.calls[0]["json"]["image_b64"]


def test_key_is_sent_when_configured(monkeypatch):
    monkeypatch.setenv("B_API_KEY", "secret123")
    fake = use_fake_b(monkeypatch)
    medicheck.chat("Waran + Ipren", "en")
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer secret123"


def test_info_label_is_kept(monkeypatch):
    use_fake_b(monkeypatch, answer="INFO: Paracetamol is a common painkiller.")
    assert medicheck.chat("What is Alvedon?", "en")["severity"] == "INFO"


def test_answer_without_label_is_unknown_not_safe(monkeypatch):
    use_fake_b(monkeypatch, answer="These can usually be combined.")
    res = medicheck.chat("Alvedon and Ipren?", "en")
    assert res["severity"] == "INSUFFICIENT_DATA"
    assert res["label_found"] is False


def test_dose_in_b_answer_is_replaced(monkeypatch):
    use_fake_b(monkeypatch, answer="CAUTION: Take 2 tablets of 500 mg at most.")
    res = medicheck.chat("Alvedon with alcohol?", "en")
    assert res["safety_flag"] == "dose_in_answer"
    assert "500 mg" not in res["explanation"]
    assert res["severity"] == "CAUTION"


def test_crisis_never_reaches_b(monkeypatch):
    fake = use_fake_b(monkeypatch)
    res = medicheck.chat("I want to end my life", "en")
    assert res["safety_flag"] == "crisis" and fake.calls == []


@pytest.mark.parametrize("problem", [
    dict(exc=requests.ConnectionError("service offline")),
    dict(exc=requests.Timeout("too slow")),
    dict(status=500, raw="Internal Server Error"),
    dict(status=422, raw='{"detail": "bad state"}'),
    dict(status=200, raw="<html>ngrok endpoint offline</html>"),
])
def test_b_problems_give_a_friendly_message(monkeypatch, problem):
    use_fake_b(monkeypatch, **problem)
    res = medicheck.chat("Waran + Ipren", "en")
    assert res["safety_flag"] == "backend_down"
    assert res["state"] is None


def test_first_turn_is_cached_but_later_turns_are_not(monkeypatch):
    fake = use_fake_b(monkeypatch, state={"meds": ["warfarin"]})
    medicheck.chat("Waran + Ipren", "en")
    again = medicheck.chat("Waran + Ipren", "en")
    assert again["from_cache"] is True and len(fake.calls) == 1
    medicheck.chat("And alcohol?", "en", state={"meds": ["warfarin"]})
    medicheck.chat("And alcohol?", "en", state={"meds": ["warfarin"]})
    assert len(fake.calls) == 3


def test_chat_log_has_no_user_text(monkeypatch, isolated):
    use_fake_b(monkeypatch)
    medicheck.chat("I am pregnant, can I take Waran with Ipren?", "en")
    log = (isolated / "logs" / "requests.jsonl").read_text(encoding="utf-8")
    assert "pregnant" not in log
