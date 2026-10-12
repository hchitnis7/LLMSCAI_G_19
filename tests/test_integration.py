"""
Integration and quality tests for the orchestrator, the result format and the safety rules.

Run:  python -m pytest -q

The tests use the placeholder backend by default. Tests marked "stub_only" depend on
placeholder behaviour and can be excluded with:  python -m pytest -q -m "not stub_only"
"""
import csv
from pathlib import Path

import pytest

from src import medicheck, safety, stub_backend
from src.contracts import DISCLAIMERS, LANGUAGES, SEVERITIES, validate_result

CASES = Path(__file__).parent / "challenger_cases.csv"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Provide a separate cache and log directory for each test."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    yield tmp_path


# ---------------- Result format ----------------
def test_validate_accepts_good_result():
    good = stub_backend.answer(["ibuprofen", "warfarin"], stub_backend.retrieve(["ibuprofen", "warfarin"], ""), "", "en")
    assert validate_result(good) == []


@pytest.mark.parametrize("broken, problem", [
    ({"severity": "MAYBE"}, "bad severity"),
    ({"confidence": 1.5}, "confidence"),
    ({"language": "de"}, "bad language"),
    ({"disclaimer": ""}, "empty disclaimer"),
    ({"citations": [{"source": "FASS"}]}, "citation missing"),
])
def test_validate_rejects_broken_results(broken, problem):
    good = stub_backend.answer(["ibuprofen", "warfarin"], stub_backend.retrieve(["ibuprofen", "warfarin"], ""), "", "en")
    errors = validate_result({**good, **broken})
    assert any(problem in e for e in errors), errors


def test_disclaimer_exists_for_every_language():
    assert set(DISCLAIMERS) == set(LANGUAGES)


# ---------------- Name parsing ----------------
@pytest.mark.parametrize("text, expected", [
    ("Waran + Ipren", ["Waran", "Ipren"]),
    ("waran, ipren", ["waran", "ipren"]),
    ("Waran och Ipren", ["Waran", "Ipren"]),
    ("Alvedon with alcohol", ["Alvedon", "alcohol"]),
    ("  ", []),
])
def test_parse_names(text, expected):
    assert medicheck.parse_names(text) == expected


def test_clean_name_strips_strength():
    assert medicheck.clean_name("Alvedon 500 mg") == "Alvedon"
    assert medicheck.clean_name("Ipren 400mg") == "Ipren"


# ---------------- Orchestrator ----------------
@pytest.mark.parametrize("lang", LANGUAGES)
def test_every_result_honours_contract_and_has_disclaimer(lang):
    res = medicheck.check(["Waran", "Ipren"], "", lang)
    assert validate_result(res) == []
    assert res["disclaimer"] == DISCLAIMERS[lang]
    assert res["language"] == lang


def test_second_call_comes_from_cache():
    first = medicheck.check(["Waran", "Ipren"], "", "sv")
    second = medicheck.check(["Ipren", "Waran"], "", "sv")  # order must not matter
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert second["severity"] == first["severity"]


def test_requests_are_logged_without_personal_data(isolated):
    medicheck.check(["Waran", "Ipren"], "Kan jag ta dessa?", "sv")
    log = (isolated / "logs" / "requests.jsonl").read_text(encoding="utf-8")
    assert "warfarin" in log
    assert "Kan jag ta dessa" not in log  # free-text questions are not stored


def test_invalid_language_is_rejected():
    with pytest.raises(ValueError):
        medicheck.check(["Waran"], "", "de")


def test_broken_backend_answer_is_never_shown(monkeypatch):
    monkeypatch.setattr(stub_backend, "answer", lambda *a, **k: {"severity": "MAYBE"})
    with pytest.raises(ValueError):
        medicheck.check(["Waran", "Ipren"], "", "en")


@pytest.mark.stub_only
def test_blurry_photo_has_low_confidence():
    ext = medicheck.read_image(b"tiny", "image/jpeg")
    assert ext["confidence"] < 0.6 and ext["names"] == []


# ---------------- Safety rules ----------------
def test_crisis_never_reaches_the_backend(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("backend called for a crisis message")
    monkeypatch.setattr(stub_backend, "normalize", boom)
    res = medicheck.check([], "I want to end my life", "en")
    assert res["safety_flag"] == "crisis"
    assert "112" in res["explanation"]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_crisis_message_has_emergency_numbers_in_every_language(lang):
    assert "112" in safety.MESSAGES["crisis"][lang] and "1177" in safety.MESSAGES["crisis"][lang]


# ---------------- Challenger cases (rows that do not require photos or the model service) ----------------
def _runnable_rows():
    with open(CASES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["names"].startswith("(") or row["notes"].startswith(("Needs", "Run in UI", "B + C")):
                continue
            if row["expected_flag"] or row["expected_severity"]:
                yield pytest.param(row, id=row["id"])


@pytest.mark.parametrize("row", list(_runnable_rows()))
def test_challenger_case(row):
    res = medicheck.check(medicheck.parse_names(row["names"]), row["question"], row["lang"])
    if row["expected_flag"]:
        assert res["safety_flag"] == row["expected_flag"], row["expected_behaviour"]
    if row["expected_severity"]:
        assert res["severity"] == row["expected_severity"], row["expected_behaviour"]
    assert res["severity"] in SEVERITIES
    assert res["disclaimer"]
