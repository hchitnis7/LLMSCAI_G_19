"""
Shared data definitions for the MediCheck application.

This module defines the result structure displayed by the user interface, the supported
severity labels and languages, and the fixed safety texts (disclaimer, pharmacist and
pregnancy notices). Changes to these definitions affect the interface, the orchestrator
and the test suite, and should be agreed within the team before they are made.

Public functions of the orchestrator (src/medicheck.py):
    chat(message, lang, state, image_bytes, mime) -> result including "state"
    check(names, question, lang)                  -> result
    read_image(image_bytes, mime)                 -> Extraction

Model service interface (provided by the model team, accessed through src/b_client.py):
    POST {B_API_URL}/answer   request:  {"text": str, "image_b64": str | null, "state": ...}
                              response: {"answer": str, "state": ...}
    GET  {B_API_URL}/health

The NormalizedDrug, Chunk and Extraction types describe the optional module-based
integration (BACKEND=modules) and the placeholder backend.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

# DANGEROUS, AVOID, CAUTION and INFO are the labels produced by the model service.
# SAFE is used only by the placeholder backend. INSUFFICIENT_DATA is assigned by the
# application when no label can be determined or when a safety reply is returned.
Severity = Literal["DANGEROUS", "AVOID", "CAUTION", "INFO", "SAFE", "INSUFFICIENT_DATA"]
SEVERITIES: tuple[str, ...] = ("DANGEROUS", "AVOID", "CAUTION", "INFO", "SAFE", "INSUFFICIENT_DATA")
MODEL_LABELS: tuple[str, ...] = ("DANGEROUS", "AVOID", "CAUTION", "INFO")

Language = Literal["sv", "en", "ar"]
LANGUAGES: tuple[str, ...] = ("sv", "en", "ar")


# ---------- Normalisation and retrieval types ----------
class NormalizedDrug(TypedDict):
    input: str            # name as entered or read from a photo, e.g. "Waran"
    generic: str          # generic name, e.g. "warfarin"; empty if unknown
    atc: str              # ATC code, e.g. "B01AA03"; empty if unknown


class Chunk(TypedDict):
    text: str
    source: str           # source name, e.g. "FASS", "Läkemedelsverket", "1177"
    url: str
    section: str          # section title, e.g. "Interaktioner"
    chunk_type: str       # content type, e.g. "interaction", "side_effects", "pregnancy"
    db_last_updated: str  # ISO date on which the source data was collected


# ---------- Image extraction and result types ----------
class Extraction(TypedDict):
    names: list[str]      # medicine names recognised in the photo
    confidence: float     # recognition confidence between 0 and 1


class Citation(TypedDict):
    source: str
    url: str
    section: str


class CheckResult(TypedDict, total=False):
    # Required fields
    drugs: list[str]            # normalised generic names
    atc_codes: list[str]
    severity: Severity
    explanation: str            # explanation in the requested language
    citations: list[Citation]
    confidence: float           # between 0 and 1; 0 when unknown
    disclaimer: str
    language: str
    # Fields set by the orchestrator
    from_cache: bool
    db_last_updated: str
    model: str
    safety_flag: Optional[str]  # name of the safety rule that was applied, otherwise None


REQUIRED_KEYS = ("drugs", "atc_codes", "severity", "explanation", "citations",
                 "confidence", "disclaimer", "language")

LOW_CONFIDENCE = 0.6   # below this value the interface asks the user to verify the names

DISCLAIMERS = {
    "sv": ("MediCheck är ett AI-verktyg i ett universitetsprojekt, inte en farmaceut eller läkare. "
           "Kontrollera alltid med ditt apotek, din läkare eller 1177 innan du ändrar din medicinering."),
    "en": ("MediCheck is an AI tool from a university project, not a pharmacist or doctor. "
           "Always check with your pharmacy, your doctor or 1177 before changing your medication."),
    "ar": ("MediCheck أداة ذكاء اصطناعي من مشروع جامعي، وليست صيدليًا أو طبيبًا. "
           "استشر دائمًا الصيدلية أو طبيبك أو 1177 قبل أي تغيير في أدويتك."),
}


# Fixed notices added by the application to every model answer
PHARMACIST_LINE = {
    "sv": "Fråga alltid ditt apotek eller din läkare innan du kombinerar läkemedel.",
    "en": "Always ask your pharmacist or doctor before combining medicines.",
    "ar": "اسأل دائمًا الصيدلي أو الطبيب قبل الجمع بين الأدوية.",
}
PREGNANCY_LINE = {
    "sv": "Du nämner graviditet: kontakta din barnmorska, läkare eller 1177 innan du tar något läkemedel.",
    "en": "You mention pregnancy: contact your midwife, doctor or 1177 before taking any medicine.",
    "ar": "ذكرتِ الحمل: تواصلي مع القابلة أو الطبيب أو 1177 قبل تناول أي دواء.",
}
LANGUAGE_INSTRUCTION = {
    "sv": "Svara på svenska.",
    "en": "Please answer in English.",
    "ar": "يرجى الإجابة باللغة العربية.",
}


def validate_result(result: dict) -> list[str]:
    """Return a list of validation errors; an empty list means the result is valid."""
    if not isinstance(result, dict):
        return ["result is not a dict"]
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in result:
            errors.append(f"missing key: {key}")
    if result.get("severity") not in SEVERITIES:
        errors.append(f"bad severity: {result.get('severity')!r}")
    if result.get("language") not in LANGUAGES:
        errors.append(f"bad language: {result.get('language')!r}")
    conf = result.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        errors.append(f"confidence must be a number 0-1, got {conf!r}")
    if not isinstance(result.get("drugs", []), list):
        errors.append("drugs must be a list")
    for c in result.get("citations", []) or []:
        if not isinstance(c, dict) or not all(k in c for k in ("source", "url", "section")):
            errors.append(f"citation missing fields: {c}")
    if not str(result.get("disclaimer", "")).strip():
        errors.append("empty disclaimer")
    return errors
