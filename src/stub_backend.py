"""
Placeholder backend with the same function signatures as the module-based backend.

Used for interface development, automated tests and offline demonstrations (BACKEND=stub).
The answers are fixed placeholders for six medicine pairs. They are not medical content and
are marked as test data in the interface.
"""
from __future__ import annotations

from .contracts import DISCLAIMERS, Chunk, CheckResult, Extraction, NormalizedDrug

DB_DATE = "2026-09-28"

# Brand or generic name -> (generic name, ATC code)
_NAMES = {
    "waran": ("warfarin", "B01AA03"),
    "warfarin": ("warfarin", "B01AA03"),
    "ipren": ("ibuprofen", "M01AE01"),
    "ibuprofen": ("ibuprofen", "M01AE01"),
    "alvedon": ("paracetamol", "N02BE01"),
    "paracetamol": ("paracetamol", "N02BE01"),
    "amoxicillin": ("amoxicillin", "J01CA04"),
    "simvastatin": ("simvastatin", "C10AA01"),
    "klaritromycin": ("clarithromycin", "J01FA09"),
    "clarithromycin": ("clarithromycin", "J01FA09"),
    "metformin": ("metformin", "A10BA02"),
    "alkohol": ("alcohol", ""),
    "alcohol": ("alcohol", ""),
}

# Sorted pair of generic names -> (severity, confidence, placeholder explanation)
_PAIRS = {
    ("ibuprofen", "warfarin"): ("DANGEROUS", 0.9,
        "[STUB] Placeholder: NSAIDs combined with warfarin are listed as a bleeding-risk interaction."),
    ("ibuprofen", "paracetamol"): ("SAFE", 0.8,
        "[STUB] Placeholder: no clinically relevant interaction listed for this pair."),
    ("amoxicillin", "paracetamol"): ("SAFE", 0.85,
        "[STUB] Placeholder: no clinically relevant interaction listed for this pair."),
    ("clarithromycin", "simvastatin"): ("AVOID", 0.85,
        "[STUB] Placeholder: this combination is listed as one to avoid."),
    ("ibuprofen", "metformin"): ("CAUTION", 0.7,
        "[STUB] Placeholder: listed as needing caution and monitoring."),
    ("alcohol", "paracetamol"): ("CAUTION", 0.7,
        "[STUB] Placeholder: listed as needing caution with regular or heavy use."),
}

_PREFIX = {"sv": "[Svenska] ", "en": "", "ar": "[العربية] "}


def normalize(names: list[str]) -> list[NormalizedDrug]:
    out: list[NormalizedDrug] = []
    for n in names:
        generic, atc = _NAMES.get(n.strip().lower(), ("", ""))
        out.append({"input": n, "generic": generic, "atc": atc})
    return out


def retrieve(drugs: list[str], question: str, k: int = 5) -> list[Chunk]:
    key = tuple(sorted(drugs))
    if key not in _PAIRS:
        return []
    return [{
        "text": _PAIRS[key][2],
        "source": "FASS",
        "url": "https://www.fass.se/",
        "section": "Interaktioner",
        "chunk_type": "interaction",
        "db_last_updated": DB_DATE,
    }]


def extract_drugs(image_bytes: bytes, mime: str = "image/jpeg") -> Extraction:
    # Every photo is treated as an Alvedon package; a very small file simulates an unreadable photo
    if len(image_bytes) < 2_000:
        return {"names": [], "confidence": 0.2}
    return {"names": ["Alvedon"], "confidence": 0.92}


def answer(drugs: list[str], chunks: list[Chunk], question: str, lang: str) -> CheckResult:
    key = tuple(sorted(drugs))
    if not chunks or key not in _PAIRS:
        severity, conf = "INSUFFICIENT_DATA", 0.0
        text = ("[STUB] Insufficient data in the database for this combination. "
                "Please ask your pharmacist (apotekare).")
    else:
        severity, conf, text = _PAIRS[key]
    return {
        "drugs": list(drugs),
        "atc_codes": [_NAMES.get(d, ("", ""))[1] for d in drugs],
        "severity": severity,
        "explanation": _PREFIX.get(lang, "") + text,
        "citations": [{"source": c["source"], "url": c["url"], "section": c["section"]} for c in chunks],
        "confidence": conf,
        "disclaimer": DISCLAIMERS[lang],
        "language": lang,
    }
