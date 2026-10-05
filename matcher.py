"""Fuzzy-match free text (OCR output or typed prompt) against the drug lookup.

Run build_drug_lookup.py once first to create drugs_lookup.csv.
"""
import os
import re

import pandas as pd
from rapidfuzz import fuzz, process

LOOKUP_FILE = os.environ.get("DRUGS_LOOKUP_FILE", "/teamspace/studios/this_studio/LLMSCAI_G_19/llm_ayesha/drugs_lookup.csv")
HIGH = 88   # auto-accept above this score
LOW = 70    # ask for confirmation between LOW and HIGH; below LOW = not found
MIN_FUZZY_LEN = 4   # ignore very short words for fuzzy matching (too noisy)
MARGIN = 3          # per candidate, drop fuzzy hits this far below its best hit

# Common English/Swedish words that must never be fuzzy-matched to a drug.
_STOPWORDS = set("""
can could may might should would will shall take taking taken this that these those with
without and the for from together same time what which when how why does did have has
medicine medication drug drugs pill pills tablet tablets capsule capsules mg dose
please about safe safely together mixed mix also both
kan jag får ska med och att den det detta dessa ta tar tagit samma tid vad vilken
hur varför har medicin medicinen läkemedel tablett tabletter kapsel kapslar tillsammans
blanda blandas
""".split())
_WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ+.-]+\b")

# Swedish brand names -> generic name. The lookup built from FASS may contain only
# generic names, so "Waran"/"Ipren" would otherwise be fuzzy-guessed into the wrong
# drugs. Aliases are resolved against the real lookup at import time; any generic
# that can't be found in drugs_lookup.csv is silently skipped.
BRAND_TO_GENERIC = {
    "waran": "warfarin", "marevan": "warfarin",
    "ipren": "ibuprofen", "brufen": "ibuprofen", "burana": "ibuprofen",
    "alvedon": "paracetamol", "panodil": "paracetamol", "pamol": "paracetamol",
    "voltaren": "diklofenak", "naprosyn": "naproxen",
    "trombyl": "acetylsalicylsyra", "aspirin": "acetylsalicylsyra",
    "cipramil": "citalopram", "zoloft": "sertralin",
    "sobril": "oxazepam", "stesolid": "diazepam", "imovane": "zopiklon",
    "lipitor": "atorvastatin", "seloken": "metoprolol",
    "losec": "omeprazol", "nexium": "esomeprazol", "zyrtec": "cetirizin",
    "eliquis": "apixaban", "xarelto": "rivaroxaban",
    "furix": "furosemid", "lasix": "furosemid", "cozaar": "losartan", "concor": "bisoprolol",
}

_df = pd.read_csv(LOOKUP_FILE)
_NAME_TO_ATC = dict(zip(_df["name"].str.lower(), _df["atc"]))
_CHOICES = list(_NAME_TO_ATC.keys())


def _resolve_aliases():
    out = {}
    for brand, generic in BRAND_TO_GENERIC.items():
        if brand in _NAME_TO_ATC:          # brand already in the lookup: nothing to do
            continue
        hit = generic if generic in _NAME_TO_ATC else None
        if hit is None:
            m = process.extractOne(generic, _CHOICES, scorer=fuzz.ratio, score_cutoff=88)
            hit = m[0] if m else None
        if hit:
            out[brand] = hit
    return out


_ALIASES = _resolve_aliases()   # {"waran": "<lookup name for warfarin>", ...}


def match_drugs(text_pieces, limit=5):
    """text_pieces: list[str] (OCR lines, or [user_prompt]).
    Returns a de-duplicated list of candidates, best match first:
        [{"name": str, "atc": str, "score": float, "confident": bool}, ...]
    """
    if isinstance(text_pieces, str):
        text_pieces = [text_pieces]

    found = {}

    for piece in text_pieces:
        piece = piece.lower().strip()
        if not piece:
            continue

        # 1) Exact drug-name matches in the text (whole-word / whole-phrase).
        covered = set()   # tokens already explained by an exact match
        for name in _CHOICES:
            pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
            if re.search(pattern, piece, flags=re.IGNORECASE):
                found[(name, _NAME_TO_ATC[name])] = 100.0
                covered.update(_WORD_RE.findall(name))
        for brand, generic in _ALIASES.items():
            if re.search(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", piece):
                found[(generic, _NAME_TO_ATC[generic])] = 100.0
                covered.add(brand)

        # 2) Fuzzy matching on single words and adjacent 2-3 word phrases,
        # skipping anything already explained by an exact match (otherwise
        # "ibuprofen" also drags in "dexibuprofen" as an unconfirmed match).
        words = _WORD_RE.findall(piece)
        candidates = set(words)
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                candidates.add(" ".join(words[i:i + n]))

        for candidate in candidates:
            toks = candidate.split()
            if len(candidate) < MIN_FUZZY_LEN:
                continue
            if all(t in _STOPWORDS for t in toks) or toks[0] in _STOPWORDS or toks[-1] in _STOPWORDS:
                continue
            if any(t in covered for t in toks):
                continue

            matches = [
                m for m in process.extract(
                    candidate, _CHOICES, scorer=fuzz.ratio, limit=limit
                ) if m[1] >= LOW
            ]
            if not matches:
                continue
            best = max(m[1] for m in matches)
            for name, score, _ in matches:
                if score < best - MARGIN:
                    continue
                key = (name, _NAME_TO_ATC[name])
                if key not in found or score > found[key]:
                    found[key] = score

    results = [
        {"name": name, "atc": atc, "score": round(score, 1), "confident": score >= HIGH}
        for (name, atc), score in found.items()
    ]

    results.sort(key=lambda r: -r["score"])
    return results


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "waran 5 mg tabletter"
    for r in match_drugs(text):
        print(r)