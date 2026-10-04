"""Fuzzy-match free text (OCR output or typed prompt) against the drug lookup.

Run build_drug_lookup.py once first to create drugs_lookup.csv.
"""
import pandas as pd
from rapidfuzz import fuzz, process

LOOKUP_FILE = "LLMSCAI_G_19/llm_ayesha/drugs_lookup.csv"
HIGH = 88   # auto-accept above this score
LOW = 70    # ask for confirmation between LOW and HIGH; below LOW = not found

_df = pd.read_csv(LOOKUP_FILE)
_NAME_TO_ATC = dict(zip(_df["name"].str.lower(), _df["atc"]))
_CHOICES = list(_NAME_TO_ATC.keys())


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
        # whole-string match (handles multi-word names)
        matches = process.extract(piece, _CHOICES, scorer=fuzz.partial_ratio, limit=limit)
        for name, score, _ in matches:
            if score < LOW:
                continue
            atc = _NAME_TO_ATC[name]
            key = (name, atc)
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
