"""Match free text (typed prompt or OCR output) to drugs in drugs_lookup.csv.

Handles: generic names, BRAND names (Waran, Ipren, Alvedon... via ATC code), typos
(fuzzy), multi-word names, and ignores ordinary words. Every result is returned under
its canonical lookup (generic) name so the rest of the pipeline sees one name per ATC.

Brand sources: BRAND_SEED_ATC below + llm_ayesha/brand_aliases.csv (built by
build_brand_aliases.py from the full FASS product list).
"""
import os
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

_HERE = Path(__file__).resolve().parent


def _first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return str(paths[-1])


LOOKUP_FILE = os.environ.get("DRUGS_LOOKUP_FILE") or _first_existing(
    _HERE / "llm_ayesha" / "drugs_lookup.csv",
    "/teamspace/studios/this_studio/LLMSCAI_G_19/llm_ayesha/drugs_lookup.csv",
)
BRAND_FILE = os.environ.get("BRAND_ALIASES_FILE") or str(_HERE / "llm_ayesha" / "brand_aliases.csv")

HIGH = 88   # auto-accept at/above this score
LOW = 70    # ask for confirmation between LOW and HIGH; below LOW = not found
MIN_FUZZY_LEN = 4
MARGIN = 3  # per candidate, keep only fuzzy hits within this many points of its best

# Common Swedish brands -> ATC code. ATC codes (not names) so spelling differences in the
# lookup never matter; anything whose ATC is not in drugs_lookup.csv is skipped silently.
BRAND_SEED_ATC = {
    "waran": "B01AA03", "marevan": "B01AA03",
    "ipren": "M01AE01", "brufen": "M01AE01", "burana": "M01AE01", "nurofen": "M01AE01",
    "alvedon": "N02BE01", "panodil": "N02BE01", "pamol": "N02BE01", "reliv": "N02BE01",
    "citodon": "N02AJ06", "panocod": "N02AJ06",
    "voltaren": "M01AB05", "naprosyn": "M01AE02", "trombyl": "B01AC06", "magnecyl": "N02BA01",
    "cipramil": "N06AB04", "zoloft": "N06AB06", "fontex": "N06AB03", "cipralex": "N06AB10",
    "remeron": "N06AX11", "efexor": "N06AX16",
    "sobril": "N05BA04", "stesolid": "N05BA01", "valium": "N05BA01", "xanor": "N05BA12",
    "imovane": "N05CF01", "stilnoct": "N05CF02", "atarax": "N05BB01",
    "lipitor": "C10AA05", "seloken": "C07AB02", "concor": "C07AB07",
    "furix": "C03CA01", "lasix": "C03CA01", "cozaar": "C09CA01",
    "losec": "A02BC01", "nexium": "A02BC05", "zofran": "A04AA01",
    "zyrtec": "R06AE07", "eliquis": "B01AF02", "xarelto": "B01AF01",
    "lamictal": "N03AX09", "tegretol": "N03AF01", "levaxin": "H03AA01",
    "glucophage": "A10BA02", "oxycontin": "N02AA05", "oxynorm": "N02AA05", "tradolan": "N02AX02",
}

# Ordinary words that must never be fuzzy-matched to a drug.
_STOPWORDS = set("""
can could may might should would will shall take taking taken this that these those with
without and the for from together same time what which when how why does did have has
medicine medication drug drugs pill pills tablet tablets capsule capsules mg dose
please about safe safely mixed mix also both with alcohol beer wine drink drinking
kan jag får ska med och att den det detta dessa ta tar tagit samma tid vad vilken
hur varför har medicin medicinen läkemedel tablett tabletter kapsel kapslar tillsammans
blanda blandas alkohol vin öl
""".split())
_WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ+.-]+\b")

# ----------------------------------------------------------------------- load data
_df = pd.read_csv(LOOKUP_FILE)
_NAME_TO_ATC = dict(zip(_df["name"].str.lower(), _df["atc"].str.upper()))
_ATC_TO_NAME = {}
for _n, _a in _NAME_TO_ATC.items():
    _ATC_TO_NAME.setdefault(_a, _n)          # canonical (first) name per ATC
_SUBSTANCES = list(_NAME_TO_ATC.keys())


def _load_brands():
    """brand (lowercase) -> ATC, restricted to ATC codes present in the lookup."""
    brands = {b: a for b, a in BRAND_SEED_ATC.items()}
    if Path(BRAND_FILE).exists():
        extra = pd.read_csv(BRAND_FILE, dtype=str)
        for b, a in zip(extra["brand"], extra["atc"]):
            brands.setdefault(str(b).lower(), str(a).upper())
    return {b: a for b, a in brands.items() if a in _ATC_TO_NAME and b not in _NAME_TO_ATC}


_BRANDS = _load_brands()
# Everything matchable: substance names + brand names -> ATC
_MATCHABLE = {**_NAME_TO_ATC, **_BRANDS}
_CHOICES = list(_MATCHABLE.keys())


def canonical_name(atc: str) -> str:
    return _ATC_TO_NAME.get(str(atc).upper(), "")


def brands_for(atc: str):
    return sorted(b for b, a in _BRANDS.items() if a == str(atc).upper())


def match_drugs(text_pieces, limit=5):
    """text_pieces: list[str] (OCR fragments, or [user_prompt]).
    Returns de-duplicated candidates, best first:
        [{"name": canonical generic name, "atc": str, "score": float,
          "confident": bool, "matched_as": text that matched (e.g. 'waran')}, ...]
    """
    if isinstance(text_pieces, str):
        text_pieces = [text_pieces]

    found = {}   # atc -> (score, matched_as)

    def keep(atc, score, matched_as):
        if atc not in found or score > found[atc][0]:
            found[atc] = (score, matched_as)

    for piece in text_pieces:
        piece = (piece or "").lower().strip()
        if not piece:
            continue

        # 1) exact whole-word / whole-phrase matches (substances and brands)
        covered = set()
        for name, atc in _MATCHABLE.items():
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", piece):
                keep(atc, 100.0, name)
                covered.update(_WORD_RE.findall(name))

        # 2) fuzzy on single words and adjacent 2-3 word phrases, skipping anything
        #    already explained by an exact match
        words = _WORD_RE.findall(piece)
        candidates = set(words)
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                candidates.add(" ".join(words[i:i + n]))

        for cand in candidates:
            toks = cand.split()
            if len(cand) < MIN_FUZZY_LEN:
                continue
            if toks[0] in _STOPWORDS or toks[-1] in _STOPWORDS:
                continue
            if any(t in covered for t in toks):
                continue
            hits = [m for m in process.extract(cand, _CHOICES, scorer=fuzz.ratio, limit=limit)
                    if m[1] >= LOW]
            if not hits:
                continue
            best = max(m[1] for m in hits)
            for name, score, _ in hits:
                if score >= best - MARGIN:
                    keep(_MATCHABLE[name], score, name)

    results = [
        {"name": canonical_name(atc), "atc": atc, "score": round(score, 1),
         "confident": score >= HIGH, "matched_as": matched_as}
        for atc, (score, matched_as) in found.items()
    ]
    results.sort(key=lambda r: -r["score"])
    return results


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "waran 5 mg tabletter"
    for r in match_drugs(text):
        print(r)
