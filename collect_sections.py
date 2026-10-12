"""Collects FASS SmPC sections (4.1 indications, 4.6 pregnancy, 4.7 driving,
4.8 side effects, etc.) for the drugs already in your data, so the chatbot
can answer "what is this for?" / "can I drive on this?" / "is this safe in
pregnancy?" -- questions fass_pair_chunks.csv and fass_fallback_chunks.csv
were never built to answer, since those only hold interaction text.

*** RUN THIS ON YOUR OWN MACHINE, NOT IN A SANDBOX *** -- it needs normal
internet access to fass.se. Per the project's established data policy: check
https://fass.se/robots.txt yourself before running this at scale, and this
is for the same academic/research use already covered for the rest of the
FASS data in this project.

*** VERIFICATION STATUS *** -- I fetched a real FASS SmPC page
(fass.se/health/product/19920703000054/smpc, Alvedon suppositories) and
confirmed the section headers really do follow the "4.1 Terapeutiska
indikationer" pattern, with two messy real-world wrinkles this parser is
built to handle: (1) some headers have NO space between the number and the
title ("4.3Kontraindikationer"), and (2) the page repeats the whole clinical
section block once; parse_sections() keeps the LAST occurrence. I tested
parse_sections() directly against a reconstruction of that real page's
content and confirmed it extracts the right text into the right section key
with no bleed from neighboring sections. What is NOT verified from here: the
URL-derivation pattern (_smpc_url_from) against YOUR actual source_url
values (I only tested it against example URLs I found via search, which
matched correctly), and the live requests.get() fetch, since this sandbox
can't reach fass.se. Still run --debug_urls and --limit 5 first and read the
output by eye before a full run -- just know the riskiest part (HTML
parsing) is the part I could actually ground-truth.

What it does:
  1. Reads every unique (product_name, atc_code, source_url) combination out
     of fass_pair_chunks.csv and fass_fallback_chunks.csv.
  2. Derives the full SmPC page URL from whatever source_url is already
     stored (interaction pages and SmPC pages share the same nplId; see
     _smpc_url_from()). If your source_urls don't match the expected pattern,
     fix _smpc_url_from() to match what's actually in your data -- print a
     few of them first with --debug_urls.
  3. Fetches each SmPC page once (cached locally, so re-runs skip what's
     already fetched), parses out each numbered subsection under "4 KLINISKA
     UPPGIFTER", and keeps the ones in SECTION_LABELS.
  4. Appends rows to llm_ayesha/fass_sections.csv (resumable -- safe to
     stop and re-run; already-collected product_name+section pairs are
     skipped).

Usage:
    python collect_sections.py --debug_urls          # just print a few derived URLs, no fetching
    python collect_sections.py --limit 5              # fetch + parse 5 products, check the output by eye
    python collect_sections.py                        # full run (can take a while; polite rate limit)
    python collect_sections.py --sections indications side_effects   # only these sections
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from knowledge import SECTION_LABELS
except ImportError:  # knowledge.py missing from this folder (or shadowed by a pip package)
    SECTION_LABELS = {
        "indications": "FASS SmPC 4.1 (indications)", "dosage": "FASS SmPC 4.2",
        "contraindications": "FASS SmPC 4.3 (contraindications)", "warnings": "FASS SmPC 4.4 (warnings)",
        "interactions": "FASS SmPC 4.5 (interactions)",
        "pregnancy_breastfeeding": "FASS SmPC 4.6 (pregnancy and breastfeeding)",
        "driving": "FASS SmPC 4.7 (driving and machines)", "side_effects": "FASS SmPC 4.8 (side effects)",
        "overdose": "FASS SmPC 4.9 (overdose)",
    }

DATA_DIR = Path(os.environ.get("MEDICHECK_DATA_DIR", "/teamspace/studios/this_studio/LLMSCAI_G_19/llm_ayesha"))
OUT_FILE = DATA_DIR / "fass_sections.csv"
CACHE_DIR = DATA_DIR / "fass_html_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (academic project, medication interaction checker)"}
RATE_LIMIT_SECONDS = float(os.environ.get("FASS_DELAY", "1.5"))  # seconds between downloads; lower = faster but ruder

# Section number -> our internal key (matches knowledge.py's SECTION_LABELS)
SECTION_NUMBER_TO_KEY = {
    "4.1": "indications", "4.2": "dosage", "4.3": "contraindications",
    "4.4": "warnings", "4.5": "interactions", "4.6": "pregnancy_breastfeeding",
    "4.7": "driving", "4.8": "side_effects", "4.9": "overdose",
}
_NPL_ID_RE = re.compile(r"nplId=(\d+)|/product/(\d+)/")


def _smpc_url_from(source_url: str) -> str | None:
    """Derives the full SmPC page URL from whatever source_url is already
    stored in the project's data. FASS's interaction/fass-text pages and its
    full SmPC page share the same nplId, just under different paths:
        https://www.fass.se/LIF/product?...&nplId=19920703000054&...
        https://fass.se/health/product/19920703000054/fass-text
        -> https://fass.se/health/product/19920703000054/smpc
    If this doesn't match what's actually in your source_url column, run
    with --debug_urls to see real examples and adjust this function."""
    if not isinstance(source_url, str) or not source_url:
        return None
    m = _NPL_ID_RE.search(source_url)
    if not m:
        return None
    npl_id = m.group(1) or m.group(2)
    return f"https://fass.se/health/product/{npl_id}/smpc"


def _cache_path(npl_id: str) -> Path:
    return CACHE_DIR / f"{npl_id}.html"


def fetch_html(url: str) -> str | None:
    npl_id_match = _NPL_ID_RE.search(url)
    npl_id = (npl_id_match.group(1) or npl_id_match.group(2)) if npl_id_match else url
    cache_path = _cache_path(npl_id)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        time.sleep(RATE_LIMIT_SECONDS)
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  FAILED to fetch {url}: {type(e).__name__}: {e}")
        return None


# A clinical-section header line, e.g. "4.1 Terapeutiska indikationer" or,
# just as commonly on FASS (confirmed on a live page), "4.3Kontraindikationer"
# with NO space before the title. \s* (zero-or-more) handles both. Headers
# are short, single lines -- the max-length check guards against a body
# sentence that happens to start with a number pattern (rare, since get_text
# with a newline separator breaks at block-element boundaries, but table
# cells can still produce short numeric-looking lines).
_HEADER_RE = re.compile(r"^(\d\.\d+)\s*([A-ZÅÄÖ].{0,100})$")
_TOP_LEVEL_RE = re.compile(r"^\d+\s+[A-ZÅÄÖ]{3,}")  # e.g. "5 FARMAKOLOGISKA EGENSKAPER" -- leaving section 4


_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')
_SKIP_VALUE_KEYS = re.compile(r'"(className|id|key|href|src|alt|type|rel|target|role|style|name|lang|dir|headingLevel|variant|size|color|as)":$')
_TAG_CTX = re.compile(r'\["\$",$')
_KEY_CTX = re.compile(r'\["\$","[^"]*",$')


def _lines_from_rsc(html: str) -> list:
    """fass.se is a Next.js app: the visible text lives inside
    self.__next_f.push([1,"..."]) JSON chunks, not in normal HTML tags.
    Decode those chunks and pull out only the human-readable strings."""
    chunks = []
    for m in _PUSH_RE.finditer(html):
        try:
            chunks.append(json.loads('"' + m.group(1) + '"'))
        except ValueError:
            continue
    stream = "".join(chunks)
    texts = []
    for m in _STR_RE.finditer(stream):
        raw = m.group(1)
        before = stream[max(0, m.start() - 40): m.start()]
        after = stream[m.end(): m.end() + 1]
        if after == ":":                      # an object key
            continue
        if _SKIP_VALUE_KEYS.search(before) or _TAG_CTX.search(before) or _KEY_CTX.search(before):
            continue
        try:
            val = json.loads('"' + raw + '"')
        except ValueError:
            continue
        val = val.strip()
        if not val or val.startswith("$") or val in ("children", "null") or re.fullmatch(r"h[1-6]", val):
            continue
        texts.append(val)
    # join "4.1" + "Terapeutiska indikationer" into one header line
    lines, i = [], 0
    while i < len(texts):
        t = texts[i]
        if re.fullmatch(r"\d\.\d+", t) and i + 1 < len(texts):
            lines.append(f"{t} {texts[i + 1]}")
            i += 2
        else:
            lines.append(t)
            i += 1
    return lines


def _lines_from_html(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return [l.strip() for l in soup.get_text("\n").split("\n") if l.strip()]


def _sections_from_lines(lines: list) -> dict:
    sections = {}
    current_key, buf = None, []

    def flush():
        if current_key and buf:
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            # FASS repeats blocks (mobile + desktop): keep the longest version
            if text and len(text) >= len(sections.get(current_key, "")):
                sections[current_key] = text

    for line in lines:
        m = _HEADER_RE.match(line)
        if m and m.group(1) in SECTION_NUMBER_TO_KEY:
            flush()
            current_key, buf = SECTION_NUMBER_TO_KEY[m.group(1)], []
            continue
        if m or _TOP_LEVEL_RE.match(line):
            flush()
            current_key, buf = None, []
            continue
        if current_key:
            buf.append(line)
    flush()
    return sections


def parse_sections(html: str) -> dict:
    """Returns {section_key: text}. Tries the Next.js payload first (what
    fass.se actually serves), then falls back to plain HTML text."""
    sections = _sections_from_lines(_lines_from_rsc(html))
    if not sections:
        sections = _sections_from_lines(_lines_from_html(html))
    return sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process N products, for a quick check")
    ap.add_argument("--sections", nargs="+", default=None,
                     help="only collect these section keys, e.g. --sections indications side_effects. "
                          "Default: all of SECTION_LABELS.")
    ap.add_argument("--debug_urls", action="store_true",
                     help="print a few derived SmPC URLs and exit, without fetching anything -- "
                          "use this first to sanity-check _smpc_url_from() against your actual data.")
    args = ap.parse_args()
    wanted_sections = set(args.sections) if args.sections else set(SECTION_NUMBER_TO_KEY.values())

    pair = pd.read_csv(DATA_DIR / "fass_pair_chunks.csv", dtype=str).fillna("")
    fallback = pd.read_csv(DATA_DIR / "fass_fallback_chunks.csv", dtype=str).fillna("")

    products = {}  # npl_id -> {"atc":..., "drug_name":..., "product_name":..., "source_url":...}
    for _, r in pair.iterrows():
        for atc_col, name_col in [("drug_1_atc", "drug_1_name"), ("drug_2_atc", "drug_2_name")]:
            url = _smpc_url_from(r.get("source_url", ""))
            if url and r.get(atc_col):
                products.setdefault(url, {"atc": r[atc_col], "drug_name": r[name_col],
                                          "product_name": r.get("product_name", r[name_col]),
                                          "source_url": url})
    for _, r in fallback.iterrows():
        url = _smpc_url_from(r.get("source_url", ""))
        if url and r.get("atc_code"):
            products.setdefault(url, {"atc": r["atc_code"], "drug_name": r["drug_name"],
                                      "product_name": r.get("product_name", r["drug_name"]),
                                      "source_url": url})

    print(f"Found {len(products)} unique products with a derivable SmPC URL.")

    if args.debug_urls:
        for url, info in list(products.items())[:10]:
            print(f"  {info['drug_name']:25s} {info['atc']:10s} -> {url}")
        print("\nOpen a couple of these in a browser and confirm they load a real SmPC page "
             "with numbered sections (4.1, 4.2, ...). If not, fix _smpc_url_from().")
        return

    already_done = set()
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE, dtype=str)
        already_done = set(zip(existing["atc_code"], existing["section"]))
        print(f"{len(already_done)} (atc, section) pairs already collected -- will skip those.")

    items = list(products.items())
    if args.limit:
        items = items[:args.limit]

    rows = []
    for i, (url, info) in enumerate(items):
        pending = wanted_sections - {sec for (atc, sec) in already_done if atc == info["atc"]}
        if not pending:
            continue
        print(f"[{i+1}/{len(items)}] {info['drug_name']} ({info['atc']}) -> {url}")
        html = fetch_html(url)
        if not html:
            continue
        sections = parse_sections(html)
        for sec_key in pending:
            text = sections.get(sec_key, "").strip()
            if not text:
                continue
            rows.append({
                "atc_code": info["atc"], "drug_name": info["drug_name"],
                "product_name": info["product_name"], "section": sec_key,
                "text": text[:3000], "source_url": url,
            })
            print(f"    + {sec_key}: {text[:80]}...")

        # Write incrementally every 20 products so a crash/interrupt doesn't lose progress.
        if rows and (i + 1) % 20 == 0:
            _append(rows)
            rows = []

    if rows:
        _append(rows)
    if not OUT_FILE.exists():
        print("\nNO ROWS WERE WRITTEN: every page either failed to download or no section "
              "could be parsed. Check the 'FAILED to fetch' lines above, and the HTML files in "
              f"{CACHE_DIR}.")
        return
    print(f"\nDone. Output: {OUT_FILE}")
    print("Next: python build_sections_db.py   (embeds this into ChromaDB)")


def _append(rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, mode="a", header=not OUT_FILE.exists(), index=False)


if __name__ == "__main__":
    main()