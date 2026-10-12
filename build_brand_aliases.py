"""Build llm_ayesha/brand_aliases.csv  (brand name -> ATC code).

Why: drugs_lookup.csv only contains substance names (warfarin, ibuprofen...), but people
type and photograph BRAND names (Waran, Ipren, Alvedon...). matcher.py loads this CSV
(plus a built-in seed list) so brands resolve to the right substance with score 100.

Sources, merged in this order (later ones never overwrite earlier ones):
  1. --master medicines_master.csv   (your full FASS product list: product_name, atc_code)
     -> the best source: every product FASS knows, so ANY brand on a package resolves.
  2. llm_ayesha/fass_pair_chunks.csv and fass_fallback_chunks.csv (product_name + ATC)
  3. the seed list in matcher.py (BRAND_SEED_ATC)

Run:
    python build_brand_aliases.py --master path/to/medicines_master.csv
    python build_brand_aliases.py            # without master: chunk CSVs only
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DATA = Path("/teamspace/studios/this_studio/LLMSCAI_G_19/llm_ayesha")
OUT = DATA / "brand_aliases.csv"

# words that are NOT a brand (generic product names such as "Ibuprofen Orifarm 400 mg")
_STRENGTH = re.compile(r"\d")


def brand_key(product_name: str) -> str:
    """'Alvedon 500 mg filmdragerade tabletter' -> 'alvedon'. First token, letters only."""
    first = re.split(r"[\s/,(]", str(product_name).strip(), maxsplit=1)[0]
    first = re.sub(r"[^\wåäöÅÄÖ-]", "", first).lower()
    return "" if _STRENGTH.search(first) else first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", help="medicines_master.csv with product_name + atc_code")
    args = ap.parse_args()

    lookup = pd.read_csv(DATA / "drugs_lookup.csv")
    substance_names = set(lookup["name"].str.lower())
    known_atc = set(lookup["atc"].str.upper())

    pairs = []
    if args.master:
        m = pd.read_csv(args.master, dtype=str)
        pairs += list(zip(m["product_name"], m["atc_code"]))
    p = pd.read_csv(DATA / "fass_pair_chunks.csv", dtype=str)
    pairs += list(zip(p["product_name"], p["drug_1_atc"]))
    f = pd.read_csv(DATA / "fass_fallback_chunks.csv", dtype=str)
    pairs += list(zip(f["product_name"], f["atc_code"]))

    rows, seen = [], {}
    for product, atc in pairs:
        if not isinstance(product, str) or not isinstance(atc, str):
            continue
        atc = atc.strip().upper()
        b = brand_key(product)
        if len(b) < 4 or b in substance_names or atc not in known_atc:
            continue
        # a brand that maps to several ATC codes is ambiguous -> keep the first, flag it
        if b in seen and seen[b] != atc:
            continue
        if b not in seen:
            seen[b] = atc
            rows.append({"brand": b, "atc": atc, "example_product": product})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} brand aliases to {OUT}")


if __name__ == "__main__":
    main()
