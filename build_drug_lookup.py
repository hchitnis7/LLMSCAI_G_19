"""Build drugs_lookup.csv from Ayesha's chunk files.

Her CSVs embed drug names + ATC codes inside the pair/fallback rows, but
there's no standalone name->ATC table. The matcher needs one. This script
extracts every unique (name, ATC) pair it can find in both files.

Run once:
    python build_drug_lookup.py
Produces:
    drugs_lookup.csv   columns: name, atc
"""
import pandas as pd

PAIR_FILE = "LLMSCAI_G_19/llm_ayesha/fass_pair_chunks.csv"
FALLBACK_FILE = "LLMSCAI_G_19/llm_ayesha/fass_fallback_chunks.csv"
OUT_FILE = "LLMSCAI_G_19/llm_ayesha/drugs_lookup.csv"

pair = pd.read_csv(PAIR_FILE)
fallback = pd.read_csv(FALLBACK_FILE)

rows = []
rows += list(zip(pair["drug_1_name"], pair["drug_1_atc"]))
rows += list(zip(pair["drug_2_name"], pair["drug_2_atc"]))
rows += list(zip(fallback["drug_name"], fallback["atc_code"]))

df = pd.DataFrame(rows, columns=["name", "atc"]).dropna()
df["name"] = df["name"].str.strip()
df["atc"] = df["atc"].str.upper().str.strip()
df = df[(df["name"] != "") & (df["atc"] != "")]

# A name can map to more than one ATC in rare cases (combo products under
# different codes) -- keep the most frequent ATC per name.
counts = df.groupby(["name", "atc"]).size().reset_index(name="n")
best = counts.sort_values("n", ascending=False).drop_duplicates("name")
best = best[["name", "atc"]].sort_values("name").reset_index(drop=True)

best.to_csv(OUT_FILE, index=False)
print(f"{len(best)} unique drug names written to {OUT_FILE}")
print(best.head(10))
