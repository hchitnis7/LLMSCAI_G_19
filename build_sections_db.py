"""Embeds llm_ayesha/fass_sections.csv (from collect_sections.py) into the
same ChromaDB used for interactions, as a SEPARATE collection "fass_sections"
-- this is exactly what knowledge.py's ChromaBackend already looks for
(self.sec_col = self.client.get_collection("fass_sections")), so once this
runs, Knowledge.section_evidence() (indications/side effects/pregnancy/
driving/etc.) starts working automatically, no other code changes needed.

Uses the same embedding model as the rest of the project
(paraphrase-multilingual-MiniLM-L12-v2) so section search is consistent with
interaction search.

Run once, after collect_sections.py has produced fass_sections.csv:
    python build_sections_db.py
Safe to re-run -- it replaces the collection's contents each time rather
than appending duplicates.
"""
import os
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.environ.get("MEDICHECK_DATA_DIR", "llm_ayesha"))
SECTIONS_CSV = DATA_DIR / "fass_sections.csv"
DB_PATH = os.environ.get("DRUG_DB_PATH", str(DATA_DIR / "drug_db"))
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def main():
    if not SECTIONS_CSV.exists():
        raise SystemExit(f"{SECTIONS_CSV} not found -- run collect_sections.py first.")

    df = pd.read_csv(SECTIONS_CSV, dtype=str).fillna("")
    df = df[df["text"].str.strip() != ""]
    print(f"Loaded {len(df)} section rows from {SECTIONS_CSV}")

    print(f"Loading embedding model {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("fass_sections")
        print("Replaced existing 'fass_sections' collection.")
    except Exception:
        pass
    col = client.create_collection("fass_sections")

    ids, documents, metadatas = [], [], []
    for i, r in df.iterrows():
        ids.append(f"section-{i}")
        documents.append(r["text"])
        metadatas.append({
            "atc_code": r["atc_code"], "drug_name": r["drug_name"],
            "product_name": r["product_name"], "section": r["section"],
            "source_url": r["source_url"],
        })

    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_docs = documents[start:end]
        embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        col.add(ids=ids[start:end], documents=batch_docs,
               metadatas=metadatas[start:end], embeddings=embeddings)
        print(f"  embedded {min(end, len(documents))}/{len(documents)}")

    print(f"\nDone. {col.count()} section chunks in ChromaDB collection 'fass_sections' at {DB_PATH}")
    print("knowledge.py's ChromaBackend will pick this up automatically on next run "
         "(Knowledge.section_evidence() needs no code changes).")


if __name__ == "__main__":
    main()
