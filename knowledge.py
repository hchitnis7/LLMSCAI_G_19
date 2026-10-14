"""Knowledge layer: everything the chatbot is allowed to know, as evidence records.

Two interchangeable backends, same methods:
  * ChromaBackend -- your persisted ChromaDB (production; collection 'medication_interactions'
                     plus optional 'fass_sections' for indications/side effects/pregnancy...).
  * CsvBackend    -- reads the chunk CSVs directly (llm_ayesha/*.csv). Used automatically if
                     ChromaDB can't be opened, and for tests.

Evidence record (what the LLM is shown, and what citations are built from):
    {"id": "E3", "kind": "pair|statement|section|atc", "atcs": [...], "drugs": [...],
     "severity": "AVOID"|None, "section": "4.5", "label": "FASS SmPC 4.5",
     "source_url": "...", "text": "..."}

Design note: severity is NEVER produced by the language model. It comes from the data
(pair chunk metadata) via worst_severity(), so a fluent answer can't downplay a warning.
"""
import os
import re
from pathlib import Path

import pandas as pd

from atc_groups import atc_group_text

_HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MEDICHECK_DATA_DIR", _HERE / "llm_ayesha"))
DB_PATH = os.environ.get("DRUG_DB_PATH", str(DATA_DIR / "drug_db"))
SEVERITY_ORDER = {"DANGEROUS": 4, "AVOID": 3, "CAUTION": 2, "INFO": 1, "UNCLASSIFIED": 1}

# FASS SmPC sections the chatbot can answer from, when collected (collect_sections.py).
SECTION_LABELS = {
    "indications": "FASS SmPC 4.1 (indications)",
    "dosage": "FASS SmPC 4.2",
    "contraindications": "FASS SmPC 4.3 (contraindications)",
    "warnings": "FASS SmPC 4.4 (warnings)",
    "interactions": "FASS SmPC 4.5 (interactions)",
    "pregnancy_breastfeeding": "FASS SmPC 4.6 (pregnancy and breastfeeding)",
    "driving": "FASS SmPC 4.7 (driving and machines)",
    "side_effects": "FASS SmPC 4.8 (side effects)",
    "overdose": "FASS SmPC 4.9 (overdose)",
}
MAX_CHARS = 900


def worst_severity(evidence):
    best, rank = None, 0
    for e in evidence:
        r = SEVERITY_ORDER.get((e.get("severity") or "").upper(), 0)
        if r > rank:
            best, rank = (e["severity"] or "").upper(), r
    return "INFO" if best == "UNCLASSIFIED" else best


def _interaction_text(chunk_text: str) -> str:
    """The actual interaction sentence(s) from a chunk ('Interaction:' / 'Interaction statement:')."""
    out = ""
    for line in str(chunk_text).splitlines():
        for key in ("Interaction statement:", "Interaction:"):
            if line.startswith(key):
                out = line[len(key):].strip()
    return out or str(chunk_text).strip()


def _trim(text: str, n: int = MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > n // 2 else cut.rsplit(" ", 1)[0]) + " ..."


def _terms_in(text: str, terms) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


def _best_sentences(text: str, terms, max_sentences: int = 3) -> str:
    """Only the sentences mentioning the topic (keeps evidence short and on-point)."""
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    hits = [p for p in parts if _terms_in(p, terms)]
    return " ".join(hits[:max_sentences]) if hits else ""


# ====================================================================== backends
class CsvBackend:
    name = "csv"

    def __init__(self, data_dir: Path = DATA_DIR):
        self.pairs = pd.read_csv(data_dir / "fass_pair_chunks.csv", dtype=str).fillna("")
        self.stmts = pd.read_csv(data_dir / "fass_fallback_chunks.csv", dtype=str).fillna("")
        sec = data_dir / "fass_sections.csv"
        self.sections_df = pd.read_csv(sec, dtype=str).fillna("") if sec.exists() else None

    # rows are returned as plain dicts: {text, severity, drug_1_atc, drug_1_name, drug_2_atc, drug_2_name, source_url}
    def pair_rows(self, atc_a, atc_b):
        p = self.pairs
        m = ((p.drug_1_atc == atc_a) & (p.drug_2_atc == atc_b)) | ((p.drug_1_atc == atc_b) & (p.drug_2_atc == atc_a))
        return [self._pair(r) for _, r in p[m].iterrows()]

    def pair_rows_for(self, atc):
        p = self.pairs
        return [self._pair(r) for _, r in p[(p.drug_1_atc == atc) | (p.drug_2_atc == atc)].iterrows()]

    def statement_rows(self, atc):
        s = self.stmts
        return [self._stmt(r) for _, r in s[s.atc_code == atc].iterrows()]

    def section_rows(self, atc, section):
        if self.sections_df is None:
            return []
        d = self.sections_df
        return [{"text": r["text"], "drug_name": r.get("drug_name", ""), "source_url": r.get("source_url", ""),
                 "product_name": r.get("product_name", "")}
                for _, r in d[(d.atc_code == atc) & (d.section == section)].iterrows()]

    @staticmethod
    def _pair(r):
        return {"text": r["chunk_text"], "severity": r["severity"] or "INFO", "drug_1_atc": r["drug_1_atc"],
                "drug_1_name": r["drug_1_name"], "drug_2_atc": r["drug_2_atc"],
                "drug_2_name": r["drug_2_name"], "source_url": r["source_url"]}

    @staticmethod
    def _stmt(r):
        return {"text": r["chunk_text"], "severity": None, "drug_1_atc": r["atc_code"],
                "drug_1_name": r["drug_name"], "source_url": r["source_url"]}


class ChromaBackend:
    name = "chroma"

    def __init__(self, path: str = DB_PATH):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_collection("medication_interactions")
        try:
            self.sec_col = self.client.get_collection("fass_sections")
        except Exception:
            self.sec_col = None

    def _get(self, where, col=None):
        r = (col or self.col).get(where=where, include=["documents", "metadatas"])
        return list(zip(r["documents"], r["metadatas"]))

    def pair_rows(self, atc_a, atc_b):
        rows = self._get({"$or": [
            {"$and": [{"drug_1_atc": {"$eq": atc_a}}, {"drug_2_atc": {"$eq": atc_b}}, {"chunk_type": {"$eq": "interaction_pair"}}]},
            {"$and": [{"drug_1_atc": {"$eq": atc_b}}, {"drug_2_atc": {"$eq": atc_a}}, {"chunk_type": {"$eq": "interaction_pair"}}]},
        ]})
        return [self._row(d, m) for d, m in rows]

    def pair_rows_for(self, atc):
        rows = self._get({"$and": [{"chunk_type": {"$eq": "interaction_pair"}},
                                   {"$or": [{"drug_1_atc": {"$eq": atc}}, {"drug_2_atc": {"$eq": atc}}]}]})
        return [self._row(d, m) for d, m in rows]

    def statement_rows(self, atc):
        rows = self._get({"$and": [{"chunk_type": {"$eq": "interaction_statement"}}, {"drug_1_atc": {"$eq": atc}}]})
        return [dict(self._row(d, m), severity=None) for d, m in rows]

    def section_rows(self, atc, section):
        if self.sec_col is None:
            return []
        rows = self._get({"$and": [{"atc_code": {"$eq": atc}}, {"section": {"$eq": section}}]}, self.sec_col)
        return [{"text": d, "drug_name": m.get("drug_name", ""), "source_url": m.get("source_url", ""),
                 "product_name": m.get("product_name", "")} for d, m in rows]

    @staticmethod
    def _row(d, m):
        return {"text": d, "severity": m.get("severity") or "INFO", "drug_1_atc": m.get("drug_1_atc", ""),
                "drug_1_name": m.get("drug_1_name", ""), "drug_2_atc": m.get("drug_2_atc", ""),
                "drug_2_name": m.get("drug_2_name", ""), "source_url": m.get("source_url", "")}


# ====================================================================== facade
class Knowledge:
    """Retrieval facade used by the chatbot. `names` maps ATC -> display name."""

    def __init__(self, backend=None, embedder=None):
        if backend is None:
            try:
                backend = ChromaBackend()
                backend.col.count()
            except Exception as e:   # LFS pointer, missing DB, version mismatch...
                print(f"[knowledge] ChromaDB unavailable ({type(e).__name__}); using CSV chunks instead.")
                backend = CsvBackend()
        self.b = backend
        self.embedder = embedder    # optional: fn(list[str]) -> list[list[float]]; improves ranking only
        self._n = 0

    # -- helpers
    def _ev(self, kind, text, atcs, drugs, severity=None, section="4.5", label=None, url=""):
        self._n += 1
        return {"id": f"E{self._n}", "kind": kind, "atcs": atcs, "drugs": drugs, "severity": severity,
                "section": section, "label": label or "FASS SmPC 4.5 (interactions)",
                "source_url": url, "text": _trim(text)}

    def reset_ids(self):
        self._n = 0

    def _rank(self, query, rows, key="text", k=4):
        """Order rows by relevance to `query` (embeddings if available, else word overlap)."""
        if len(rows) <= 1:
            return rows[:k]
        if self.embedder:
            try:
                import numpy as np
                vecs = np.array(self.embedder([query] + [r[key][:600] for r in rows]))
                sims = vecs[1:] @ vecs[0] / (np.linalg.norm(vecs[1:], axis=1) * np.linalg.norm(vecs[0]) + 1e-9)
                order = sims.argsort()[::-1]
                return [rows[i] for i in order[:k]]
            except Exception:
                pass
        q = set(re.findall(r"\w{4,}", query.lower()))
        scored = sorted(rows, key=lambda r: -len(q & set(re.findall(r"\w{4,}", r[key].lower()))))
        return scored[:k]

    # -- interaction between two named drugs
    def pair_evidence(self, a, b):
        """a, b: {'name','atc'}. Structured pair chunks first; otherwise statements that
        mention the other drug by name (bidirectional)."""
        rows = self.b.pair_rows(a["atc"], b["atc"])
        # Worst severity first: only the top 3 rows are shown, and severity is
        # taken from what is shown -- never let a DANGEROUS row fall off the list.
        rows = sorted(rows, key=lambda r: -SEVERITY_ORDER.get(str(r["severity"]).upper(), 0))
        if rows:
            return [self._ev("pair", _interaction_text(r["text"]), [a["atc"], b["atc"]],
                             [a["name"], b["name"]], severity=r["severity"], url=r["source_url"]) for r in rows[:3]]
        out = []
        for x, y in ((a, b), (b, a)):
            names = {y["name"].lower()} | {n.lower() for n in y.get("aliases", [])}
            names = {n for n in names if len(n) >= 4}
            hits = [r for r in self.b.statement_rows(x["atc"]) if _terms_in(r["text"], names)]
            for r in self._rank(f"{x['name']} {y['name']}", hits, k=2):
                out.append(self._ev("statement", _interaction_text(r["text"]), [a["atc"], b["atc"]],
                                    [x["name"], y["name"]], severity=None, url=r["source_url"]))
        if not out:
            # last resort: the full SmPC 4.5 (interactions) text of either drug, if scraped, naming the other drug
            for x, y in ((a, b), (b, a)):
                names = {y["name"].lower()} | {n.lower() for n in y.get("aliases", [])}
                names = {n for n in names if len(n) >= 4}
                try:
                    sec_rows = self.b.section_rows(x["atc"], "interactions")
                except Exception:
                    sec_rows = []
                for r in sec_rows:
                    if _terms_in(r["text"], names):
                        out.append(self._ev("section", _best_sentences(r["text"], names), [a["atc"], b["atc"]],
                                            [x["name"], y["name"]], severity=None, section="4.5",
                                            label="FASS SmPC 4.5 (interactions)", url=r.get("source_url", "")))
                        break
        return out

    def has_interaction_data(self, drug):
        """True if we hold ANY interaction text for this drug (so 'no mention of the other drug' is a
        meaningful finding, not just missing data)."""
        atc = drug["atc"]
        if self.b.pair_rows_for(atc) or self.b.statement_rows(atc):
            return True
        try:
            return bool(self.b.section_rows(atc, "interactions"))
        except Exception:
            return False

    # -- non-drug interactant (alcohol, grapefruit, St John's wort, caffeine, ...)
    def topic_evidence(self, drug, terms, label_topic):
        rows = self.b.statement_rows(drug["atc"]) + self.b.pair_rows_for(drug["atc"])
        out, seen = [], set()
        for r in rows:
            text = _interaction_text(r["text"])
            snippet = _best_sentences(text, terms)
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            other = r.get("drug_2_name") if r.get("drug_1_atc") == drug["atc"] else r.get("drug_1_name")
            out.append(self._ev("statement", snippet, [drug["atc"]], [drug["name"]],
                                severity=(r.get("severity") if r.get("severity") else None), url=r.get("source_url", "")))
            if len(out) >= 4:
                break
        return out

    # -- single-drug interaction overview (nothing specific asked)
    def overview_evidence(self, drug, query, k=4):
        rows = self.b.pair_rows_for(drug["atc"])
        rows.sort(key=lambda r: -SEVERITY_ORDER.get((r["severity"] or "").upper(), 0))
        out = [self._ev("pair", _interaction_text(r["text"]), [r["drug_1_atc"], r["drug_2_atc"]],
                        [r["drug_1_name"], r["drug_2_name"]], severity=r["severity"], url=r["source_url"])
               for r in rows[:2]]
        stm = self._rank(query, self.b.statement_rows(drug["atc"]), k=k - len(out))
        out += [self._ev("statement", _interaction_text(r["text"]), [drug["atc"]], [drug["name"]], url=r["source_url"])
                for r in stm]
        return out

    # -- other FASS sections (needs collect_sections.py + build_sections_db.py output)
    def section_evidence(self, drug, section, query, k=3):
        rows = self.b.section_rows(drug["atc"], section)
        rows = self._rank(query, rows, k=k)
        label = SECTION_LABELS[section]
        num = re.search(r"4\.\d", label).group(0)
        return [self._ev("section", r["text"], [drug["atc"]], [drug["name"]], section=num, label=label,
                         url=r.get("source_url", "")) for r in rows]

    def atc_evidence(self, drug):
        g = atc_group_text(drug["atc"])
        if not g:
            return []
        return [self._ev("atc", f"{drug['name']} has ATC code {drug['atc']}, which falls under {g}. "
                                f"This is a classification of the medicine's group, not a list of approved uses.",
                         [drug["atc"]], [drug["name"]], section="ATC", label="WHO ATC classification (from project lookup)")]