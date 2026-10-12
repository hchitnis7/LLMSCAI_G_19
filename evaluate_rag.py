"""Component-level evaluation of the RAG side of MediCheck (NO LLM, NO GPU needed).

Measures everything that happens BEFORE the language model writes a word:
  A. Drug recognition   (matcher.py)   - generic names, brand names, typos, Swedish/English, false positives
  B. Evidence retrieval (knowledge.py) - is the right pair found, is severity right, order-invariance, coverage
  C. Intent + topic detection          - small hand-labelled set
  D. FASS section coverage             - how much of the indication/side-effect data you actually have
  E. Ayesha's hybrid-search results    - Recall@k / MRR from her *_evaluation_results*.csv files (if present)

Run from the project folder (needs llm_ayesha/ with the CSVs + drugs_lookup.csv):
    python evaluate_rag.py
    python evaluate_rag.py --n 300 --backend chroma      # use the real ChromaDB instead of the CSVs
Writes: rag_metrics.json  and  rag_metrics.md  (paste-ready tables for the report).
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd

import matcher
from intents import detect_intents
from knowledge import SEVERITY_ORDER, CsvBackend, Knowledge, worst_severity
from topics import detect_topics

random.seed(7)


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def pstats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {"mean_ms": round(1000 * sum(xs) / len(xs), 1),
            "p50_ms": round(1000 * xs[len(xs) // 2], 1),
            "p95_ms": round(1000 * xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1)}


# ------------------------------------------------------------------ A. matcher
EN_T = ["Can I take {a} together with {b}?", "Is it safe to combine {a} and {b}?", "I use {a} and {b}"]
SV_T = ["Kan jag ta {a} tillsammans med {b}?", "Går det bra att kombinera {a} och {b}?", "Jag äter {a} och {b}"]
NEGATIVES = [
    "What is the weather like today?", "How do I reset my password?", "Tell me a joke",
    "Hur kommer jag till apoteket?", "Vad är klockan?", "I have a headache and feel tired",
    "Can you recommend a good book?", "Jag mår dåligt sedan i går", "What should I eat for dinner?",
    "Is it okay to exercise after lunch?", "Hej, hur mår du?", "I need help with my homework",
]


def typo(word):
    """One realistic OCR/typing error: drop, swap, or substitute a letter."""
    if len(word) < 6:
        return word
    i = random.randint(1, len(word) - 3)
    kind = random.choice(["drop", "swap", "sub"])
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + random.choice("aeiou") + word[i + 1:]


def detected_atcs(text):
    return {m["atc"] for m in matcher.match_drugs([text])}


def eval_matcher(pairs, n):
    out = {}
    names = dict(matcher._ATC_TO_NAME)            # atc -> canonical generic name
    atcs = list(names)
    sample = random.sample(atcs, min(n, len(atcs)))

    # single generic name, bare and inside a sentence
    hit = sum(1 for a in sample if a in detected_atcs(names[a]))
    hit_s = sum(1 for a in sample if a in detected_atcs(f"What is {names[a]} used for?"))
    out["generic_name_bare_recall_%"] = pct(hit, len(sample))
    out["generic_name_in_sentence_recall_%"] = pct(hit_s, len(sample))

    # brands
    brand_rows = [(b, a) for b, a in matcher._BRANDS.items() if a in names]
    bs = random.sample(brand_rows, min(n, len(brand_rows)))
    bh = sum(1 for b, a in bs if a in detected_atcs(f"Can I take {b} with alcohol?"))
    out["brand_name_recall_%"] = pct(bh, len(bs))
    out["n_brand_aliases"] = len(brand_rows)

    # typos (names >= 6 letters only; shorter words are not fuzz-matched by design)
    long_ = [a for a in sample if len(names[a]) >= 6]
    th = sum(1 for a in long_ if a in detected_atcs(typo(names[a])))
    out["typo_recall_%"] = pct(th, len(long_))

    # two drugs in one sentence, EN + SV, generic + brand mixed
    pr = random.sample(pairs, min(n, len(pairs)))
    both = both_sv = 0
    for a, b in pr:
        na, nb = names.get(a), names.get(b)
        if not na or not nb:
            continue
        found = detected_atcs(random.choice(EN_T).format(a=na, b=nb))
        both += {a, b} <= found
        found = detected_atcs(random.choice(SV_T).format(a=na, b=nb))
        both_sv += {a, b} <= found
    out["pair_both_found_en_%"] = pct(both, len(pr))
    out["pair_both_found_sv_%"] = pct(both_sv, len(pr))

    # false positives on drug-free text
    fp_any = sum(1 for t in NEGATIVES if matcher.match_drugs([t]))
    fp_conf = sum(1 for t in NEGATIVES if any(m["confident"] for m in matcher.match_drugs([t])))
    out["false_positive_confident_%_on_drug_free_text"] = pct(fp_conf, len(NEGATIVES))
    out["false_positive_any_(incl. 'did you mean?')_%"] = pct(fp_any, len(NEGATIVES))
    out["n_negative_sentences"] = len(NEGATIVES)
    return out


# ------------------------------------------------------------------ B. retrieval
def _tokens(t):
    return set(re.findall(r"\w{4,}", t.lower()))


def eval_retrieval(kn, pair_df, n, all_atcs):
    out = {}
    gold = {}
    for _, r in pair_df.iterrows():
        key = frozenset((r.drug_1_atc, r.drug_2_atc))
        sev = str(r.severity).upper()
        if key not in gold or SEVERITY_ORDER.get(sev, 0) > SEVERITY_ORDER.get(gold[key]["sev"], 0):
            gold[key] = {"sev": sev, "text": r.chunk_text, "a": (r.drug_1_name, r.drug_1_atc),
                         "b": (r.drug_2_name, r.drug_2_atc)}
    keys = random.sample(list(gold), min(n, len(gold)))
    found = sev_ok = content_ok = sym_ok = 0
    sev_conf = {}
    lat = []
    for k in keys:
        g = gold[k]
        a = {"name": g["a"][0], "atc": g["a"][1]}
        b = {"name": g["b"][0], "atc": g["b"][1]}
        kn.reset_ids()
        t0 = time.time()
        ev = kn.pair_evidence(a, b)
        lat.append(time.time() - t0)
        kn.reset_ids()
        ev_rev = kn.pair_evidence(b, a)
        pair_ev = [e for e in ev if e["kind"] == "pair"]
        if pair_ev:
            found += 1
            got = worst_severity(ev)
            sev_ok += got == g["sev"]
            sev_conf[(g["sev"], got)] = sev_conf.get((g["sev"], got), 0) + 1
            gt = _tokens(g["text"])
            content_ok += any(len(_tokens(e["text"]) & gt) / max(1, len(_tokens(e["text"]))) >= 0.6 for e in pair_ev)
        sym_ok += worst_severity(ev) == worst_severity(ev_rev)
    out["n_known_pairs_tested"] = len(keys)
    out["pair_found_%"] = pct(found, len(keys))
    out["severity_correct_%_of_found"] = pct(sev_ok, found)
    out["gold_sentence_retrieved_%_of_found"] = pct(content_ok, found)
    out["order_invariant_severity_%"] = pct(sym_ok, len(keys))
    out["severity_confusion(gold->got)"] = {f"{a}->{b}": c for (a, b), c in sorted(sev_conf.items())}
    out["latency"] = pstats(lat)

    # coverage on RANDOM drug pairs (what a real user would ask about)
    names = matcher._ATC_TO_NAME
    pool = [a for a in all_atcs if a in names]
    kinds = {"pair": 0, "statement": 0, "none": 0}
    m = min(n, 300)
    for _ in range(m):
        x, y = random.sample(pool, 2)
        kn.reset_ids()
        ev = kn.pair_evidence({"name": names[x], "atc": x}, {"name": names[y], "atc": y})
        kinds["pair" if any(e["kind"] == "pair" for e in ev) else
              "statement" if ev else "none"] += 1
    out["random_pair_coverage"] = {k: pct(v, m) for k, v in kinds.items()}
    out["random_pair_coverage_note"] = ("'none' means the bot must say 'no data' - expected for most random "
                                         "pairs; the safety property is that it never invents a severity.")
    return out


# ------------------------------------------------------------------ C. intents/topics
INTENT_SET = [
    ("Can I take Waran with Ipren?", 2, {"interaction"}, set()),
    ("Kan jag ta Alvedon tillsammans med Waran?", 2, {"interaction"}, set()),
    ("What is Alvedon used for?", 1, {"indication"}, set()),
    ("Vad används Waran till?", 1, {"indication"}, set()),
    ("What are the side effects of ibuprofen?", 1, {"side_effects"}, set()),
    ("Vilka biverkningar har Ipren?", 1, {"side_effects"}, set()),
    ("Is it safe during pregnancy?", 1, {"pregnancy"}, set()),
    ("Kan jag ta detta när jag är gravid?", 1, {"pregnancy"}, set()),
    ("Can I drive after taking this?", 1, {"driving"}, set()),
    ("Får jag köra bil efter att jag tagit detta?", 1, {"driving"}, set()),
    ("Can I drink alcohol with Alvedon?", 1, {"topic"}, {"alcohol"}),
    ("Kan jag dricka alkohol med Waran?", 1, {"topic"}, {"alcohol"}),
    ("Is grapefruit juice a problem with this?", 1, {"topic"}, {"grapefruit"}),
    ("Can I have coffee with this medicine?", 1, {"topic"}, {"caffeine"}),
    ("Does St John's wort interact with Waran?", 1, {"topic"}, {"st_johns_wort"}),
    ("What if I smoke while taking this?", 1, {"topic"}, {"smoking"}),
    ("Tell me about Waran", 1, {"overview"}, set()),
    ("Waran och Ipren", 2, {"interaction"}, set()),
    ("Who are contraindicated for ibuprofen?", 1, {"contraindication"}, set()),
    ("What is the weather today?", 0, {"overview"}, set()),
]


def eval_intents():
    ok_int = ok_top = 0
    wrong = []
    for text, nd, want_i, want_t in INTENT_SET:
        t = set(detect_topics(text))
        i = set(detect_intents(text, nd, list(t)))
        good_i = want_i <= i
        good_t = bool(want_t <= t and (want_t or not t))
        ok_int += good_i
        ok_top += good_t
        if not (good_i and good_t):
            wrong.append({"q": text, "intents": sorted(i), "topics": sorted(t)})
    n = len(INTENT_SET)
    return {"n": n, "intent_recall_%": pct(ok_int, n), "topic_exact_%": pct(ok_top, n), "misses": wrong}


# ------------------------------------------------------------------ D. section coverage
def eval_sections(data_dir, all_atcs):
    f = data_dir / "fass_sections.csv"
    if not f.exists():
        return {"note": "fass_sections.csv not found - run collect_sections.py. Indication/side-effect "
                        "questions will be answered with an honest 'no data'."}
    d = pd.read_csv(f, dtype=str).fillna("")
    total = len(set(all_atcs))
    out = {"rows": len(d), "drugs_with_any_section_%": pct(d.atc_code.nunique(), total)}
    for sec, g in d.groupby("section"):
        out[f"{sec}_coverage_%_of_drugs"] = pct(g.atc_code.nunique(), total)
    out["median_chars_per_section"] = int(d.text.str.len().median())
    return out


# ------------------------------------------------------------------ E. Ayesha's results
def eval_ayesha(data_dir):
    out = {}
    for name in ["retrieval_evaluation_results_exact_hybrid.csv", "fallback_evaluation_results_final.csv"]:
        f = data_dir / name
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip().lstrip("﻿") for c in d.columns]
        found = d["found"].astype(str).str.lower().isin(["true", "1", "yes"])
        rank = pd.to_numeric(d["rank"], errors="coerce")
        res = {"n": len(d), "found_%": pct(int(found.sum()), len(d))}
        for k in (1, 3, 5):
            res[f"recall@{k}_%"] = pct(int(((rank <= k) & found).sum()), len(d))
        mrr = (1.0 / rank[found & rank.notna() & (rank > 0)]).sum() / len(d)
        res["MRR"] = round(float(mrr), 3)
        out[name] = res
    return out or {"note": "no hybrid-search result CSVs found in data_dir"}


# ------------------------------------------------------------------ main
def md_table(title, d):
    lines = [f"### {title}", "", "| metric | value |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="llm_ayesha")
    ap.add_argument("--n", type=int, default=200, help="samples per test")
    ap.add_argument("--backend", choices=["csv", "chroma", "auto"], default="csv",
                    help="csv = read chunk CSVs directly (fast, deterministic); chroma = your real DB")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    pair_df = pd.read_csv(data_dir / "fass_pair_chunks.csv", dtype=str).fillna("")
    fb_df = pd.read_csv(data_dir / "fass_fallback_chunks.csv", dtype=str).fillna("")
    all_atcs = set(pair_df.drug_1_atc) | set(pair_df.drug_2_atc) | set(fb_df.atc_code)
    pairs = list({(r.drug_1_atc, r.drug_2_atc) for r in pair_df.itertuples()})

    backend = CsvBackend(data_dir) if args.backend == "csv" else None
    kn = Knowledge(backend=backend)
    print(f"Retrieval backend: {kn.b.name}")

    results = {"backend": kn.b.name, "n": args.n}
    print("A. matcher ...");   results["A_matcher"] = eval_matcher(pairs, args.n)
    print("B. retrieval ..."); results["B_retrieval"] = eval_retrieval(kn, pair_df, args.n, all_atcs)
    print("C. intents ...");   results["C_intents_topics"] = eval_intents()
    print("D. sections ...");  results["D_section_coverage"] = eval_sections(data_dir, all_atcs)
    print("E. hybrid search results ..."); results["E_ayesha_hybrid_search"] = eval_ayesha(data_dir)

    Path("rag_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    md = "# RAG component evaluation\n\n" + "\n".join(
        md_table(k, v) for k, v in results.items() if isinstance(v, dict) and k[0] in "ABCDE")
    Path("rag_metrics.md").write_text(md)
    print("\n" + md)
    print("Saved rag_metrics.json and rag_metrics.md")


if __name__ == "__main__":
    main()
