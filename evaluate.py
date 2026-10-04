"""Compare base model vs fine-tuned adapter on sft_data/test.jsonl.

Usage:
    python evaluate.py --adapter none        # baseline (no adapter)
    python evaluate.py --adapter adapter     # your trained adapter
Run both and compare the printed numbers -- that comparison is your report result.
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import SYSTEM_PROMPT, detect_language_name

VALID_LABELS = {"DANGEROUS", "AVOID", "CAUTION", "INFO"}


def load_model(model_id, adapter):
    # Matches chatbot.py: full bf16 inference, no quantization (96GB card).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LEFT padding is required for batched generation with a decoder-only
    # model: generation continues from the right edge of each sequence, so
    # all real tokens must end at the same position across the batch. With
    # left padding, every row's prompt ends at the same index, so a single
    # inputs.input_ids.shape[1] slice (used below) correctly strips the
    # prompt from every row in the batch at once.
    tok.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("Using Flash Attention 2.")
    except Exception as e:
        print(f"Flash Attention 2 unavailable ({type(e).__name__}), using default attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )

    if adapter and adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def _build_prompt(question, chunks):
    # Must mirror chatbot.py's _generate() exactly -- including the explicit
    # language instruction -- or these eval numbers don't reflect what's
    # actually served in production.
    ctx = ("\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
          if chunks else "(no matching FASS source found)")
    lang_name = detect_language_name(question)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}\n\n"
                                        f"Respond in {lang_name}, since that is the "
                                        f"language of the question above."}]


def generate_batch(tok, model, questions, chunks_list, max_new_tokens=200):
    """Generates for a whole batch of (question, chunks) pairs in one GPU call
    instead of one sequence at a time -- the main eval speedup, since a 96GB
    (or even 24GB) card has far more throughput available than a batch size
    of 1 uses. Returns a list of decoded answer strings, same order as input."""
    prompts = [
        tok.apply_chat_template(_build_prompt(q, c), tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for q, c in zip(questions, chunks_list)
    ]
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    # Left-padding means every row's prompt ends at the same column, so this
    # single slice correctly strips the prompt from every row at once.
    gen_only = out[:, inputs.input_ids.shape[1]:]
    return [tok.decode(row, skip_special_tokens=True) for row in gen_only]


def extract_label(text):
    text = text.strip().upper()
    for label in VALID_LABELS:
        if text.startswith(label):
            return label
    m = re.search(r"\b(DANGEROUS|AVOID|CAUTION|INFO)\b", text)
    return m.group(1) if m else None


def gold_label(answer):
    return extract_label(answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--test_file", default="sft_data/test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap #examples for a quick run")
    ap.add_argument("--batch_size", type=int, default=8,
                     help="Examples generated per GPU call. Raise if you have VRAM to "
                          "spare (96GB card can likely go higher); lower to 2-4 if you "
                          "hit OOM, e.g. on a 24GB card with a large model.")
    ap.add_argument("--max_new_tokens", type=int, default=200,
                     help="Lowered from 300 -- real answers here are short "
                          "(severity line + a few sentences + disclaimer); "
                          "fewer tokens requested means less wasted generation time.")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.adapter)
    examples = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    if args.limit:
        examples = examples[:args.limit]

    n = len(examples)
    correct_label, valid_format, said_unsafe_safe = 0, 0, 0
    has_citation_when_chunks, wrong_language = 0, 0
    rows = []

    for start in range(0, n, args.batch_size):
        batch = examples[start:start + args.batch_size]
        preds = generate_batch(
            tok, model,
            [ex["question"] for ex in batch],
            [ex["chunks"] for ex in batch],
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  {min(start + len(batch), n)}/{n}", end="\r")

        for ex, pred in zip(batch, preds):
            p_label = extract_label(pred)
            g_label = gold_label(ex["answer"])

            if p_label == g_label:
                correct_label += 1
            if p_label is not None:
                valid_format += 1
            # Word-boundary match, not a plain substring check -- "safe" as a
            # plain substring also matches inside "safety"/"säkerhet", and
            # our own correct template answers legitimately contain phrases
            # like "NOT a safety confirmation", which were being miscounted
            # as violations. \bsafe\b only matches the standalone word.
            if re.search(r"\bsafe\b", pred, re.I) or re.search(r"\bsäker\b", pred, re.I):
                said_unsafe_safe += 1
            # Citation check: the original bracket-only check undercounted --
            # the model often names the source in prose ("According to
            # FASS...", "Enligt FASS...") without bracket syntax. Count
            # either form as a citation.
            if ex["chunks"] and (("[" in pred and "]" in pred) or "fass" in pred.lower()):
                has_citation_when_chunks += 1
            # Rough language check: question in English but answer heavy on
            # Swedish-specific characters is a likely language mismatch.
            # Crude on purpose -- good enough to flag the pattern for manual
            # review, not meant to be a precise classifier.
            q_is_english = not re.search(r"[åäöÅÄÖ]", ex["question"])
            a_looks_swedish = len(re.findall(r"[åäöÅÄÖ]", pred)) >= 3
            if q_is_english and a_looks_swedish:
                wrong_language += 1
            rows.append({"question": ex["question"], "gold": g_label, "pred": p_label,
                        "answer": pred[:200]})
    print()

    print(f"\n=== Results: model={args.model} adapter={args.adapter} n={n} ===")
    print(f"Severity accuracy:      {correct_label}/{n} = {correct_label/n:.1%}")
    print(f"Valid format (has label):{valid_format}/{n} = {valid_format/n:.1%}")
    print(f"Said 'safe' (should be 0):{said_unsafe_safe}/{n}")
    chunked = sum(1 for e in examples if e["chunks"])
    if chunked:
        print(f"Cited source when evidence given: {has_citation_when_chunks}/{chunked}")
    n_english_q = sum(1 for e in examples if not re.search(r"[åäöÅÄÖ]", e["question"]))
    if n_english_q:
        print(f"English question answered in Swedish (likely mismatch): "
             f"{wrong_language}/{n_english_q}")

    out_path = f"eval_{args.adapter.replace('/', '_')}.json"
    json.dump(rows, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Per-example predictions saved to {out_path}")


if __name__ == "__main__":
    main()