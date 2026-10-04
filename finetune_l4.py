"""QLoRA fine-tuning sized for a single 24GB L4, as a fallback to finetune.py
(which targets Qwen3-30B-A3B for a 96GB RTX PRO 6000).

IMPORTANT: Qwen3-30B-A3B is a Mixture-of-Experts model. MoE only reduces
COMPUTE per token (~3B active), not VRAM -- ALL 30B parameters must still be
loaded regardless of which ones fire. In 4-bit that's still ~16-18GB of
weights alone, which leaves too little headroom on a 24GB card once you add
LoRA, optimizer state, activations, and KV cache. So this script uses a
DENSE model instead: Qwen3-14B, whose full 14B parameters in 4-bit is
~7-8GB -- comfortable on an L4 with real headroom.

The adapter this produces is tied to Qwen3-14B. If you later train on the
96GB box with finetune.py (Qwen3-30B-A3B), that produces a SEPARATE,
incompatible adapter -- don't mix them. Recommended: save this run's output
to adapter_l4/ instead of the default adapter/ name, and point chatbot.py /
evaluate.py at whichever base model + adapter pair you're actually serving.

Usage:
    python finetune_l4.py --smoke_test
    python finetune_l4.py
    python finetune_l4.py --epochs 1          # faster, if time is tight
"""
import argparse
import inspect

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def make_sft_config(**kwargs):
    """Builds SFTConfig using only the keyword arguments this installed TRL
    version actually accepts. TRL has renamed/dropped SFTConfig fields across
    releases (e.g. warmup_ratio, max_length/max_seq_length), so instead of
    guessing which names this version wants, we introspect its signature and
    silently drop anything it doesn't recognize, printing what was dropped."""
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = {k: v for k, v in kwargs.items() if k not in accepted}
    if dropped:
        print(f"NOTE: this TRL version's SFTConfig doesn't accept {list(dropped)}, "
              f"skipping them (defaults will be used instead).")
    return SFTConfig(**filtered)

from prompts import SYSTEM_PROMPT


def build_prompt_fn(tok):
    def to_example(ex):
        chunks = ex.get("chunks") or []
        if chunks:
            ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
        else:
            ctx = "(no matching FASS source found)"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {ex['question']}"},
        ]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return {"prompt": prompt, "completion": ex["answer"] + tok.eos_token}
    return to_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B",
                     help="Dense model that fits 24GB in 4-bit QLoRA. Drop to "
                          "Qwen/Qwen3-8B if you still hit OOM.")
    ap.add_argument("--data_dir", default="sft_data")
    ap.add_argument("--out_dir", default="adapter_l4")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_length", type=int, default=1024,
                     help="Lowered from 1536 -- real examples are much shorter than this, "
                          "and packing (see --no_packing) uses this as the packed block size.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2,
                     help="More conservative than the 96GB script, 24GB is tighter. "
                          "Try 4 if you have headroom; drop to 1 if you hit OOM.")
    ap.add_argument("--grad_accum", type=int, default=8,
                     help="batch_size * grad_accum = effective batch size (kept at 16).")
    ap.add_argument("--no_packing", action="store_true",
                     help="Disable sequence packing (on by default -- the main speedup, "
                          "since real examples are much shorter than max_length).")
    ap.add_argument("--max_train_examples", type=int, default=None,
                     help="Cap the training set size for a quick run, e.g. "
                          "--max_train_examples 2000, if time is the binding constraint.")
    ap.add_argument("--smoke_test", action="store_true",
                     help="Train on 20 examples for a few steps, to check everything runs.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data_files = {"train": f"{args.data_dir}/train.jsonl", "val": f"{args.data_dir}/val.jsonl"}
    data = load_dataset("json", data_files=data_files)

    if args.max_train_examples and len(data["train"]) > args.max_train_examples:
        print(f"Capping training set: {len(data['train'])} -> {args.max_train_examples} examples")
        data["train"] = data["train"].shuffle(seed=42).select(range(args.max_train_examples))

    if args.smoke_test:
        data["train"] = data["train"].select(range(min(20, len(data["train"]))))
        data["val"] = data["val"].select(range(min(10, len(data["val"]))))
        args.epochs = 1

    to_example = build_prompt_fn(tok)
    data = data.map(to_example, remove_columns=data["train"].column_names)

    print("=== Example formatted training prompt (sanity check) ===")
    print(data["train"][0]["prompt"][:800])
    print("--- completion ---")
    print(data["train"][0]["completion"][:300])
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_cfg, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("Using Flash Attention 2.")
    except Exception as e:
        print(f"Flash Attention 2 unavailable ({type(e).__name__}), falling back to default "
             f"attention. Install with: pip install flash-attn --no-build-isolation")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_cfg, device_map="auto",
        )

    # Unlike finetune.py's Qwen3-30B-A3B target (an MoE model whose expert FFN
    # layers are raw batched parameters PEFT's target_modules can't reach --
    # see the long comment in finetune.py), Qwen3-14B is DENSE: gate_proj/
    # up_proj/down_proj here ARE normal nn.Linear modules, confirmed reachable
    # (Together.ai's published default LoRA target list for Qwen/Qwen3-14B
    # includes all seven of these). So this script's adapter is actually MORE
    # completely fine-tuned than the 30B MoE one -- worth knowing when
    # deciding which result to report as your primary fine-tuning result.
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = make_sft_config(
        output_dir=args.out_dir + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,   # essential on 24GB -- trades speed for memory, keep ON
        max_length=args.max_length,
        max_seq_length=args.max_length,  # older TRL versions use this name instead
        packing=not args.no_packing,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        eval_strategy="steps",
        evaluation_strategy="steps",     # older TRL versions use this name instead
        eval_steps=50 if not args.smoke_test else 5,
        save_strategy="steps",
        save_steps=50 if not args.smoke_test else 5,
        save_total_limit=2,
        logging_steps=10 if not args.smoke_test else 1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=data["train"], eval_dataset=data["val"],
        peft_config=peft_cfg, processing_class=tok,
    )
    trainer.train()

    trainer.model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"\nAdapter saved to ./{args.out_dir}")
    print(f"This adapter is tied to base model {args.model} -- point chatbot.py's "
          f"MODEL_ID at the SAME model and ADAPTER_PATH at '{args.out_dir}' to serve it.")
    print("Next: python evaluate.py --model", args.model, "--adapter", args.out_dir)

    if args.smoke_test:
        print("\nSMOKE TEST DONE. If this looked fine (loss printed, no errors), "
              "rerun WITHOUT --smoke_test for the real training run.")
        print("If you hit OOM even on the smoke test, rerun with --model Qwen/Qwen3-8B.")


if __name__ == "__main__":
    main()