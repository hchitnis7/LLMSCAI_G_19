"""QLoRA fine-tuning on the dataset produced by build_finetune_dataset.py.

Usage:
    python finetune.py                           # defaults below
    python finetune.py --model Qwen/Qwen3-4B --epochs 1   # faster/smaller
    python finetune.py --smoke_test                       # 20 examples, few steps, to check it runs

Before the real run: do a --smoke_test, confirm it completes and loss drops,
THEN kick off the full run. Don't skip this -- a broken run that fails at
step 400 wastes hours you don't have.
"""
import argparse
import inspect
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from prompts import SYSTEM_PROMPT


def make_sft_config(**kwargs):
    """Builds SFTConfig using only the keyword arguments this installed TRL
    version actually accepts -- field names have changed across TRL releases
    (e.g. warmup_ratio, max_length/max_seq_length, eval_strategy/
    evaluation_strategy). Anything unsupported is dropped with a printed note
    instead of crashing."""
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = {k: v for k, v in kwargs.items() if k not in accepted}
    if dropped:
        print(f"NOTE: this TRL version's SFTConfig doesn't accept {list(dropped)}, "
              f"skipping them (defaults will be used instead).")
    return SFTConfig(**filtered)


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
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--data_dir", default="sft_data")
    ap.add_argument("--out_dir", default="adapter")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_length", type=int, default=1024,
                     help="Lowered from 2048 -- your actual examples (short question + one "
                          "FASS chunk + a few sentences) are much shorter than that, so the "
                          "old default was mostly wasted padding compute.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4,
                     help="96GB has room for this; raises GPU utilization per step vs "
                          "the old batch_size=1. Lower if you hit OOM.")
    ap.add_argument("--grad_accum", type=int, default=4,
                     help="batch_size * grad_accum = effective batch size (kept at 16, "
                          "same as before, just spread differently for better throughput).")
    ap.add_argument("--no_packing", action="store_true",
                     help="Disable sequence packing (on by default -- packs multiple short "
                          "examples into each full-length sequence instead of padding each "
                          "one separately, which is the single biggest speedup here).")
    ap.add_argument("--no_grad_checkpointing", action="store_true",
                     help="Disable gradient checkpointing for extra speed (~1.3-1.7x) at the "
                          "cost of more VRAM. Try this on the 96GB card if --no_packing alone "
                          "still leaves VRAM to spare; leave checkpointing ON if unsure.")
    ap.add_argument("--max_train_examples", type=int, default=None,
                     help="Cap the training set size for a quick, lower-quality run when "
                          "time is the binding constraint, e.g. --max_train_examples 2000.")
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

    # Training still uses 4-bit QLoRA even on the 96GB card: inference only
    # holds weights + KV cache, but training also needs memory for gradients,
    # optimizer states, and activations -- QLoRA keeps that comfortable and
    # fast here. (Full bf16 fine-tuning of a 30B model would need several
    # times this card's memory.)
    #
    # IMPORTANT, confirmed directly (inspect_model_layers.py + HF PEFT docs +
    # Together.ai's published default LoRA target list for this exact model):
    # Qwen3-30B-A3B's MoE expert FFN layers (gate_up_proj / down_proj) are
    # implemented as raw batched nn.Parameter tensors, NOT nn.Linear modules.
    # PEFT's standard target_modules mechanism can only attach LoRA to named
    # nn.Linear submodules, so it CANNOT reach these expert weights no matter
    # what string you put here -- only q_proj/k_proj/v_proj/o_proj (attention)
    # are reachable this way. This matches Together.ai's own default target
    # list for Qwen3-30B-A3B-Instruct-2507: k_proj, o_proj, q_proj, v_proj.
    #
    # A newer PEFT mechanism (target_parameters, PEFT 0.18+) can reach the
    # fused expert parameters directly, but as of writing it has an open,
    # unresolved bug producing incorrect expert scaling when served via vLLM
    # (huggingface/peft#3874), and needs careful per-expert rank_pattern math
    # and dropout=0. Not used here given the project timeline.
    #
    # Practical effect: this adapter tunes attention behavior only (how the
    # model selects/weighs the retrieved RAG context and follows the output
    # format/refusal rules), not the model's underlying factual knowledge.
    # That's an acceptable scope for this project: answers are grounded by
    # RAG retrieval, not by facts baked into the weights, so the fine-tuning
    # job here is about output format/severity-label discipline/refusal
    # reliability -- an attention/instruction-following concern -- rather
    # than knowledge injection. State this explicitly in the report rather
    # than treating it as a shortfall.
    # tf32 matmuls: free speedup on Ampere/Ada/Hopper (A100/L4/RTX6000 etc),
    # no accuracy-relevant effect for training at this precision.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # Flash Attention 2 is a large speedup over the default attention kernel,
    # especially with packing (longer effective sequences per step). Falls
    # back automatically if the flash-attn package isn't installed --
    # `pip install flash-attn --no-build-isolation` to get it.
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

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
        gradient_checkpointing=not args.no_grad_checkpointing,
        max_length=args.max_length,
        max_seq_length=args.max_length,  # older TRL versions use this name instead
        # Packs multiple short examples into each full-length sequence
        # instead of padding each one to max_length separately -- the main
        # speedup here, since real examples are much shorter than 1024/2048
        # tokens. completion_only_loss keeps the loss masked to the answer
        # (not the question/sources), same as before packing was added.
        packing=not args.no_packing,
        completion_only_loss=True,
        optim="paged_adamw_8bit",  # lower optimizer memory/overhead than default adamw
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
    print("Next: run evaluate.py --adapter", args.out_dir,
          "and compare against --adapter none (the base model) on sft_data/test.jsonl")

    if args.smoke_test:
        print("\nSMOKE TEST DONE. If this looked fine (loss printed, no errors), "
              "rerun WITHOUT --smoke_test for the real training run.")


if __name__ == "__main__":
    main()