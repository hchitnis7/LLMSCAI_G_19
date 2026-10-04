"""Run this to check whether the base model actually has separate
gate_proj/up_proj layers (what finetune.py's target_modules expects) or a
fused gate_up_proj layer instead (which would mean those weights were never
targeted by LoRA during training).

Usage:
    python inspect_model_layers.py --model Qwen/Qwen3-30B-A3B-Instruct-2507
    python inspect_model_layers.py --model Qwen/Qwen3-14B   # if checking the L4 run
    python inspect_model_layers.py --adapter adapter        # also checks against the adapter's own config
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
ap.add_argument("--adapter", default=None)
args = ap.parse_args()

from transformers import AutoModelForCausalLM

print(f"Loading {args.model} (just to inspect layer names, not for inference)...")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="cpu")

names = [n for n, _ in model.named_modules()]
has_gate_proj = any(n.endswith(".gate_proj") for n in names)
has_up_proj = any(n.endswith(".up_proj") for n in names)
has_gate_up_proj = any(n.endswith(".gate_up_proj") for n in names)
has_down_proj = any(n.endswith(".down_proj") for n in names)

print("\n=== Linear layer name check ===")
print(f"separate .gate_proj found:  {has_gate_proj}")
print(f"separate .up_proj found:    {has_up_proj}")
print(f"fused .gate_up_proj found:  {has_gate_up_proj}")
print(f".down_proj found:           {has_down_proj}")

print("\n=== A few example matching layer names ===")
for suffix in [".gate_proj", ".up_proj", ".gate_up_proj", ".down_proj", ".q_proj"]:
    examples = [n for n in names if n.endswith(suffix)][:2]
    if examples:
        print(f"{suffix}: {examples}")

if has_gate_up_proj and not (has_gate_proj and has_up_proj):
    print("\n*** This model uses FUSED gate_up_proj. ***")
    print("finetune.py's target_modules=['gate_proj','up_proj',...] did NOT match these layers.")
    print("The gate/up projections were almost certainly NOT fine-tuned.")
    print("Fix: in finetune.py / finetune_l4.py, change target_modules to use 'gate_up_proj'")
    print("instead of separate 'gate_proj' and 'up_proj', then retrain.")
elif has_gate_proj and has_up_proj:
    print("\n*** This model uses separate gate_proj/up_proj. ***")
    print("finetune.py's target_modules list is correct for this architecture.")
    print("The warning you saw is likely a harmless leftover pattern in the adapter config")
    print("(e.g. from a PEFT auto-detect step) that simply doesn't apply here -- safe to ignore.")
else:
    print("\nUnexpected: found neither pattern clearly. Check the printed layer names above by hand.")

if args.adapter and Path(args.adapter, "adapter_config.json").exists():
    cfg = json.load(open(Path(args.adapter, "adapter_config.json")))
    print(f"\n=== {args.adapter}/adapter_config.json ===")
    print("target_modules:", cfg.get("target_modules"))
    print("rank_pattern:", cfg.get("rank_pattern"))
    print("alpha_pattern:", cfg.get("alpha_pattern"))
