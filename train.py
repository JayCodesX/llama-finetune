"""
train.py — QLoRA fine-tune with unsloth.

Run on Vast.ai (RTX 3090/4090, 24GB VRAM):
  python train.py                          # default: Llama 3.1 8B
  MODEL=70b python train.py                # Llama 3.3 70B (needs 2xA100)

Expects: /workspace/dataset.jsonl (from prepare_dataset.py)
Output:  /workspace/output/final-adapter/
"""

import os
import sys
from pathlib import Path

# ── Config (tweak these) ────────────────────────────────────────────────────

MODEL_SIZE = os.environ.get("MODEL", "8b")
MODEL_NAME = (
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" if MODEL_SIZE == "70b"
    else "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
MAX_SEQ_LENGTH = 2048       # reduce to 1024 if you get OOM
LORA_R = 16                 # rank — 16 is good balance of quality/speed
LORA_ALPHA = 32             # typically 2x rank
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 1              # keep at 1 for 24GB VRAM
GRAD_ACCUM_STEPS = 4        # effective batch size = BATCH_SIZE * GRAD_ACCUM
WARMUP_RATIO = 0.05
SAVE_STEPS = 100

DATASET_PATH = os.environ.get("DATASET_PATH", "/workspace/dataset.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/output")

# ── Imports ──────────────────────────────────────────────────────────────────

print("Loading unsloth...")
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ── HuggingFace auth (needed for gated Llama models) ────────────────────────

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("Using HF_TOKEN for authentication")
else:
    print("No HF_TOKEN set — using unsloth's pre-quantized model (no auth needed)")

# ── Load model ───────────────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # auto-detect bf16/fp16
    token=hf_token,
)

# ── Add LoRA adapters ────────────────────────────────────────────────────────

print(f"Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% VRAM savings
    random_state=42,
)

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

# ── Load dataset ─────────────────────────────────────────────────────────────

print(f"Loading dataset from {DATASET_PATH}...")
if not Path(DATASET_PATH).exists():
    print(f"ERROR: {DATASET_PATH} not found. Run prepare_dataset.py first.")
    sys.exit(1)

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Dataset: {len(dataset)} examples")

def format_chat(example):
    """Apply the model's chat template to the messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

# ── Training ─────────────────────────────────────────────────────────────────

print(f"\nStarting training: {EPOCHS} epochs, lr={LEARNING_RATE}, batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}")
print(f"Output: {OUTPUT_DIR}\n")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        seed=42,
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="none",
    ),
)

trainer.train()

# ── Save final adapter ───────────────────────────────────────────────────────

adapter_path = os.path.join(OUTPUT_DIR, "final-adapter")
print(f"\nSaving adapter to {adapter_path}...")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("\nTraining complete!")
print(f"Adapter saved to: {adapter_path}")
print(f"Size: {sum(f.stat().st_size for f in Path(adapter_path).rglob('*') if f.is_file()) / 1e6:.1f} MB")
