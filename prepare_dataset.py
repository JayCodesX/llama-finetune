"""
prepare_dataset.py — Builds a blended fine-tune dataset for general-purpose QLoRA.

Sources:
  1. No Robots (human-written, natural tone)
  2. Capybara (multi-turn reasoning)
  3. SlimOrca (instruction following)
  4. my_examples.jsonl (your curated Claude conversations — highest value)

Run:
  pip install datasets
  python prepare_dataset.py

Output: dataset.jsonl (ready for training)
"""

import json
import random
from pathlib import Path

random.seed(42)

# Allow running without datasets library if user just wants to use my_examples.jsonl
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("WARNING: `datasets` not installed. Run: pip install datasets")
    print("Only loading my_examples.jsonl\n")
    HAS_DATASETS = False

output: list[dict] = []

MIN_LEN = 50    # skip very short assistant responses
MAX_LEN = 4000  # skip extremely long ones (saves VRAM)


def normalize_messages(msgs: list[dict]) -> list[dict]:
    """Normalize role names and filter empty messages."""
    normalized = []
    for m in msgs:
        role = m.get("from", m.get("role", ""))
        content = m.get("value", m.get("content", ""))
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role in ("user", "assistant", "system") and content.strip():
            normalized.append({"role": role, "content": content.strip()})
    return normalized


def assistant_text_length(msgs: list[dict]) -> int:
    return sum(len(m["content"]) for m in msgs if m["role"] == "assistant")


def is_valid_conversation(msgs: list[dict]) -> bool:
    if len(msgs) < 2:
        return False
    has_user = any(m["role"] == "user" for m in msgs)
    has_assistant = any(m["role"] == "assistant" for m in msgs)
    length = assistant_text_length(msgs)
    return has_user and has_assistant and MIN_LEN < length < MAX_LEN


# ── 1. No Robots (human-written, most natural tone) ─────────────────────────

if HAS_DATASETS:
    print("Loading No Robots...")
    try:
        nr = load_dataset("HuggingFaceH4/no_robots", split="train")
        nr_indices = random.sample(range(len(nr)), min(300, len(nr)))
        nr_count = 0
        for i in nr_indices:
            msgs = normalize_messages(nr[i]["messages"])
            if is_valid_conversation(msgs):
                output.append({"messages": msgs})
                nr_count += 1
                if nr_count >= 200:
                    break
        print(f"  No Robots: {nr_count} examples")
    except Exception as e:
        print(f"  No Robots failed: {e}")

# ── 2. Capybara (multi-turn reasoning) ──────────────────────────────────────

if HAS_DATASETS:
    print("Loading Capybara...")
    try:
        cap = load_dataset("LDJnr/Capybara", split="train")
        cap_indices = random.sample(range(len(cap)), min(400, len(cap)))
        cap_count = 0
        for i in cap_indices:
            conv = cap[i]["conversation"]
            msgs = []
            for turn in conv:
                if turn.get("input"):
                    msgs.append({"role": "user", "content": turn["input"].strip()})
                if turn.get("output"):
                    msgs.append({"role": "assistant", "content": turn["output"].strip()})
            if is_valid_conversation(msgs):
                output.append({"messages": msgs})
                cap_count += 1
                if cap_count >= 200:
                    break
        print(f"  Capybara: {cap_count} examples")
    except Exception as e:
        print(f"  Capybara failed: {e}")

# ── 3. SlimOrca (instruction following) ─────────────────────────────────────

if HAS_DATASETS:
    print("Loading SlimOrca...")
    try:
        orca = load_dataset("Open-Orca/SlimOrca", split="train")
        orca_indices = random.sample(range(len(orca)), min(400, len(orca)))
        orca_count = 0
        for i in orca_indices:
            msgs = normalize_messages(orca[i]["conversations"])
            if is_valid_conversation(msgs):
                output.append({"messages": msgs})
                orca_count += 1
                if orca_count >= 200:
                    break
        print(f"  SlimOrca: {orca_count} examples")
    except Exception as e:
        print(f"  SlimOrca failed: {e}")

# ── 4. Your curated examples (highest value) ────────────────────────────────

my_examples_path = Path(__file__).parent / "my_examples.jsonl"
if my_examples_path.exists():
    print("Loading my_examples.jsonl...")
    my_count = 0
    with open(my_examples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                msgs = normalize_messages(ex.get("messages", []))
                if len(msgs) >= 2:
                    output.append({"messages": msgs})
                    my_count += 1
            except json.JSONDecodeError:
                continue
    print(f"  Your examples: {my_count}")
else:
    print("  Skipping my_examples.jsonl (not found — create it for best results)")

# ── Shuffle and save ─────────────────────────────────────────────────────────

random.shuffle(output)

out_path = Path(__file__).parent / "dataset.jsonl"
with open(out_path, "w") as f:
    for ex in output:
        f.write(json.dumps(ex) + "\n")

print(f"\nTotal: {len(output)} examples -> {out_path}")

# ── Stats ────────────────────────────────────────────────────────────────────

avg_turns = sum(len(ex["messages"]) for ex in output) / max(len(output), 1)
avg_len = sum(
    sum(len(m["content"]) for m in ex["messages"])
    for ex in output
) / max(len(output), 1)

print(f"Avg turns per example: {avg_turns:.1f}")
print(f"Avg total chars per example: {avg_len:.0f}")
