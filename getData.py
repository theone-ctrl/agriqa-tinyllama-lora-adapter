from datasets import load_dataset
import json

# Load AgriQA dataset
dataset = load_dataset("shchoi83/agriQA", split="train")

# Convert to Alpaca-style format
alpaca_data = []

for item in dataset:
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    if question and answer:
        alpaca_data.append({
            "instruction": question,
            "input": "",
            "output": answer
        })

# Save to JSON
with open("agriqa_alpaca.json", "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

print(f"✅ Converted {len(alpaca_data)} entries to Alpaca format")
