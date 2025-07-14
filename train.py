import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def format_agri(example):
    return {
        "instruction": "Answer the agricultural question.",
        "input": example["questions"],
        "output": example["answers"]
    }

def generate_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",  # Optional: "longest" if you want dynamic
        max_length=256
    )

def main():
    torch.set_num_threads(4)  # Limit to 4 CPU threads (optional for balance)

    # ---------------------------- 1. Load dataset ---------------------------- (we have directly loaded the dataset instead of using a JSON alpaca format)
    raw_dataset = load_dataset("shchoi83/agriQA", split="train")
    dataset = raw_dataset.map(format_agri)
    dataset = dataset.map(generate_prompt, num_proc=4)

    # ---------------------------- 2. Tokenizer & Model ------------------------
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # ---------------------------- 3. Apply LoRA ------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(">>> Model is on:", next(model.parameters()).device)

    # ---------------------------- 4. Tokenize --------------------------------
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=dataset.column_names, #(remove_columns to avoid issues with unused columns)
        num_proc=4
    )

    # ---------------------------- 5. Training Setup --------------------------
    training_args = TrainingArguments(
        output_dir="./lora-agriqa",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        remove_unused_columns=False,
        save_safetensors=True,  # ✅ Add this line
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) #No MLM for causal LM

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ---------------------------- 6. Train & Save ----------------------------
    trainer.train()
    model.save_pretrained("./lora-agriqa-adapter", safe_serialization=True)
    print(">>> Adapter model saved to ./lora-agriqa-adapter")

# ---------------------------- Run Safely on Windows -------------------------
if __name__ == "__main__":
    main()
