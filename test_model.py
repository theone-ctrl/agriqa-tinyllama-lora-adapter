# 📘 Fine-Tuned vs Base Model Evaluation (TinyLlama + LoRA on AgriQA)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load fine-tuned model with LoRA adapter
adapter_path = "theone049/agriqa-tinyllama-lora-adapter"
ft_model = PeftModel.from_pretrained(base_model, adapter_path)
ft_model.eval()

# Helper function to generate answer
def generate_answer(model, question, max_tokens=256):
    prompt = f"""### Instruction:
Answer the agricultural question.

### Input:
{question}

### Response:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()

# Sample AgriQA-style questions
questions = [
    "My paddy leaves are turning pale and the growth is stunted. Is it zinc deficiency? What should I apply?",
    "My okra leaves have black circular spots with yellow halos. What disease is this and how to treat it?",
    "What is the best intercrop with groundnut for rainfed farms in Tamil Nadu?",
    "My tapioca plants have white wool-like insects. How do I control mealybugs organically?",
    "Is it good to leave sugarcane trash on the field? Will it help soil fertility?",
]


# Run comparison
for q in questions:
    print(f"\n🔹 Question: {q}")
    print("🔸 Base Model Answer:")
    print(generate_answer(base_model, q))
    print("🔸 Fine-Tuned Model Answer (LoRA):")
    print(generate_answer(ft_model, q))
