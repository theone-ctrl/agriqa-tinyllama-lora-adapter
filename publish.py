from huggingface_hub import notebook_login
from peft import PeftModel

# Log in to Hugging Face if not already
notebook_login()  # This will prompt you for token in notebook

# Assuming you used a base model like TinyLlama previously
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # or whatever you used
model = PeftModel.from_pretrained(base_model, "lora-agriqa-adapter")

# Push the adapter to the hub
model.push_to_hub("theone049/agriqa-tinyllama-lora-adapter")
