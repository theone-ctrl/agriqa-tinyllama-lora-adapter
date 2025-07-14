# 🌾 AgriQA TinyLLaMA LoRA Adapter

Fine-tuning TinyLLaMA on the AgriQA dataset using LoRA adapters for efficient agriculture-specific question answering. Lightweight, low-resource, and domain-adapted.

---

## 📝 Description

This project fine-tunes a **TinyLLaMA** model using **LoRA (Low-Rank Adaptation)** on the [AgriQA](https://huggingface.co/datasets/AI4AGR/agriqa) dataset — a collection of agriculture-specific Q\&A pairs. It produces a compact LoRA adapter that enhances model performance for domain-specific tasks without the need for full model retraining.

### 🔍 Key Features

* 🌱 **Domain-adapted**: Trained specifically for agriculture-based queries.
* ⚡ **Efficient**: Uses LoRA for parameter-efficient fine-tuning.
* 💡 **Deployable**: Plug-and-play adapter with any compatible LLaMA base.
* 🧠 **Educational**: Great example for fine-tuning LLMs in niche fields.

---

## 📆 Use Cases

* Rural AI assistants and chatbots
* Agri-tech digital advisory platforms
* Agricultural education and extension tools
* Research applications in agri-informatics

---

## 📁 Project Structure

```
Agri model/
│
├── agriqa_alpaca.json         # Converted dataset in Alpaca-style JSON
├── getData.py                 # Script to prepare and convert data
├── train.py                   # Script to fine-tune TinyLLaMA using LoRA
├── test_model.py              # Script to run evaluation or test the model
├── publish.py                 # Script to push adapter to Hugging Face Hub
```

---

## 🏃‍♂️ How to Run

### 1️⃣ Prepare Environment

```bash
# Create and activate a virtual environment (if not already)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Prepare the Dataset

Make sure your dataset is formatted in Alpaca-style JSON format:

```bash
python getData.py
```

This generates `agriqa_alpaca.json`.

---

### 3️⃣ Train the LoRA Adapter

```bash
python train.py
```

This will fine-tune the base model using LoRA and save the adapter weights.

---

### 4️⃣ Run Inference Test

```bash
python test_model.py
```

This loads the base TinyLLaMA model along with your trained LoRA adapter and performs inference on sample AgriQA questions.

---

### 5️⃣ (Optional) Push to Hugging Face

To publish your adapter:

```bash
python publish.py
```

> Make sure you're logged in first:

```bash
huggingface-cli login
```

---

## 🧩 Using the Adapter with the Same Base Model

To use the trained adapter, you must load it with the **same base model** that was used during fine-tuning. The model architecture, hidden sizes, and tokenizer must match exactly for the adapter to function correctly:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("<base-model-path>")
tokenizer = AutoTokenizer.from_pretrained("<base-model-path>")

# Load adapter
model = PeftModel.from_pretrained(base_model, "<adapter-path>")
model.eval()

# Example inference
inputs = tokenizer("Your agriculture question here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Note:** Replace `<base-model-path>` and `<adapter-path>` with the correct paths. Attempting to use the adapter with a different base model (e.g., different LLaMA variant or hidden size) will likely fail due to architectural mismatches.

---

## 🔗 Related

* Base model: [TinyLLaMA](https://huggingface.co/codellama/CodeLlama-7b-hf) or other LLaMA-compatible models
* Dataset: [AgriQA Dataset on Hugging Face](https://huggingface.co/datasets/AI4AGR/agriqa)
* LoRA: [PEFT Library](https://github.com/huggingface/peft)

---

## ✨ Maintainer

You can read the full story and technical breakdown on my Substack blog: [Builder Series #4 – Fine-Tuning a Domain-Specific LLM with LoRA](https://nithyanandamv.substack.com/p/builder-series-4-fine-tuning-a-domain)

Built with ❤️ by [Nithyanandam Venu](https://github.com/your-username)

> ⭐ Star this repo if you find it useful or want to support domain-specific AI research!
