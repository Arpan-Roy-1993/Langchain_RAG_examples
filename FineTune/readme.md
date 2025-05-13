# 🚀 Fine-Tuning LLaMA 2 with QLoRA  
**A Hands-On Guide Using Hugging Face Transformers**

Unlock the power of fine-tuning LLaMA 2 with efficiency and ease using QLoRA, PEFT, and the Hugging Face ecosystem.

---

## 🧠 Introduction

Large language models like **Meta’s LLaMA 2** are incredibly powerful, but fine-tuning them can be resource-intensive. This project demonstrates how to efficiently fine-tune `LLaMA-2-7b-chat` using:

- **QLoRA** (Quantized Low-Rank Adaptation)
- **LoRA** (Low-Rank Adaptation via PEFT)
- **BitsAndBytes** for 4-bit quantization
- Hugging Face’s **Transformers**, **TRL**, and **Accelerate**

---

## 📦 Step 1: Install Required Packages

```bash
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
📚 Step 2: Import Libraries
python
Copy
Edit
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
🧾 Prompt Format for LLaMA 2 Chat
Use this template for supervised fine-tuning:

vbnet
Copy
Edit
System: <optional system prompt>
User: <instruction/question>
Assistant: <model response>
⚙️ Configuration
🔹 Model and Dataset
python
Copy
Edit
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"
🔹 LoRA Settings
python
Copy
Edit
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
🔹 BitsAndBytes (4-bit Quantization)
python
Copy
Edit
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
🔹 Training Arguments
python
Copy
Edit
output_dir = "./results"
num_train_epochs = 1
per_device_train_batch_size = 4
learning_rate = 2e-4
lr_scheduler_type = "cosine"
gradient_accumulation_steps = 1
gradient_checkpointing = True
logging_steps = 25
save_steps = 0
group_by_length = True
🏋️ Fine-Tuning with SFTTrainer
You can initialize and run the fine-tuning like this:

python
Copy
Edit
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    tokenizer=tokenizer,
)
trainer.train()
✅ Benefits
✅ Cost-efficient: Train 7B models on a single GPU with 4-bit quantization.

✅ Effective: Use LoRA to fine-tune only a small subset of weights.

✅ Modular: Built using Hugging Face’s ecosystem.

📈 Result
After training, the fine-tuned model can be saved and pushed to the Hugging Face Hub or used locally:

python
Copy
Edit
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
📌 References
Hugging Face Transformers

PEFT by Hugging Face

TRL Library

QLoRA Paper

💬 Feedback
Pull requests, suggestions, and issues are welcome! If you find this helpful, consider starring the repo 
