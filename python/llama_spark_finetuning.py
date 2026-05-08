import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# =========================
# Configuration
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Use English version by default - change to "korean" for Korean text
LANGUAGE = "english"  # or "korean"

TRAIN_FILE = f"./spark_{LANGUAGE}_train.json"
VALID_FILE = f"./spark_{LANGUAGE}_val.json"
TEST_FILE = f"./spark_{LANGUAGE}_test.json"
OUTPUT_DIR = f"./llama_spark_{LANGUAGE}_finetuned"

MAX_SEQ_LENGTH = 256

# Get unique emotions from the dataset
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

unique_emotions = sorted(list(set([item['label'] for item in train_data])))
print(f"Found {len(unique_emotions)} unique emotions: {unique_emotions}")

LABELS = unique_emotions

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

print("Label mapping:")
for label, id in label2id.items():
    print(f"  {id}: {label}")

# =========================
# Metrics helpers
# =========================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

# =========================
# Load dataset
# =========================
data_files = {"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE}

dataset = load_dataset("json", data_files=data_files)

# Convert string label -> integer label id
def encode_labels(example):
    return {
        "sentence": example["sentence"],
        "label": label2id[example["label"]]
    }

dataset = dataset.map(encode_labels)

print("Dataset info:")
print(dataset)
print("\nExample:")
print(dataset["train"][0])

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# 4-bit quantization config
# =========================
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

# =========================
# Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=compute_dtype,
)

# =========================
# LoRA config for sequence classification
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",  # Changed from CAUSAL_LM to SEQ_CLS
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =========================
# Tokenization
# =========================
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # Let DataCollator handle padding
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=f"./logs/llama_spark_{LANGUAGE}",
    num_train_epochs=3,
    per_device_train_batch_size=4,   # Smaller batch size for Llama
    gradient_accumulation_steps=4,   # Effective batch size = 16
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =========================
# Train
# =========================
print(f"\nStarting training for {LANGUAGE} SPARK dataset with Llama...")
trainer.train()

# =========================
# Evaluate on test set
# =========================
print(f"\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# =========================
# Save
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save label mappings
import pickle
with open(f"{OUTPUT_DIR}/label_mappings.pkl", "wb") as f:
    pickle.dump({"label2id": label2id, "id2label": id2label}, f)

print(f"\nSaved model to {OUTPUT_DIR}")
print(f"Saved label mappings to {OUTPUT_DIR}/label_mappings.pkl")