import re
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# =========================
# Config
# =========================
MODEL_NAME = "monologg/kobert"  # KoBERT model
LANGUAGE = "korean"  # KoBERT is specifically for Korean

TRAIN_FILE = f"./spark_{LANGUAGE}_train.json"
VALID_FILE = f"./spark_{LANGUAGE}_val.json"
TEST_FILE = f"./spark_{LANGUAGE}_test.json"
OUTPUT_DIR = f"./kobert_spark_{LANGUAGE}_finetuned"

MAX_LENGTH = 256

# Get unique emotions from the dataset
import json
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
)

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=f"./logs/kobert_spark_{LANGUAGE}",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=2,  # Keep only the best 2 checkpoints
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
print(f"\nStarting training for {LANGUAGE} SPARK dataset with KoBERT...")
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