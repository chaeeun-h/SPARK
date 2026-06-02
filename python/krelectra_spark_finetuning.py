import json
import numpy as np
import torch
import pickle
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
# Korean GoEmotions model — XLM-RoBERTa fine-tuned on Korean GoEmotions
MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"
LANGUAGE   = "korean"

TRAIN_FILE = f"./data/spark_{LANGUAGE}_train.json"
VALID_FILE = f"./data/spark_{LANGUAGE}_val.json"
TEST_FILE  = f"./data/spark_{LANGUAGE}_test.json"
OUTPUT_DIR = f"./data/kobert_spark_{LANGUAGE}_finetuned"

MAX_LENGTH = 256

# =========================
# Label mapping (from train file)
# =========================
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

LABELS   = sorted(list(set([item['label'] for item in train_data])))
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

print(f"Found {len(LABELS)} unique emotions")

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy":        acc,
        "macro_precision": precision,
        "macro_recall":    recall,
        "macro_f1":        f1,
    }

# =========================
# Load & encode dataset
# =========================
data_files = {"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE}
dataset    = load_dataset("json", data_files=data_files)

def encode_labels(example):
    return {"sentence": example["sentence"],
            "label":    label2id[example["label"]]}

dataset = dataset.map(encode_labels)

# =========================
# Tokenizer & collator
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# Model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)
model.to(device)
print(f"Using device: {device}")

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    logging_dir                 = f"./logs/kobert_spark_{LANGUAGE}",
    learning_rate               = 1e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    num_train_epochs            = 10,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    logging_strategy            = "steps",
    logging_steps               = 10,
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",
    greater_is_better           = True,
    report_to                   = "none",
    save_total_limit            = 2,
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_dataset["train"],
    eval_dataset    = tokenized_dataset["validation"],
    processing_class= tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# =========================
# Train & evaluate
# =========================
print(f"\nStarting training — KR-ELECTRA ({LANGUAGE})...")
trainer.train()

print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# =========================
# Save
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(f"{OUTPUT_DIR}/label_mappings.pkl", "wb") as f:
    pickle.dump({"label2id": label2id, "id2label": id2label}, f)

print(f"\nSaved model to {OUTPUT_DIR}")
