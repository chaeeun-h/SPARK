import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os
from datetime import datetime

# =========================
# Config
# =========================
MODEL_DIR = "./kobert_spark_korean_finetuned"  # KoBERT model directory
TEST_FILE = "./spark_korean_test.json"  # Korean test data

# Create logs directory
os.makedirs("./logs/kobert_spark_korean", exist_ok=True)

# Setup logging to file
log_filename = f"./logs/kobert_spark_korean/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, 'w', encoding='utf-8')

def log_print(text):
    print(text)
    log_file.write(text + '\n')
    log_file.flush()

# Load label mappings
with open(f"{MODEL_DIR}/label_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

label2id = mappings["label2id"]
id2label = mappings["id2label"]
LABELS = list(label2id.keys())

log_print(f"Loaded KoBERT model from {MODEL_DIR}")
log_print(f"Labels: {LABELS}")

# =========================
# Load model and tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=True)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

log_print(f"Using device: {device}")

# =========================
# Load test data
# =========================
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

log_print(f"Loaded {len(test_data)} test samples")

# =========================
# Inference
# =========================
predictions = []
true_labels = []

log_print("Running inference...")
for item in test_data:
    sentence = item["sentence"]
    true_label = item["label"]

    # Tokenize
    inputs = tokenizer(
        sentence,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        pred_logits = outputs.logits
        pred_label_id = torch.argmax(pred_logits, dim=1).item()
        pred_label = id2label[pred_label_id]

    predictions.append(pred_label)
    true_labels.append(true_label)

# =========================
# Convert true labels to string format for evaluation
# =========================
true_labels_str = [label for label in true_labels]

# =========================
# Metrics
# =========================
def compute_metrics_all(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

def compute_metrics_excluding_rare(y_true, y_pred, rare_labels):
    """Exclude rare emotion classes that appear < 5 times in test set"""
    y_true_filtered = []
    y_pred_filtered = []

    for true, pred in zip(y_true, y_pred):
        if true not in rare_labels:
            y_true_filtered.append(true)
            y_pred_filtered.append(pred)

    if len(y_true_filtered) == 0:
        return {"error": "No samples left after filtering"}

    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "filtered_samples": len(y_true_filtered)
    }

# Calculate metrics
metrics_all = compute_metrics_all(true_labels_str, predictions)

# Find rare labels (appear < 5 times in test set)
from collections import Counter
label_counts = Counter(true_labels_str)
rare_labels = [label for label, count in label_counts.items() if count < 5]
log_print(f"Rare labels (excluded from filtered metrics): {rare_labels}")

metrics_filtered = compute_metrics_excluding_rare(true_labels_str, predictions, rare_labels)

# =========================
# Print results
# =========================
log_print("\n" + "="*50)
log_print("KoBERT SPARK Model Evaluation Results")
log_print("="*50)

log_print("\nAll Classes Metrics:")
for key, value in metrics_all.items():
    log_print(f"  {key}: {value:.4f}")

log_print(f"\nFiltered Metrics (excluding rare classes: {rare_labels}):")
for key, value in metrics_filtered.items():
    if key != "filtered_samples":
        log_print(f"  {key}: {value:.4f}")
    else:
        log_print(f"  {key}: {value}")

log_print(f"\nDetailed Classification Report:")
log_print(classification_report(true_labels_str, predictions, zero_division=0))

# =========================
# Save predictions
# =========================
results_df = []
for i, (true, pred) in enumerate(zip(true_labels_str, predictions)):
    results_df.append({
        "index": i,
        "sentence": test_data[i]["sentence"],
        "true_label": true,
        "predicted_label": pred,
        "correct": true == pred
    })

import pandas as pd
results_df = pd.DataFrame(results_df)
results_df.to_csv(f"{MODEL_DIR}/test_predictions.csv", index=False)

log_print(f"\nSaved detailed predictions to {MODEL_DIR}/test_predictions.csv")

# Show some examples
log_print("\nSample predictions:")
correct_samples = results_df[results_df["correct"] == True].head(3)
incorrect_samples = results_df[results_df["correct"] == False].head(3)

log_print("\nCorrect predictions:")
for _, row in correct_samples.iterrows():
    log_print(f"  Text: {row['sentence'][:50]}...")
    log_print(f"  True: {row['true_label']}, Pred: {row['predicted_label']}")

log_print("\nIncorrect predictions:")
for _, row in incorrect_samples.iterrows():
    log_print(f"  Text: {row['sentence'][:50]}...")
    log_print(f"  True: {row['true_label']}, Pred: {row['predicted_label']}")

log_print(f"\nEvaluation log saved to: {log_filename}")
log_file.close()