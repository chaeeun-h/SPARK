# SPARK Emotion Classification Fine-tuning

This directory contains scripts to fine-tune multiple transformer models (RoBERTa, BERT, DeBERTa, KoBERT, Llama) for emotion classification on the SPARK dataset.

## Dataset

The SPARK dataset contains Korean social media posts with English translations and emotion labels. We've split it into:
- Train: 1,394 samples (80%)
- Validation: 171 samples (10%)
- Test: 171 samples (10%)

Available in both English (`sentence_en`) and Korean (`sentence_ko`) versions.

## Project Structure

```
CIKM/
├── python/                    # Python scripts
│   ├── spark_data_split.py
│   ├── *_spark_finetuning.py  # Fine-tuning scripts
│   └── *_spark_testing.py     # Evaluation scripts
├── data/                      # Dataset files
│   ├── SPARK_experiment_dataset_1635.csv (share upon request)
│   └── spark_*_{train,val,test}.json
├── logs/                      # Training and evaluation logs
│   ├── roberta_spark_english/
│   ├── bert_spark_english/
│   ├── deberta_spark_english/
│   ├── kobert_spark_korean/
│   └── llama_spark_english/
├── figures/                   # Generated plots and visualizations
├── SPARK_emotion_analysis.ipynb  # Analysis notebook
└── SPARK_FINE_TUNING_README.md   # This documentation
```

## Scripts Overview

### 1. Data Preparation
- `python/spark_data_split.py`: Splits the dataset and creates JSON files for training

### 2. RoBERTa Fine-tuning (English)
- `python/roberta_spark_finetuning.py`: Fine-tunes RoBERTa-base for emotion classification
- `python/roberta_spark_testing.py`: Evaluates the fine-tuned RoBERTa model

### 3. Llama Fine-tuning (English)
- `python/llama_spark_finetuning.py`: Fine-tunes Llama-3.1-8B with LoRA for emotion classification
- `python/llama_spark_testing.py`: Evaluates the fine-tuned Llama model

### 4. KoBERT Fine-tuning (Korean)
- `python/kobert_spark_finetuning.py`: Fine-tunes KoBERT for Korean emotion classification
- `python/kobert_spark_testing.py`: Evaluates the fine-tuned KoBERT model

### 5. BERT Fine-tuning (English)
- `python/bert_spark_finetuning.py`: Fine-tunes BERT-base for emotion classification
- `python/bert_spark_testing.py`: Evaluates the fine-tuned BERT model

### 6. DeBERTa Fine-tuning (English)
- `python/deberta_spark_finetuning.py`: Fine-tunes DeBERTa-base for emotion classification
- `python/deberta_spark_testing.py`: Evaluates the fine-tuned DeBERTa model

## Usage

### Step 1: Prepare Data
```bash
python3 python/spark_data_split.py
```

### Step 2: Fine-tune Models

#### English Models:
```bash
# RoBERTa (English)
python3 python/roberta_spark_finetuning.py

# BERT (English)
python3 python/bert_spark_finetuning.py

# DeBERTa (English)
python3 python/deberta_spark_finetuning.py

# Llama (English)
python3 python/llama_spark_finetuning.py
```

#### Korean Models:
```bash
# KoBERT (Korean)
python3 python/kobert_spark_finetuning.py
```

### Step 3: Evaluate Models
```bash
# RoBERTa evaluation
python3 python/roberta_spark_testing.py

# BERT evaluation
python3 python/bert_spark_testing.py

# DeBERTa evaluation
python3 python/deberta_spark_testing.py

# Llama evaluation
python3 python/llama_spark_testing.py

# KoBERT evaluation
python3 python/kobert_spark_testing.py
```

## Logging

### Training Logs
During fine-tuning, training progress is logged to both console and files:
- **Console Output**: Real-time training metrics (loss, learning rate, validation metrics)
- **Log Files**: Saved in `logs/{model}_{language}/` directory with timestamps
- **Format**: `logs/roberta_spark_english/20241208_143022.log`

### Evaluation Logs
Evaluation results are saved to timestamped log files:
- **Metrics**: Accuracy, F1, precision, recall for all classes and filtered metrics

### Log File Locations
```
logs/
├── roberta_spark_english/
│   └── evaluation_20241208_143022.log
├── bert_spark_english/
├── deberta_spark_english/
├── kobert_spark_korean/
└── llama_spark_english/
```

## Configuration

### RoBERTa (English)
- Model: `roberta-base`
- Language: English
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 5
- Max length: 256

### BERT (English)
- Model: `bert-base-uncased`
- Language: English
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 5
- Max length: 256

### DeBERTa (English)
- Model: `microsoft/deberta-base`
- Language: English
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 5
- Max length: 256

### KoBERT (Korean)
- Model: `monologg/kobert`
- Language: Korean
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 5
- Max length: 256

### Llama (English)
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Language: English
- LoRA rank: 16
- Batch size: 4 (with gradient accumulation = 4, effective = 16)
- Learning rate: 2e-4
- Epochs: 3
- Max length: 256
- 4-bit quantization

## Output Files

After training, each model directory contains:
- `pytorch_model.bin` or adapter files
- `tokenizer.json`, `tokenizer_config.json`
- `label_mappings.pkl`: Label to ID mappings
- `test_predictions.csv`: Detailed predictions on test set

## Metrics

Models are evaluated on:
- Accuracy
- Macro Precision/Recall/F1
- Per-class metrics (classification report)

Rare classes (< 5 samples in test set) are excluded from filtered metrics.

## Requirements

Install dependencies:
```bash
pip install transformers datasets torch peft accelerate bitsandbytes scikit-learn pandas
```

For Llama fine-tuning, you'll need access to the Llama model (requires Hugging Face token).

## Notes

- The dataset has 26 emotion classes, some with very few samples
- Training time: ~30-60 minutes for BERT/RoBERTa/DeBERTa, ~2-4 hours for Llama, ~30-60 minutes for KoBERT (depending on hardware)
- Models are saved with the best checkpoint based on validation macro F1
