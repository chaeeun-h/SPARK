import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the SPARK dataset
spark_df = pd.read_csv("./SPARK_experiment_dataset_1635.csv")

# Check the emotion distribution
print("Original emotion distribution:")
print(spark_df['emotion'].value_counts())
print(f"\nTotal samples: {len(spark_df)}")

# Split the data into train/val/test (80/10/10)
# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    spark_df,
    test_size=0.2,
    random_state=42
    # Removed stratify due to classes with only 1 sample
)

# Second split: 50% val, 50% test from temp (10% each of original)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

print(f"\nTrain set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

print("\nTrain emotion distribution:")
print(train_df['emotion'].value_counts())
print("\nValidation emotion distribution:")
print(val_df['emotion'].value_counts())
print("\nTest emotion distribution:")
print(test_df['emotion'].value_counts())

# Function to convert DataFrame to JSON format for transformers
def df_to_json(df, text_column='sentence_en', label_column='emotion'):
    """Convert DataFrame to JSON format expected by transformers"""
    records = []
    for _, row in df.iterrows():
        record = {
            "sentence": row[text_column],
            "label": row[label_column]
        }
        records.append(record)
    return records

# Create JSON files for both English and Korean versions
datasets = {
    "english": "sentence_en",
    "korean": "sentence_ko"
}

for lang_name, text_col in datasets.items():
    print(f"\nCreating {lang_name} datasets...")

    # Convert to JSON format
    train_json = df_to_json(train_df, text_column=text_col)
    val_json = df_to_json(val_df, text_column=text_col)
    test_json = df_to_json(test_df, text_column=text_col)

    # Save JSON files
    train_file = f"./spark_{lang_name}_train.json"
    val_file = f"./spark_{lang_name}_val.json"
    test_file = f"./spark_{lang_name}_test.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_json, f, ensure_ascii=False, indent=2)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)

    print(f"Saved {lang_name} datasets:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print(f"  Test: {test_file}")

print("\nData splitting completed!")