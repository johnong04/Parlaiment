# %% [markdown]
# # Phase 2: Stance Classification (XLM-RoBERTa)
# This script trains a stance classifier on labeled Hansard data and predicts stance for the full dataset.
# Optimized for Google Colab (T4 GPU).

# %% [markdown]
# ### Cell 1: Environment Setup
# Install required packages if running in Google Colab.

# %%
# !pip install transformers datasets scikit-learn accelerate seaborn matplotlib

# %% [markdown]
# ### Cell 2: Imports and Configuration
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
GOLD_DATA = "data/gold_stance_data.csv"
FULL_DATA = "data/hansard_with_topics.csv"
OUTPUT_CSV = "data/hansard_final_analyzed.csv"
MODEL_DIR = "stance_model"
CONF_MATRIX_PATH = "data/confusion_matrix.png"

# Label mapping
LABEL_MAP = {"Pro": 0, "Con": 1, "Neutral": 2, "Evasive": 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# %% [markdown]
# ### Cell 3: Load and Split Data
if not os.path.exists(GOLD_DATA):
    print(f"ERROR: {GOLD_DATA} not found.")
else:
    df_gold = pd.read_csv(GOLD_DATA)
    
    # Ensure labels are mapped correctly
    df_gold['label'] = df_gold['Stance'].map(LABEL_MAP)
    
    # Drop rows with missing labels if any
    df_gold = df_gold.dropna(subset=['label'])
    df_gold['label'] = df_gold['label'].astype(int)

    # Stratified split: 80% Train, 20% Test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_gold['Text'].tolist(),
        df_gold['label'].tolist(),
        test_size=0.2,
        stratify=df_gold['label'].tolist(),
        random_state=42
    )

    print(f"Total labeled rows: {len(df_gold)}")
    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")
    print("\nClass distribution in Test Set:")
    print(pd.Series(test_labels).map(INV_LABEL_MAP).value_counts())

# %% [markdown]
# ### Cell 4: Tokenization
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Create HF Datasets
train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Map tokenization
train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# %% [markdown]
# ### Cell 5: Model Initialization
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": acc}

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)

# %% [markdown]
# ### Cell 6: Training Configuration (FIXED)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",          # <-- ADD THIS LINE to disable wandb
)

# %% [markdown]
# ### Cell 7: Training Execution
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# Save best model
trainer.save_model(MODEL_DIR)
print(f"Best model saved to {MODEL_DIR}")

# %% [markdown]
# ### Cell 8: Evaluation - Confusion Matrix
print("\n--- Final Evaluation on Test Set ---")
predictions = trainer.predict(test_ds)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = test_labels

# Classification Report
print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.keys())))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(LABEL_MAP.keys()), 
            yticklabels=list(LABEL_MAP.keys()))
plt.title('Confusion Matrix: Stance Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(CONF_MATRIX_PATH)
plt.show()
print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

# %% [markdown]
# ### Cell 9: Full Inference (16k Rows)
if not os.path.exists(FULL_DATA):
    print(f"ERROR: {FULL_DATA} not found.")
else:
    df_full = pd.read_csv(FULL_DATA)
    full_texts = df_full['Text'].astype(str).tolist()
    
    # Create dataset for inference
    full_ds = Dataset.from_dict({"text": full_texts})
    full_ds = full_ds.map(tokenize_function, batched=True)
    
    print(f"Running inference on {len(df_full)} rows...")
    predictions_full = trainer.predict(full_ds)
    y_pred_full = np.argmax(predictions_full.predictions, axis=-1)
    
    # Map back to labels
    df_full['Stance'] = [INV_LABEL_MAP[pred] for pred in y_pred_full]
    
# %% [markdown]
# ### Cell 10: Export Final CSV
df_full.to_csv(OUTPUT_CSV, index=False)
print(f"Success! Final analyzed data saved to {OUTPUT_CSV}")

# Summary Stats
print("\nFinal Stance Distribution:")
print(df_full['Stance'].value_counts())
