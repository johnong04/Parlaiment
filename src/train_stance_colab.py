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
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
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
    df_gold["label"] = df_gold["Stance"].map(LABEL_MAP)

    # Drop rows with missing labels if any
    df_gold = df_gold.dropna(subset=["label"])
    df_gold["label"] = df_gold["label"].astype(int)

    # Stratified split: 80% Train, 20% Test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_gold["Text"].tolist(),
        df_gold["label"].tolist(),
        test_size=0.2,
        stratify=df_gold["label"].tolist(),
        random_state=42,
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
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=256
    )


# Create HF Datasets
train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Map tokenization
train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# %% [markdown]
# ### Cell 5: Model Initialization
from transformers import EarlyStoppingCallback


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": acc}


model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=4
)

# %% [markdown]
# ### Cell 6: Training Configuration (OPTIMIZED)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  # Increased for deeper learning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,  # Slightly higher for small data
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=5,
    report_to="none",
)

# %% [markdown]
# ### Cell 7: "All In" Training with Class Weights
from transformers import EarlyStoppingCallback, Trainer
import torch.nn as nn

# Calculate class weights (Neutral is easy, Pro/Evasive are hard)
# We give higher weights to minority/harder classes
class_weights = torch.tensor([2.0, 1.5, 1.0, 2.0]).to(
    "cuda"
)  # Pro, Con, Neutral, Evasive


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting 'All In' training...")
trainer.train()
trainer.save_model(MODEL_DIR)

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
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(LABEL_MAP.keys()),
    yticklabels=list(LABEL_MAP.keys()),
)
plt.title("Confusion Matrix: Stance Classification")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.savefig(CONF_MATRIX_PATH)
plt.show()
print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

# %% [markdown]
# ### Cell 9: Full Inference (16k Rows)
if not os.path.exists(FULL_DATA):
    print(f"ERROR: {FULL_DATA} not found.")
else:
    df_full = pd.read_csv(FULL_DATA)
    full_texts = df_full["Text"].astype(str).tolist()

    # Create dataset for inference
    full_ds = Dataset.from_dict({"text": full_texts})
    full_ds = full_ds.map(tokenize_function, batched=True)

    print(f"Running inference on {len(df_full)} rows...")
    predictions_full = trainer.predict(full_ds)
    y_pred_full = np.argmax(predictions_full.predictions, axis=-1)

    # Map back to labels
    df_full["Stance"] = [INV_LABEL_MAP[pred] for pred in y_pred_full]

# %% [markdown]
# ### Cell 10: Export Final CSV
df_full.to_csv(OUTPUT_CSV, index=False)
print(f"Success! Final analyzed data saved to {OUTPUT_CSV}")

# %% [markdown]
# ### Cell 11: Scientific Proof & EDA Visuals
# This cell generates the specific visualizations needed for your FYP slides.

# 1. Training Loss Curve (Slide 5: Methodology)
history = trainer.state.log_history
train_loss = [x["loss"] for x in history if "loss" in x]
eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.title("Fine-Tuning: Training vs Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("data/loss_curve.png", bbox_inches="tight")
plt.show()

# 2. The "Evasiveness Leaderboard" (Top 10 MPs)
evasive_counts = (
    df_full[df_full["Stance"] == "Evasive"]["Speaker"].value_counts().head(10)
)
plt.figure(figsize=(10, 6))
sns.barplot(x=evasive_counts.values, y=evasive_counts.index, palette="viridis")
plt.title("Top 10 Most Evasive Speakers (Total Count)")
plt.xlabel("Number of Evasive Speeches")
plt.savefig("data/evasive_leaderboard.png", bbox_inches="tight")
plt.show()

# 3. Stance Distribution by Coalition (Stacked Bar)
# Filter out neutral to focus on interesting patterns
pivot_df = pd.crosstab(df_full["Coalition"], df_full["Stance"], normalize="index") * 100
pivot_df = pivot_df[["Pro", "Con", "Evasive", "Neutral"]]  # Reorder for clarity
pivot_df.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=["#2ecc71", "#e74c3c", "#FFBF00", "#95a5a6"],
)
plt.title("Stance Distribution by Coalition (%)")
plt.ylabel("Percentage")
plt.legend(title="Stance", bbox_to_anchor=(1.05, 1))
plt.savefig("data/stance_by_coalition.png", bbox_inches="tight")
plt.show()

# 4. Topic vs. Evasiveness Heatmap (with HUMAN NAMES)
# Filter for meaningful topics only
noise_ids = [-1, 0, 1, 2, 4, 7, 8, 9, 11, 12]
useful_df = df_full[~df_full["Topic"].astype(float).astype(int).isin(noise_ids)].copy()

# Apply the human-friendly mapping from the dashboard
topic_human_map = {
    3: "National Administration",
    5: "Education & Schools",
    6: "Committee Business",
    10: "Budget & Finance",
    13: "Urban & Town Planning",
    14: "Islamic Affairs & Syariah",
    15: "Healthcare & Hospitals",
    16: "Regional (KL/Selayang)",
    20: "Water & Infrastructure",
    22: "Housing & Development",
    24: "Taxation & GST",
    30: "Entrepreneurship & SMEs",
    33: "Telecommunications & Digital",
    34: "Public Health & Tobacco",
    37: "Oil & Gas Industry",
    38: "Women & Gender Policy",
    41: "1MDB & Asset Recovery",
    44: "Public Safety & Police",
    47: "Regional (Hulu Langat)",
    48: "Financial Matters",
}

useful_df["Topic_Human"] = (
    useful_df["Topic"]
    .astype(float)
    .astype(int)
    .map(lambda x: topic_human_map.get(x, f"Topic {x}"))
)

# Get top 10 substantive topics
top_topics = useful_df["Topic_Human"].value_counts().head(10).index
heatmap_data = useful_df[useful_df["Topic_Human"].isin(top_topics)]

pivot_heatmap = pd.crosstab(
    heatmap_data["Topic_Human"], heatmap_data["Stance"], normalize="index"
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_heatmap, annot=True, cmap="YlGnBu")
plt.title("Evasiveness Intensity by Policy Topic (%)")
plt.ylabel("Policy Area")
plt.savefig("data/topic_stance_heatmap.png", bbox_inches="tight")
plt.show()
