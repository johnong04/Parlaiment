# %% [markdown]
# # Phase 2: Topic Modeling (BERTopic)
# This script trains a BERTopic model on Malaysian Hansard data.
# Optimized for Multilingual (Malay/English) text.

# %% [markdown]
# ### Cell 1: Environment Setup
# Install required packages if running in Google Colab.

# %%
# !pip install bertopic sentence-transformers pandas numpy matplotlib

# %% [markdown]
# ### Cell 2: Imports and Configuration
import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import os

# Configuration
INPUT_FILE = "data/hansard_master_final.csv"
OUTPUT_CSV = "data/hansard_with_topics.csv"
OUTPUT_CHART = "data/topic_chart.html"

# %% [markdown]
# ### Cell 3: Load and Preview Data
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found. Please upload it to the data/ folder.")
else:
    df = pd.read_csv(INPUT_FILE)

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    print(f"Loaded {len(df)} rows.")
    display(df.head())

# %% [markdown]
# ### Cell 4: Quick Test Run (Time-Saving)
# Validate settings on a small sample before full training.
sample_df = df.sample(min(1000, len(df)), random_state=42)
test_docs = sample_df["Text"].tolist()

print("Running test fit on 1,000 rows...")
test_model = BERTopic(language="multilingual", nr_topics=20, verbose=True)
test_topics, _ = test_model.fit_transform(test_docs)
display(test_model.get_topic_info().head(10))
print("Test run complete. If results look okay, proceed to Full Training.")

# %% [markdown]
# ### Cell 5: Full Training (The Real Deal)
# Using optimized multilingual embeddings and KeyBERT representation.

# 1. Initialize Embedding Model (Multilingual Paraphrase)
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Initialize Representation Model (Better topic labels)
representation_model = KeyBERTInspired()

# 3. Initialize BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
    min_topic_size=30,  # Avoid too many tiny topics
    nr_topics=50,  # Auto-reduce to a manageable number
    calculate_probabilities=False,  # Faster
    verbose=True,
)

# 4. Fit and Transform
docs = df["Text"].tolist()
topics, probs = topic_model.fit_transform(docs)

# Add to dataframe
df["Topic"] = topics

# %% [markdown]
# ### Cell 6: Topic Exploration
# Review the generated topics to decide on manual labels.
topic_info = topic_model.get_topic_info()
display(topic_info.head(10))

# %% [markdown]
# ### Cell 7: Manual Topic Mapping
# Based on your results, we are renaming the procedural "noise" and highlighting thematic topics.

topic_mapping = {
    -1: "General Noise/Procedural",
    0: "Political Banter & Interjections",
    1: "Ministerial Responses",
    2: "Honorifics & Greetings",
    3: "National Administration",
    4: "Floor Interruptions (Minta Laluan)",
    5: "Education & Schools",
    6: "Committee & Departmental Business",
    7: "Parliamentary Protocol",
    8: "Procedural Permission (Silakan)",
}

df["Topic_Label"] = df["Topic"].map(lambda x: topic_mapping.get(x, f"Topic {x}"))

# %% [markdown]
# ### Cell 8: Topics Over Time Visualization
print("Generating Topics over Time chart...")
topics_over_time = topic_model.topics_over_time(docs, df["Date"].tolist())
fig = topic_model.visualize_topics_over_time(topics_over_time)

# Save as HTML
fig.write_html(OUTPUT_CHART)
print(f"Chart saved to {OUTPUT_CHART}")

# %% [markdown]
# ### Cell 9: Export Final CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Success! Enriched data saved to {OUTPUT_CSV}")
# %% [markdown]
# ### Cell 10: Evaluation - Topic Coherence (ROBUST FIX)

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# 1. Tokenize and LOWERCASE (critical for matching)
tokenized_docs = [doc.lower().split() for doc in docs if isinstance(doc, str) and len(doc) > 10]

# 2. Create Dictionary
dictionary = corpora.Dictionary(tokenized_docs)

# 3. Get vocabulary set for fast lookup
vocab_set = set(dictionary.token2id.keys())

# 4. Get words for valid topics, FILTER to only words in our vocab
topic_info = topic_model.get_topic_info()
valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].values

topic_words = []
for t in valid_topics:
    words = [word.lower() for word, _ in topic_model.get_topic(t)]
    # Keep only words that exist in the dictionary
    filtered_words = [w for w in words if w in vocab_set]
    if len(filtered_words) >= 3:  # Need at least 3 words for coherence
        topic_words.append(filtered_words)

# 5. Calculate Coherence
if len(topic_words) < 2:
    print("Warning: Not enough valid topics for coherence calculation.")
else:
    coherence_model = CoherenceModel(
        topics=topic_words, 
        texts=tokenized_docs, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence Score (C_v): {coherence_score:.4f}")
    print(f"Evaluated on {len(topic_words)} topics.")