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
# Based on the final dashboard configuration for consistent human-friendly labeling.

topic_mapping = {
    -1: "General Noise",
    0: "Political Banter",
    1: "Ministerial Procedures",
    2: "Honorifics & Greetings",
    3: "National Administration",
    4: "Procedural Interjections",
    5: "Education & Schools",
    6: "Committee Business",
    7: "Greetings & Acknowledgments",
    8: "Procedural Interjections",
    9: "General Negations",
    10: "Budget & Finance",
    11: "Time Management",
    12: "General Discussions",
    13: "Urban & Town Planning",
    14: "Islamic Affairs & Syariah",
    15: "Healthcare & Hospitals",
    16: "Regional (KL/Selayang)",
    17: "Procedural (Standing)",
    20: "Water & Infrastructure",
    21: "Regional (Kota Tinggi)",
    22: "Housing & Development",
    24: "Taxation & GST",
    26: "Regional (Kelantan)",
    27: "Regional (Cameron Highlands)",
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

df["Topic_Label"] = df["Topic"].map(lambda x: topic_mapping.get(x, f"Topic {x}"))

# %% [markdown]
# ### Cell 8: Topics Over Time Visualization
print("Generating Topics over Time chart...")
topics_over_time = topic_model.topics_over_time(docs, df["Date"].tolist())

# Apply labels to the model first to ensure they appear in the chart
topic_model.set_topic_labels(topic_mapping)
fig = topic_model.visualize_topics_over_time(topics_over_time, custom_labels=True)

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
tokenized_docs = [
    doc.lower().split() for doc in docs if isinstance(doc, str) and len(doc) > 10
]

# 2. Create Dictionary
dictionary = corpora.Dictionary(tokenized_docs)

# 3. Get vocabulary set for fast lookup
vocab_set = set(dictionary.token2id.keys())

# 4. Get words for valid topics, FILTER to only words in our vocab
topic_info = topic_model.get_topic_info()
valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].values

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
        topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence Score (C_v): {coherence_score:.4f}")
    print(f"Evaluated on {len(topic_words)} topics.")

# %% [markdown]
# ### Cell 11: Scientific Visuals for Presentation
# These visualizations provide scientific proof of the clustering quality for your FYP.

# 1. Intertopic Distance Map (Slide 5: Methodology)
# Shows how distinct the topics are from each other.
fig_distance = topic_model.visualize_topics()
fig_distance.write_html("data/intertopic_distance.html")
display(fig_distance)

# 2. Topic Hierarchical Clustering (Optional, but looks very "academic")
# Shows how topics relate to each other in a tree structure.
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html("data/topic_hierarchy.html")
display(fig_hierarchy)

# 3. Topic Word Scores (Shows how clean the clusters are)
# Shows the top words for the most frequent topics.
fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
fig_barchart.write_html("data/topic_word_scores.html")
display(fig_barchart)

# %% [markdown]
# ### Cell 11: Scientific Visuals with HUMAN LABELS
# This version uses your 'topic_mapping' names instead of IDs.

# 1. Apply your human names to the model
# (Ensure topic_mapping is the dictionary you defined in Cell 7)
topic_model.set_topic_labels(topic_mapping)

# 2. Intertopic Distance Map (Slide 5: Methodology)
# Note: set 'custom_labels=True' to use your names!
fig_distance = topic_model.visualize_topics(custom_labels=True)
fig_distance.write_html("data/intertopic_distance.html")
display(fig_distance)

# 3. Topic Word Scores (Slide 5: Methodology)
# Note: set 'custom_labels=True' here too!
fig_barchart = topic_model.visualize_barchart(top_n_topics=10, custom_labels=True)
fig_barchart.write_html("data/topic_word_scores.html")
display(fig_barchart)

# 4. Topic Hierarchical Clustering
# Note: Use 'custom_labels=True'
fig_hierarchy = topic_model.visualize_hierarchy(custom_labels=True)
fig_hierarchy.write_html("data/topic_hierarchy.html")
display(fig_hierarchy)
