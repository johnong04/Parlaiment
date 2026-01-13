# ParlAIment: Malaysian Hansard Analytics Dashboard 🏛️

**Live Demo:** [parlaiment.streamlit.app](https://parlaiment.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badge_indicator.svg)](https://parlaiment.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Intelligence • Accountability • Governance**

ParlAIment is an advanced analytics platform designed to bridge the **"Hansard Gap"** in Malaysian parliamentary proceedings. By leveraging Natural Language Processing (NLP), the project transforms raw Hansard PDF records into actionable insights, specifically focusing on identifying "Evasiveness" in political speech during critical debates (2018–2022).

---

## 🌟 Key Features

### 📈 The Pulse (Temporal Analysis)
Visualize how parliamentary focus shifts over time. Track specific topics like **Economy**, **Healthcare**, and **Education** through interactive line charts, and overlay the **Evasiveness Index** to see when accountability was most challenged.

### 📊 The Stance (Accountability Dashboard)
A high-level view of political sentiment.
- **Party Sentiment Profiles**: Distribution of Pro, Con, Neutral, and Evasive stances across major parties.
- **Evasiveness Heatmap**: Identifies which topics (e.g., 1MDB, Taxation) triggered the highest rates of non-answers from different coalitions.

### 🔍 The Evidence Explorer
A transparent "Ground Truth" engine. Search over **16,000 unique speech turns** by keyword, MP, or party. Every speech is presented with its AI-classified stance and dynamic keyword highlighting.

### 👤 MP Insights
Individualized profiles for Members of Parliament.
- **Evasive Index**: A metric-driven approach to accountability.
- **Focus Areas**: Automated extraction of an MP's primary policy interests using topic chips.
- **Stance Distribution**: Visualizing an MP's parliamentary persona.

---

## 🧠 AI Methodology

The platform utilizes a state-of-the-art dual-model pipeline:

1.  **Topic Discovery (BERTopic)**:
    - Uses `paraphrase-multilingual-MiniLM-L12-v2` embeddings.
    - Unsupervised clustering to isolate substantive policy debates from procedural "noise."
    - Achieved a **Topic Coherence (Cv) of 0.3909**.

2.  **Stance & Evasiveness Classification (XLM-RoBERTa)**:
    - Fine-tuned `xlm-roberta-base` on a specialized dataset.
    - Optimized to distinguish between "Neutral" replies and "Evasive" non-answers.
    - **69.1% Accuracy** on multi-class political stance detection.

---

## 🛠️ Technical Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Custom Navy & Gold branding)
- **Visualizations**: [Plotly](https://plotly.com/python/)
- **Data Processing**: Pandas, Regex State-Machine Parser
- **PDF Extraction**: [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Machine Learning**: PyTorch, HuggingFace Transformers, BERTopic

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- [Virtual Environment](https://docs.python.org/3/library/venv.html) (Recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/USER/Parlaiment.git
   cd Parlaiment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: Main Streamlit application entry point.
- `src/parser.py`: State-machine PDF parser for Hansard records.
- `src/config.py`: Global theme, color mappings, and topic labels.
- `data/hansard_final_analyzed.csv`: The final processed and AI-enriched dataset.
- `assets/`: Custom CSS and branding assets.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  <i>Developed for a Final Year Project in Data Science.</i><br>
  <b>Intelligence • Accountability • Governance</b>
</p>
