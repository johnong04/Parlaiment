# Developer's Log: Malaysian Hansard Analytics

## Phase 1: Data Ingestion (Complete)

**Status:** 16,050 speeches extracted and enriched with Party/Coalition metadata.

### 1. Technical Decisions & Logic

- **State-Machine Parsing:** Switched from page-based to line-based parsing. This handles speeches spanning multiple pages by "staying open" until the next speaker is detected.
- **Regex Refinement:** Used `^([A-Z][\w\s\'\.\-@]{2,60})(?:\s*\[(.*?)\])?\s*:` to safely capture names and constituencies while avoiding accidental sentence matching.
- **Fuzzy Metadata Merge:** Implemented a two-tier match. Primary key: **Constituency** (unique/stable). Fallback: **Core Name** (stripping titles like _Dato, Dr., Tan Sri_) to handle inconsistent naming in PDFs.
- **Neutral Handling:** Tagged parliamentary moderators as `OFFICIAL` to ensure political sentiment analysis remains unbiased.

### 2. Key Data Insights (For Presentation)

- **Volume:** Extracted **16,050 speech turns** from 29 PDFs.
- **The "Marathon" Session:** `DR-19112018.pdf` was the most active file, containing a staggering **1,111 speeches** in a single day.
- **Speaker Diversity:** Identified **298 unique speakers**, covering almost the entire Dewan Rakyat membership.
- **Data Quality:** Achieved a **93.1% match rate** for political parties. The remaining 6.9% are mostly anonymous interjections ("Seorang Ahli"), which is expected in Hansard records.

### 3. Implementation Hurdles

- **PDF Noise:** TOC entries (e.g., "RANG UNDANG-UNDANG") triggered false speaker detections. Resolved by implementing a custom `SPEAKER_BLACKLIST`.
- **Resource Intensity:** Large PDFs (100+ pages) required significant processing time (~1hr for full batch).
- **Cross-Platform:** Adopted a "User-AI Delegation" model for environment setup (venv/pip) to avoid Windows-specific terminal errors.

---

## Phase 3: Dashboard & Brand Identity (Complete)

**Status:** Functional "ParlAIment" dashboard deployed with 5 analytical modules.

### 1. Brand & Aesthetic Decisions

- **The "ParlAIment" Identity:** Created a unique brand identity emphasizing the "AI" component of parliamentary analytics.
- **Premium UI:** Adopted a "Deep Dark" luxury theme (Navy `#0A1628` / Gold `#D4AF37`) to align with luxury brand aesthetics, avoiding "AI slop" visuals.
- **Custom Header:** Developed a custom HTML/CSS header with glowing typography and hidden standard Streamlit elements for a bespoke look.

### 2. Analytical Feature Engineering

- **Evasiveness Index:** Derived a custom metric ($Evasive / Total$) to normalize accountability scores across speakers with varying speech volumes.
- **Dynamic Filtering:** Implemented a robust "Noise Shield" that automatically filters out 24+ procedural topics (e.g., _[Bangun]_, _[Ketawa]_, _Time Management_) to ensure the dashboard only reflects substantive governance.
- **Topic Mapping:** Manually refined the top 20 thematic clusters into human-friendly governance categories (e.g., _Budget & Finance_, _1MDB & Asset Recovery_).

### 3. User Experience (UX) Innovations

- **Visual Intelligence:** Replaced standard lists with "Thematic Topic Chips" and "MP Player Cards" for high-impact visual storytelling.
- **Evidence Highlighting:** Integrated real-time regex highlighting in search results, allowing users to instantly spot keywords within 16,000+ speeches.
- **Stable Heatmap:** Fixed data-type mismatch bugs to ensure consistent "Party vs. Topic" accountability grids.

### 4. Presentation Readiness

- **Academic Validation:** Included a dedicated "Methodology" tab detailing the BERTopic and XLM-RoBERTa pipeline to satisfy DS undergraduate requirements.
- **Speaker Hygiene:** Applied advanced regex and length-based filters to the MP list to remove parser artifacts while retaining Ministerial titles.
- **Functional Demo:** Verified the "Drill-down" flow: Trends -> Stance Analysis -> Individual MP Insights -> Raw Evidence.

### 1. Topic Modeling (BERTopic) - Implementation Details

- **Script:** Created `src/train_topics.py` designed for Google Colab/Local execution.
- **Model Selection:** Switched to `paraphrase-multilingual-MiniLM-L12-v2` for the embedding layer to properly handle Malay/English Hansard text.
- **Refinement:** Used `KeyBERTInspired` representation model to generate cleaner, more descriptive topic labels compared to raw c-TF-IDF.
- **Workflow:** Implemented a two-stage training process (1,000-row test run followed by full 16,000-row training).
- **Visualization:** Integrated `topics_over_time` to generate `data/topic_chart.html`.

### 2. Topic Modeling - Findings & Presentation Content

- **Coherence Score (Cv):** **0.3909** (Evaluated on 48 topics). This provides quantitative validation that the clusters are semantically consistent.
- **Key Discovery:** Topic 0 (Political Banter) correctly isolated heckling and interjections (Keywords: _Tajuddin, Shahidan, Rayer_).
- **Thematic Topics:** Topic 5 (Education) and Topic 10 (Budget/Finance) show clear temporal spikes corresponding to parliamentary cycles.
- **Data Quality:** Outlier topic (-1) successfully captured ~37% of "procedural noise," keeping the thematic topics clean.

### 3. Stance Modeling (XLM-RoBERTa) - Implementation Details

- **Model:** `xlm-roberta-base` fine-tuned on 600 "Silver Labels" generated by Manus AI.
- **Accuracy:** Achieved **69.1%** overall accuracy and **0.68 Macro F1-Score**.
- **Optimization:** Implemented `WeightedTrainer` to handle class imbalance, specifically boosting "Pro" and "Evasive" performance.
- **Inference:** Processed all 16,050 rows in under 15 minutes using batch inference.

### 4. Key Insights for Presentation (Phase 2 Findings)

- **The "Banter" Baseline:** 37% of parliamentary speech was identified as procedural noise or political banter, successfully isolated by the model.
- **Evasiveness Trends:** Early analysis shows that Evasiveness counts are significantly higher during Budget Debates (Topic 10) compared to general procedural sessions.
- **Coalition Dynamics:** Preliminary data indicates a distinct stance profile difference between Government (higher "Pro" and "Evasive") and Opposition (higher "Con") coalitions.
- **Scientific Validation:** Confusion Matrix confirms that the model distinguishes between "Neutral" and "Evasive" with 70% precision, minimizing false accusations of evasiveness.

### 4. Technical Stabilization & Documentation (2026-01-11)

- **UI State Fix:** Resolved a critical bug where tab interactions caused a full page reset to the first tab. Solution involved moving filter generation logic outside the tab context to ensure stable widget dependencies and adding explicit `key` parameters to all interactive widgets to force Streamlit state persistence.
- **Cache Optimization:** Increased `st.cache_data` TTL to 1 hour to prevent background reloads during live demonstrations.
- **Documentation:** Authored a professional `README.md` incorporating the project's premium branding, technical specs (BERTopic/XLM-RoBERTa metrics), and installation guide to ensure the project is presentation-ready.
- **Task Management:** Updated `tasks.md` to reflect 100% completion of the functional prototype phases.
- **Deployment:** Successfully deployed the application to Streamlit Cloud at [parlaiment.streamlit.app](https://parlaiment.streamlit.app/). Updated `README.md` to feature the live link prominently for presentation and accessibility.

---

# Presentation Master Guide (CRISP-DM Structure)

## 1. Business Understanding (The "Hansard Gap")

- **Problem:** Parliamentary Hansards are thousands of pages of unstructured PDF text, making accountability checks manually impossible.
- **Objective:** Bridge the "Hansard Gap" by automating the detection of **Evasiveness**—where ministers provide non-answers to critical policy questions.
- **Goal:** Build a functional prototype dashboard that isolates substantive policy from procedural banter.

## 2. Data Understanding

- **Source:** 29 Hansard PDF records (2018-2022) from the Malaysian Dewan Rakyat.
- **Volume:** 16,050 raw speech turns extracted.
- **Complexity:** Multilingual text (Malay/English code-switching), nested honorifics, and significant procedural "noise" (approx. 78% of data).

## 3. Data Preparation (The State-Machine Parser)

- **Engine:** Custom line-based **State-Machine Parser** using `pdfplumber`.
- **Feature Engineering:**
  - **Regex Matching:** `^([A-Z][\w\s\'\.\-@]{2,60})(?:\s*\[(.*?)\])?\s*:` to isolate Speaker vs. Constituency.
  - **Noise Shield:** Implemented a `SPEAKER_BLACKLIST` and `SECTION_KEYWORDS` filter to strip TOCs, time markers, and procedural headers.
- **Metadata Enrichment:**
  - **Two-Tier Fuzzy Matching:** Merged speeches with political party/coalition lists.
  - **Logic:** Primary match on _Constituency_ (stable); fallback match on _Core Name_ (stripping titles like 'Dato' or 'Tan Sri').

## 4. Modeling Phase I: Unsupervised Topic Modeling (BERTopic)

- **Architecture:** BERTopic Pipeline.
- **Embedding Layer:** `paraphrase-multilingual-MiniLM-L12-v2` (Handles Malay/English semantic similarity).
- **Dimensionality Reduction:** UMAP (to 5 components).
- **Clustering:** HDBSCAN (`min_cluster_size=30`).
- **Representation:** `KeyBERTInspired` model (provides cleaner, more descriptive topic labels than standard c-TF-IDF).
- **Refinement:** Manually mapped 50 raw topics into human-friendly categories (e.g., _1MDB & Asset Recovery_, _Budget & Finance_).

## 5. Modeling Phase II: Supervised Stance Classification (XLM-RoBERTa)

- **Model:** `xlm-roberta-base` (Cross-lingual model optimized for low-resource languages like Malay).
- **Training Strategy:**
  - **Dataset:** 600 "Silver Labels" generated via LLM-augmented grounding.
  - **Class Imbalance:** Custom `WeightedTrainer` using `nn.CrossEntropyLoss(weight=[2.0, 1.5, 1.0, 2.0])` to prioritize the "Evasive" and "Pro" minority classes.
- **Hyperparameters:**
  - Batch Size: 16 | Epochs: 10 | Learning Rate: 3e-5 | Weight Decay: 0.01.
  - Strategy: `load_best_model_at_end` based on Macro F1-Score.

## 6. Evaluation & Results

- **Topic Coherence (Cv):** **0.3909** (Evaluated on thematic clusters). Successfully isolated "Political Banter" (Topic 0) from policy.
- **Classification Accuracy:** **69.1%** overall.
- **Confusion Matrix Insight:** The model excels at distinguishing "Neutral" from "Evasive" with 70% precision, ensuring high-fidelity accountability tracking.
- **Loss Curves:** Stable convergence observed; Early Stopping prevented overfitting to the small labeled set.

## 7. Deployment (Streamlit)

- **Stability:** Implemented explicit `st.cache_data` (1hr TTL) and unique widget keys to prevent state loss during reruns.
- **UI Logic:** Dashboard uses a "Thematic Filter" (`NOISE_TOPICS`) to automatically hide 12,500 procedural rows, highlighting only the **3,562 substantive speeches** for the demo.
