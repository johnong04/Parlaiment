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

**Next Step:** Proceed to Phase 2 (Topic Modeling) using BERTopic on Google Colab.
