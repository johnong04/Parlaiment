# %% [markdown]
# Hansard PDF Parser
# This script extracts parliamentary speeches from Malaysian Hansard PDFs.

# %%
import pdfplumber
import pandas as pd
import re
import os
import glob
from datetime import datetime

# %% [markdown]
# ### Configuration & Regex Patterns

# %%
# Regex for Speaker and Constituency
# Group 1: Speaker Name
# Group 2: Constituency (Optional)
SPEAKER_REGEX = r"^([A-Z][\w\s\'\.\-@]{2,60})(?:\s*\[(.*?)\])?\s*:"

# Noise Filters
# 1. Header dates (e.g., DR.15.10.2018)
HEADER_DATE_REGEX = r"^DR\.\d{1,2}\.\d{1,2}\.\d{4}"
# 2. Time markers (e.g., ■1020)
TIME_MARKER_REGEX = r"^■\d+"
# 3. Page numbers (standalone digits)
PAGE_NUMBER_REGEX = r"^\d+$"
# 4. Section titles (All caps, length > 5, common keywords)
SECTION_KEYWORDS = [
    "JAWAPAN",
    "PERTANYAAN",
    "USUL",
    "MENGANGKAT",
    "SUMPAH",
    "PEMASYHURAN",
    "KANDUNGAN",
    "AKTA-AKTA",
]

# Blacklisted words in Speaker names (to avoid titles/sentences ending in colons)
SPEAKER_BLACKLIST = [
    "MELULUSKAN",
    "BERIKUT",
    "TERSEBUT",
    "DITERBITKAN",
    "DIBAWA",
    "WAKTU",
    "URUSAN",
    "USUL",
    "RANG UNDANG-UNDANG",
]

# %% [markdown]
# ### Helper Functions


# %%
def is_noise(line):
    line = line.strip()
    if not line:
        return True

    # 1. Check for standalone symbols or very short lines
    if len(line) < 3:
        return True

    # 2. Match standard noise patterns
    if re.match(HEADER_DATE_REGEX, line):
        return True
    if re.match(TIME_MARKER_REGEX, line):
        return True
    if re.match(PAGE_NUMBER_REGEX, line):
        return True

    # 3. Check for all-caps section titles or specific page headers
    line_upper = line.upper()
    if line_upper.isupper() and len(line) > 5:
        if any(kw in line_upper for kw in SECTION_KEYWORDS):
            return True

    # 4. Common procedural headers/footers
    if any(
        line.startswith(day)
        for day in ["Bil.", "Isnin", "Selasa", "Rabu", "Khamis", "Jumaat"]
    ):
        return True

    return False


def is_valid_speaker(speaker_name):
    # Check length
    if len(speaker_name) > 70 or len(speaker_name) < 3:
        return False
    # Check blacklist
    speaker_upper = speaker_name.upper()
    if any(word in speaker_upper for word in SPEAKER_BLACKLIST):
        return False
    return True


def extract_date_from_filename(filename):
    # Example: DR-15102018.pdf -> 15-10-2018
    match = re.search(r"DR-(\d{2})(\d{2})(\d{4})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "Unknown"


# %% [markdown]
# ### Main Extraction Logic


# %%
def parse_hansard(pdf_path):
    filename = os.path.basename(pdf_path)
    file_date = extract_date_from_filename(filename)

    speeches = []
    current_speaker = None
    current_constituency = None
    current_text = []

    with pdfplumber.open(pdf_path) as pdf:
        # Note: We don't skip pages explicitly, the noise filter handles admin lists
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")
            for line in lines:
                if is_noise(line):
                    continue

                # Check for new speaker
                speaker_match = re.match(SPEAKER_REGEX, line)
                if speaker_match:
                    speaker_name = speaker_match.group(1).strip()
                    if is_valid_speaker(speaker_name):
                        # Save previous speech if exists
                        if current_speaker:
                            speeches.append(
                                {
                                    "Date": file_date,
                                    "Speaker": current_speaker,
                                    "Constituency": current_constituency,
                                    "Text": " ".join(current_text).strip(),
                                }
                            )

                        # Start new speech
                        current_speaker = speaker_name
                        cons = (
                            speaker_match.group(2).strip()
                            if speaker_match.group(2)
                            else "N/A"
                        )
                        # Clean up constituency (remove brackets if they were partially captured)
                        current_constituency = (
                            cons.replace("[", "").replace("]", "").strip()
                        )

                        # Check if there's text after the colon on the same line
                        text_after_colon = line[speaker_match.end() :].strip()
                        current_text = [text_after_colon] if text_after_colon else []
                        continue  # Move to next line to avoid appending the speaker line twice

                # Append to current speaker's text if we are in a speech
                if current_speaker:
                    current_text.append(line.strip())

        # Save the last speech
        if current_speaker:
            speeches.append(
                {
                    "Date": file_date,
                    "Speaker": current_speaker,
                    "Constituency": current_constituency,
                    "Text": " ".join(current_text).strip(),
                }
            )

    return speeches


# %% [markdown]
# ### Execution and Verification


# %%
def run_batch_processing(target_pdfs):
    all_data = []
    summary = []

    for pdf_path in target_pdfs:
        filename = os.path.basename(pdf_path)
        try:
            speeches = parse_hansard(pdf_path)
            all_data.extend(speeches)
            summary.append({"Filename": filename, "Speeches": len(speeches)})
            print(f"DONE: {filename} ({len(speeches)} speeches)")
        except Exception as e:
            print(f"ERROR: Failed to parse {filename}: {e}")
            continue

    if not all_data:
        print("No speeches extracted.")
        return None

    df = pd.DataFrame(all_data)

    # Convert Date to datetime for proper sorting
    df["dt"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values("dt").drop(columns=["dt"])

    return df, summary


if __name__ == "__main__":
    # Path configuration
    RAW_DIR = os.path.join("data", "raw_hansards")
    OUTPUT_FILE = os.path.join("data", "hansard_master.csv")

    # Get all PDF files
    pdf_files = glob.glob(os.path.join(RAW_DIR, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR}")
    else:
        print(f"Found {len(pdf_files)} PDF files. Starting batch processing...")

        df_master, processing_summary = run_batch_processing(pdf_files)

        if df_master is not None:
            # Save final master CSV
            df_master.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

            print("\n" + "=" * 50)
            print("FINAL BATCH SUMMARY")
            print("=" * 50)
            print(f"Total PDFs processed: {len(processing_summary)}")
            print(f"Total speeches extracted: {len(df_master)}")
            print(f"Unique speakers: {df_master['Speaker'].nunique()}")
            print(f"Output saved to: {OUTPUT_FILE}")
            print("=" * 50 + "\n")

            # Print first few rows for spot check
            print("SPOT CHECK: First 5 rows of master data")
            print(df_master.head(5)[["Date", "Speaker", "Constituency"]])
