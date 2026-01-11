# %% [markdown]
# Hansard PDF Parser
# This script extracts parliamentary speeches from Malaysian Hansard PDFs.

# %%
import pdfplumber
import pandas as pd
import re
import os
from datetime import datetime

# %% [markdown]
# ### Configuration & Regex Patterns

# %%
# Regex for Speaker and Constituency
# Group 1: Speaker Name
# Group 2: Constituency (Optional)
SPEAKER_REGEX = r'^([A-Z][\w\s\'\.\-@]{2,60})(?:\s*\[(.*?)\])?\s*:'

# Noise Filters
# 1. Header dates (e.g., DR.15.10.2018)
HEADER_DATE_REGEX = r'^DR\.\d{1,2}\.\d{1,2}\.\d{4}'
# 2. Time markers (e.g., ■1020)
TIME_MARKER_REGEX = r'^■\d+'
# 3. Page numbers (standalone digits)
PAGE_NUMBER_REGEX = r'^\d+$'
# 4. Section titles (All caps, length > 5, common keywords)
SECTION_KEYWORDS = ['JAWAPAN', 'PERTANYAAN', 'USUL', 'MENGANGKAT', 'SUMPAH', 'PEMASYHURAN', 'KANDUNGAN', 'AKTA-AKTA']

# Blacklisted words in Speaker names (to avoid titles/sentences ending in colons)
SPEAKER_BLACKLIST = ['MELULUSKAN', 'BERIKUT', 'TERSEBUT', 'DITERBITKAN', 'DIBAWA', 'WAKTU', 'URUSAN', 'USUL']

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
    if any(line.startswith(day) for day in ['Bil.', 'Isnin', 'Selasa', 'Rabu', 'Khamis', 'Jumaat']):
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
    match = re.search(r'DR-(\d{2})(\d{2})(\d{4})', filename)
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
    
    print(f"Processing {filename}...")
    
    with pdfplumber.open(pdf_path) as pdf:
        # Note: We don't skip pages explicitly, the noise filter handles admin lists
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
                
            lines = text.split('\n')
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
                            speeches.append({
                                'Date': file_date,
                                'Speaker': current_speaker,
                                'Constituency': current_constituency,
                                'Text': " ".join(current_text).strip()
                            })
                        
                        # Start new speech
                        current_speaker = speaker_name
                        cons = speaker_match.group(2).strip() if speaker_match.group(2) else "N/A"
                        # Clean up constituency (remove brackets if they were partially captured)
                        current_constituency = cons.replace('[', '').replace(']', '').strip()
                        
                        # Check if there's text after the colon on the same line
                        text_after_colon = line[speaker_match.end():].strip()
                        current_text = [text_after_colon] if text_after_colon else []
                        continue # Move to next line to avoid appending the speaker line twice
                
                # Append to current speaker's text if we are in a speech
                if current_speaker:
                    current_text.append(line.strip())
        
        # Save the last speech
        if current_speaker:
            speeches.append({
                'Date': file_date,
                'Speaker': current_speaker,
                'Constituency': current_constituency,
                'Text': " ".join(current_text).strip()
            })
            
    return speeches

# %% [markdown]
# ### Execution and Verification

# %%
if __name__ == "__main__":
    target_pdf = "DR-15102018.pdf"
    
    if os.path.exists(target_pdf):
        all_speeches = parse_hansard(target_pdf)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_speeches)
        
        # Save to CSV
        output_path = os.path.join("data", "parsed_speeches.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"Total speeches extracted: {len(df)}")
        print(f"Unique speakers found: {df['Speaker'].nunique()}")
        print(f"Data saved to: {output_path}")
        print("="*50 + "\n")
        
        # Verification: Print first 5 speeches
        print("VERIFICATION: First 5 Speeches")
        print("-" * 30)
        for i, row in df.head(5).iterrows():
            print(f"SPEAKER: {row['Speaker']} [{row['Constituency']}]")
            print(f"TEXT: {row['Text'][:200]}...")
            print("-" * 30)
    else:
        print(f"Error: {target_pdf} not found.")
