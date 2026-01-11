# %% [markdown]
# Metadata Merger
# This script merges the parsed Hansard speeches with the Party Map metadata.

# %%
import pandas as pd
import os


# %%
def merge_metadata():
    master_path = os.path.join("data", "hansard_master.csv")
    pmap_path = os.path.join("data", "party_map.csv")
    output_path = os.path.join("data", "hansard_master_final.csv")

    if not os.path.exists(master_path) or not os.path.exists(pmap_path):
        print("Error: Missing input files.")
        return

    # Load data
    df_master = pd.read_csv(master_path)
    df_pmap = pd.read_csv(pmap_path)

    # Clean Constituency strings for matching
    df_master["Constituency_Clean"] = (
        df_master["Constituency"].fillna("N/A").str.strip().str.upper()
    )
    df_pmap["Constituency_Clean"] = df_pmap["Constituency"].str.strip().str.upper()

    # Drop speaker clean from pmap to avoid confusion, we keep the original speaker name from hansard
    df_pmap["Speaker_Key"] = df_pmap["Speaker_Clean"].str.lower().str.strip()

    # Create two mapping dictionaries
    constituency_to_party = df_pmap.set_index("Constituency_Clean")["Party"].to_dict()
    name_to_party = df_pmap.set_index("Speaker_Key")["Party"].to_dict()
    name_to_constituency = df_pmap.set_index("Speaker_Key")["Constituency"].to_dict()

    def get_metadata(row):
        speaker = str(row["Speaker"]).lower()
        constituency = str(row["Constituency_Clean"])

        # 1. Try matching by Constituency first (most reliable)
        if constituency in constituency_to_party:
            return (
                constituency_to_party[constituency],
                df_pmap[df_pmap["Constituency_Clean"] == constituency][
                    "Constituency"
                ].iloc[0],
            )

        # 2. Try matching by Speaker Name (fuzzy-ish)
        # Remove common titles
        titles = [
            "tuan ",
            "puan ",
            "dato' ",
            "datuk ",
            "seri ",
            "dr. ",
            "tan sri ",
            "haji ",
            "ustaz ",
            "yang berhormat ",
        ]
        clean_speaker = speaker
        for title in titles:
            clean_speaker = clean_speaker.replace(title, "")
        clean_speaker = clean_speaker.strip()

        if clean_speaker in name_to_party:
            return name_to_party[clean_speaker], name_to_constituency[clean_speaker]

        # 3. Handle official roles
        if "yang di-pertua" in speaker or "pengerusi" in speaker:
            return "OFFICIAL", "N/A"

        return None, row["Constituency"]

    print("Merging data...")
    # Apply matching logic
    results = df_master.apply(get_metadata, axis=1)
    df_final = df_master.copy()
    df_final["Party"] = [r[0] for r in results]
    df_final["Constituency"] = [r[1] for r in results]

    # Define Coalition Mapping (2018-2022 Context)
    coalition_map = {
        "PKR": "PH",
        "DAP": "PH",
        "AMANAH": "PH",
        "PPBM": "PH",
        "WARISAN": "PH-Allied",
        "UPKO": "PH-Allied",
        "UMNO": "BN",
        "MCA": "BN",
        "MIC": "BN",
        "PBRS": "BN",
        "PAS": "PAS/GS",
        "PBB": "GPS",
        "PRS": "GPS",
        "PDP": "GPS",
        "SUPP": "GPS",
        "PBS": "GBS/GRS",
        "STAR": "GBS/GRS",
        "PBRS": "GBS/GRS",
        "BEBAS": "Independent",
    }

    df_final["Coalition"] = (
        df_final["Party"].map(coalition_map).fillna("Unknown/Official")
    )

    # Handle the "Speaker of the House" or "Tuan Yang di-Pertua" which won't have a constituency/party in the map
    df_final.loc[
        df_final["Speaker"].str.contains("Yang di-Pertua", na=False), "Party"
    ] = "OFFICIAL"
    df_final.loc[
        df_final["Speaker"].str.contains("Yang di-Pertua", na=False), "Coalition"
    ] = "OFFICIAL"

    # Cleanup
    df_final = df_final.drop(columns=["Constituency_Clean"])

    # Save
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Stats
    total_rows = len(df_final)
    matched_rows = df_final[df_final["Party"].notna()].shape[0]
    unmatched_rows = df_final[df_final["Party"].isna()].shape[0]

    print("\n" + "=" * 50)
    print("MERGE SUMMARY")
    print("=" * 50)
    print(f"Total speeches: {total_rows}")
    print(f"Matched with Party: {matched_rows} ({(matched_rows/total_rows)*100:.1f}%)")
    print(
        f"Unmatched/Official: {unmatched_rows} ({(unmatched_rows/total_rows)*100:.1f}%)"
    )
    print(f"Final data saved to: {output_path}")
    print("=" * 50 + "\n")

    if unmatched_rows > 0:
        print("Top 5 Unmatched Speakers:")
        print(df_final[df_final["Party"].isna()]["Speaker"].value_counts().head(5))


if __name__ == "__main__":
    merge_metadata()
