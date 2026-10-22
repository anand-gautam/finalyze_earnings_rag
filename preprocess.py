# preprocess.py
import argparse
import json
import os

import numpy as np
import pandas as pd



# def parse_structured(content):
#     """Split structured_content into prepared_remarks and qa sections."""
#     if content is None:
#         return "", ""
#
#     if isinstance(content, np.ndarray):
#         segments = content.tolist()
#     elif isinstance(content, list):
#         segments = content
#     elif isinstance(content, str) and content.strip():
#         try:
#             segments = json.loads(content)
#         except Exception:
#             return content, ""
#     else:
#         return "", ""
#
#     if not isinstance(segments, list):
#         return "", ""
#
#     prepared, qa = [], []
#     in_qa = False
#
#     for seg in segments:
#         if isinstance(seg, dict):
#             speaker = str(seg.get("speaker", "") or "").lower()
#             text = str(seg.get("text", "") or "").strip()
#         else:
#             continue
#
#         if not text:
#             continue
#
#         if any(k in speaker for k in ["operator", "host"]):
#             continue
#
#         if "analyst" in speaker or "question" in speaker or "q:" in speaker:
#             in_qa = True
#         elif any(k in speaker for k in ["ceo", "cfo", "president", "chief", "executive", "management"]):
#             in_qa = False
#
#         if in_qa:
#             qa.append(f"{seg.get('speaker', 'Analyst')}: {text}")
#         else:
#             prepared.append(f"{seg.get('speaker', 'Management')}: {text}")
#
#     return " ".join(prepared), " ".join(qa)


def parse_structured(content):
    """
    Robust split of structured_content into prepared_remarks and qa sections.
    Handles raw names, unexpected dictionaries, and structural text layouts safely.
    """
    if content is None:
        return "", ""

    if isinstance(content, np.ndarray):
        segments = content.tolist()
    elif isinstance(content, list):
        segments = content
    elif isinstance(content, str) and content.strip():
        try:
            segments = json.loads(content)
        except Exception:
            # If it's a raw unstructured text string, split it cleanly down the middle
            text_str = content.strip()
            midpoint = len(text_str) // 2
            return text_str[:midpoint], text_str[midpoint:]
    else:
        return "", ""

    if not isinstance(segments, list):
        return "", ""

    prepared, qa = [], []

    # SYSTEM STATE: Start inside prepared remarks by default
    in_qa = False

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        speaker = str(seg.get("speaker", "") or "").lower().strip()
        text = str(seg.get("text", "") or "").lower().strip()

        if not text:
            continue

        # Ignore clean operator setup lines
        if any(k in speaker for k in ["operator", "host", "instructions"]):
            continue

        # 1. EXPLICIT QA TRACKERS (Analyst questions or explicit labels)
        if any(k in speaker for k in ["analyst", "question", "q:", "q & a", "q&a", "q-and-a"]):
            in_qa = True
        # 2. EXPLICIT EXECUTIVE TRACKERS
        elif any(k in speaker for k in ["ceo", "cfo", "president", "chief", "executive", "management", "director"]):
            in_qa = False
        # 3. ROBUST FALLBACK STATE: If an executive responds to a question by name,
        # it belongs in the Q&A text category block, not the opening presentation deck.
        # So we leave the active state tracking unmodified instead of dropping text fields.

        if in_qa:
            qa.append(f"{seg.get('speaker', 'Analyst')}: {text}")
        else:
            prepared.append(f"{seg.get('speaker', 'Management')}: {text}")

    # If the structural parsing returned completely empty fields, run emergency split logic
    prep_out = " ".join(prepared).strip()
    qa_out = " ".join(qa).strip()

    return prep_out, qa_out


def main():
    parser = argparse.ArgumentParser(description="Preprocess earnings call transcripts.")
    parser.add_argument("--sample", type=int, default=500, help="Number of transcripts to sample (0 for all). Default: 500.")
    args = parser.parse_args()

    # if pd is None:
    #     raise RuntimeError("pandas is required to preprocess transcripts.")

    raw_path = "data/raw/transcripts.parquet"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"'{raw_path}' not found. Run load_data.py first.")

    print(f"Loading '{raw_path}'...")
    df = pd.read_parquet(raw_path)

    if args.sample > 0 and len(df) > args.sample:
        print(f"Sampling {args.sample} transcripts for processing...")
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    print("Parsing structured content into prepared remarks and Q&A...")
    parsed = df["structured_content"].apply(parse_structured)
    df["prepared_remarks"] = parsed.apply(lambda x: x[0])
    df["qa_section"] = parsed.apply(lambda x: x[1])

    # Fallback to 'content' column if prepared_remarks parsing yielded empty text
    empty_mask = df["prepared_remarks"].str.len() <= 100
    if "content" in df.columns:
        df.loc[empty_mask, "prepared_remarks"] = df.loc[empty_mask, "content"].astype(str)

    df = df[df["prepared_remarks"].str.len() > 100].reset_index(drop=True)
    print(f"[OK] Usable transcripts extracted: {len(df)}")

    out_path = "data/raw/processed.parquet"
    df.to_parquet(out_path)
    print(f"[OK] Saved processed dataset to '{out_path}'")


if __name__ == "__main__":
    main()