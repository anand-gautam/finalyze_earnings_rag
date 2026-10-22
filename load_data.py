# load_data.py
import argparse
import os
from pathlib import Path

from datasets import load_dataset
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description="Load S&P 500 earnings transcripts dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download dataset.")
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    target_path = "data/raw/transcripts.parquet"

    if os.path.exists(target_path) and not args.force:
        print(f"[OK] Found existing transcript dataset at '{target_path}'.")
        df = pd.read_parquet(target_path)
        print(f"  Shape: {df.shape}")
        if "symbol" in df.columns:
            print(f"  Unique companies: {df['symbol'].nunique()}")
        if "year" in df.columns:
            print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
        return

    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is required to download the transcript data.")

    print("Downloading 'kurry/sp500_earnings_transcripts' from HuggingFace...")
    ds = load_dataset("kurry/sp500_earnings_transcripts")
    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    print(f"[OK] Dataset downloaded. Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    if "symbol" in df.columns:
        print(f"  Unique companies: {df['symbol'].nunique()}")
    if "year" in df.columns:
        print(f"  Year range: {df['year'].min()}-{df['year'].max()}")

    df.to_parquet(target_path)
    print(f"[OK] Saved to {target_path}")


if __name__ == "__main__":
    main()