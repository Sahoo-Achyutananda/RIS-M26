"""
Phase 2 (paper's terminology): Data preprocessing.

Loads a dataset's raw file(s), collapses the multi-class attack label into a
binary Normal(0)/Attack(1) label, cleans column names, drops irrelevant columns,
one-hot encodes categoricals, removes duplicates/NaNs/Infs, and writes a single
clean CSV to data/processed/<dataset>/dataset_processed.csv.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import get_config, DATA_PROCESSED

# Bookkeeping columns kept next to the features for Phase 2 (per-attack-type
# analysis, NSL-KDD's official split). They are never used as model inputs.
META_COLUMNS = ["attack_type", "source"]


def _clean_col(col: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip())


def load_raw(cfg: dict) -> pd.DataFrame:
    frames = []
    encoding = cfg.get("encoding", "utf-8")
    per_file_sample = cfg.get("per_file_sample_fraction")
    label_col = cfg["label_column"]
    for path in cfg["raw_files"]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw file not found: {path}")
        if cfg["has_header"]:
            df = pd.read_csv(path, low_memory=False, encoding=encoding)
        else:
            df = pd.read_csv(path, header=None, names=cfg["column_names"], low_memory=False, encoding=encoding)

        df.columns = [str(c).strip() for c in df.columns]  # CIC-IDS2017 headers have leading spaces

        if label_col in df.columns:
            # Some CIC files repeat the header row inside the data; drop those rows.
            df = df[df[label_col].astype(str).str.strip().str.lower() != "label"]

        if per_file_sample and label_col in df.columns:
            # Paper's approach: stratified 0.2% sample per raw file *before* merging,
            # so huge multi-file datasets (e.g. CSE-CIC-IDS2018) stay memory-feasible.
            df = df.groupby(label_col, group_keys=False).sample(frac=per_file_sample, random_state=42)
            print(f"[preprocess] Stratified-sampled {os.path.basename(path)} to {df.shape}")
        df["source"] = os.path.basename(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def preprocess(dataset: str) -> str:
    cfg = get_config(dataset)
    print(f"[preprocess] Loading raw data for '{dataset}'...")
    df = load_raw(cfg)
    print(f"[preprocess] Raw shape: {df.shape}")

    label_col = cfg["label_column"]
    benign_values = {v.strip().lower() for v in cfg["benign_values"]}

    # Read the attack name before the label column is turned into 0/1 (NSL-KDD's is called "label").
    attack_type = df[cfg.get("type_column", label_col)].astype(str).str.strip()
    df["label"] = df[label_col].astype(str).str.strip().str.lower().apply(
        lambda v: 0 if v in benign_values else 1
    )
    df["attack_type"] = attack_type.where(df["label"] == 1, "benign")
    if label_col != "label":
        df.drop(columns=[label_col], inplace=True)

    drop_cols = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    df.columns = [_clean_col(c) for c in df.columns]

    categorical_cols = [_clean_col(c) for c in cfg.get("categorical_columns", [])]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Low-cardinality categoricals the paper one-hot encodes (e.g. Protocol -> Protocol_0/6/17).
    onehot_cols = [c for c in (_clean_col(c) for c in cfg.get("onehot_columns", [])) if c in df.columns]
    if onehot_cols:
        for col in onehot_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).astype(str)
        df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, dtype=int)
        print(f"[preprocess] One-hot encoded {onehot_cols}")

    # Duplicates are judged on features + label only, as before the meta columns existed.
    data_cols = [c for c in df.columns if c not in META_COLUMNS]
    if cfg.get("drop_duplicates", True):
        before = len(df)
        df.drop_duplicates(subset=data_cols, inplace=True)
        print(f"[preprocess] Dropped {before - len(df)} duplicate rows.")
    else:
        print(f"[preprocess] Keeping duplicate rows ({int(df.duplicated(subset=data_cols).sum())} present).")

    numeric_cols = df.columns.drop(["label"] + META_COLUMNS)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Must come after to_numeric: CIC files contain the text "Infinity", which
    # to_numeric turns into a real inf.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    sample_fraction = cfg.get("sample_fraction")
    if sample_fraction:
        df = df.groupby("label", group_keys=False).sample(frac=sample_fraction, random_state=42)
        print(f"[preprocess] Stratified-sampled to {sample_fraction:.4%} -> shape {df.shape}")

    out_dir = os.path.join(DATA_PROCESSED, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_processed.csv")
    df.to_csv(out_path, index=False)

    print(f"[preprocess] Final shape: {df.shape}")
    print(f"[preprocess] Label distribution:\n{df['label'].value_counts()}")
    print(f"[preprocess] Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    preprocess(args.dataset)
