"""
Phase 3: Feature selection, replicating the paper's two-step approach:
  1. Pearson correlation pruning - drop one feature from any pair correlated > 0.9
  2. Mutual information ranking - keep the top-k most informative remaining features
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

from config import get_config, DATA_PROCESSED


def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"[feature_selection] Dropping {len(to_drop)} highly-correlated features (>{threshold}).")
    return X.drop(columns=to_drop)


def select_features(dataset: str) -> str:
    cfg = get_config(dataset)
    in_path = os.path.join(DATA_PROCESSED, dataset, "dataset_processed.csv")
    df = pd.read_csv(in_path)
    print(f"[feature_selection] Loaded {in_path}, shape {df.shape}")

    y = df["label"]
    X = df.drop(columns=["label"])

    var_selector = VarianceThreshold(threshold=0.0)
    X_arr = var_selector.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns[var_selector.get_support()])
    print(f"[feature_selection] After removing zero-variance features: {X.shape[1]} features")

    X = drop_correlated_features(X, threshold=0.9)
    print(f"[feature_selection] After correlation pruning: {X.shape[1]} features")

    k = min(cfg["n_features_to_select"], X.shape[1])
    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_k_idx = np.argsort(mi_scores)[::-1][:k]
    top_features = X.columns[top_k_idx].tolist()
    print(f"[feature_selection] Top {k} features by mutual information:\n{top_features}")

    final_df = X[top_features].copy()
    final_df["label"] = y.values

    out_path = os.path.join(DATA_PROCESSED, dataset, "dataset_final.csv")
    final_df.to_csv(out_path, index=False)
    print(f"[feature_selection] Saved {out_path}, shape {final_df.shape}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    select_features(args.dataset)
