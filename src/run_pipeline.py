"""
Orchestrator: runs the full pipeline for one dataset end-to-end.

    python run_pipeline.py --dataset nsl_kdd

Steps: preprocess -> feature selection -> train/test split -> SMOTE-Tomek
balance (train only) -> train baselines + paper's averaging-hybrid + our
stacked-hybrid -> evaluate everything -> SHAP explainability -> save results.
"""

import argparse
import os
import time

# Windows-only workaround for the memory errors seen on the dev laptop. Limiting
# threads everywhere would make torch/XGBoost single-threaded on Kaggle (Linux).
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from sklearn.model_selection import train_test_split

from config import get_config, DATA_PROCESSED, RESULTS_DIR
from preprocess import preprocess
from feature_selection import select_features
from balance import balance_train_set
from baselines import get_baseline_models
from hybrid_baseline import AveragingHybrid
from stacked_hybrid import StackedHybrid
from ft_transformer import FTTransformerClassifier
from evaluate import compute_metrics, save_confusion_matrix, save_roc_curve
from explain_shap import explain_model


def run(dataset: str, skip_prep: bool = False, mode: str = "reproduce", only=None):
    cfg = get_config(dataset)
    out_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not skip_prep:
        preprocess(dataset)
        select_features(dataset)

    final_path = os.path.join(DATA_PROCESSED, dataset, "dataset_final.csv")
    df = pd.read_csv(final_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"[run_pipeline] Train: {X_train.shape}, Test: {X_test.shape}")

    X_train_bal, y_train_bal = balance_train_set(X_train, y_train)

    # Convert to plain numpy arrays before handing off to any model. XGBoost's
    # pandas-columnar ingestion path (_ref_data_from_columnar) segfaults with
    # this numpy 2.x build when given a DataFrame directly -- numpy arrays
    # avoid that code path entirely for every model, not just XGBoost.
    feature_names = X_train_bal.columns.tolist()
    X_train_bal = X_train_bal.to_numpy()
    y_train_bal = y_train_bal.to_numpy()
    X_test_arr = X_test.to_numpy()
    y_test = y_test.to_numpy()

    results = []
    # reproduce: paper's baselines + averaging hybrid only.
    # novel: paper's averaging hybrid (as the reference point) + our models.
    if mode == "reproduce":
        models = get_baseline_models()
        models["Averaging Hybrid (paper's method)"] = AveragingHybrid()
    else:
        models = {
            "Averaging Hybrid (paper's method)": AveragingHybrid(),
            "Stacked Hybrid (ours, novel)": StackedHybrid(),
            "FT-Transformer (ours, novel)": FTTransformerClassifier(),
        }
    if only:
        wanted = [w.strip().lower() for w in only.split(",") if w.strip()]
        models = {k: v for k, v in models.items() if any(w in k.lower() for w in wanted)}
        if not models:
            raise SystemExit(f"--models {only!r} matched no model")
        print(f"[run_pipeline] Running only: {list(models)}")

    xgb_for_shap = None

    for name, model in models.items():
        print(f"\n[run_pipeline] === Training: {name} ===")
        start = time.time()
        model.fit(X_train_bal, y_train_bal)
        elapsed = time.time() - start

        y_pred = model.predict(X_test_arr)
        y_score = model.predict_proba(X_test_arr)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics["Model"] = name
        metrics["TrainSeconds"] = round(elapsed, 2)
        results.append(metrics)
        print(f"[run_pipeline] {name}: {metrics}")

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")
        save_confusion_matrix(
            y_test, y_pred,
            os.path.join(out_dir, mode, f"confusion_matrix_{safe_name}.png"),
            f"Confusion Matrix - {name} ({cfg['display_name']})",
        )
        if y_score is not None:
            save_roc_curve(
                y_test, y_score,
                os.path.join(out_dir, mode, f"roc_curve_{safe_name}.png"),
                f"ROC Curve - {name} ({cfg['display_name']})",
            )

        if name == "XGBoost":
            xgb_for_shap = model

    results_df = pd.DataFrame(results).set_index("Model")
    suffix = "" if not only else "_" + "_".join(w.strip().lower().replace(" ", "") for w in only.split(",") if w.strip())
    results_csv = os.path.join(out_dir, f"metrics_{mode}{suffix}.csv")
    results_df.to_csv(results_csv)
    print(f"\n[run_pipeline] Saved metrics to {results_csv}")
    print(results_df.round(4).to_string())

    if xgb_for_shap is not None:
        sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
        explain_model(
            xgb_for_shap, sample,
            os.path.join(out_dir, mode, "shap_summary.png"),
            f"SHAP Feature Importance ({cfg['display_name']})",
        )

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--skip-prep", action="store_true", help="Reuse existing processed/feature-selected data")
    parser.add_argument("--mode", choices=["reproduce", "novel"], default="reproduce",
                        help="reproduce = paper's methods only; novel = our models vs the paper's hybrid")
    parser.add_argument("--models", default=None,
                        help="comma-separated name fragments to run a subset, e.g. 'ft' or 'averaging,stacked'")
    args = parser.parse_args()
    run(args.dataset, skip_prep=args.skip_prep, mode=args.mode, only=args.models)
