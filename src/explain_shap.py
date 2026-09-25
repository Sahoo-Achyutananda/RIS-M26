"""
SHAP explainability, replicating the paper's Section 5.2: shows which features
drive the model's decisions (top features), using the XGBoost component of the
hybrid as the explained model (TreeExplainer is exact and fast for tree models).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def explain_model(xgb_model, X_sample, out_path: str, title: str, max_display: int = 15):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[explain_shap] Saved SHAP summary plot to {out_path}")
