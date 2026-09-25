"""
Replication of the paper's proposed method: a hybrid ensemble of Random Forest
and XGBoost, combined by averaging their predicted probabilities (equivalent to
what the paper achieves with the `combo` package's `average()` aggregator).
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np


class AveragingHybrid:
    """RF + XGBoost, combined via simple probability averaging (paper's method)."""

    def __init__(self, random_state: int = 42):
        self.rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=4)
        self.xgb = XGBClassifier(
            n_estimators=200, random_state=random_state, n_jobs=4,
            eval_metric="logloss", use_label_encoder=False,
        )

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        return self

    def predict_proba(self, X):
        p_rf = self.rf.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        return (p_rf + p_xgb) / 2.0

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
