"""
NOVEL CONTRIBUTION: Stacked Hybrid Ensemble.

The paper's "hybrid" model is just probability-averaging of RF + XGBoost (see
hybrid_baseline.py). Their own Conclusion & Future Works section (Saini et al.,
2023, p.26) states they intend to "combine machine learning and deep learning
algorithms... such as LSTM" in future work. We implement that combination now,
and replace the naive averaging aggregator with a *trained* meta-learner
(stacking) fitted on out-of-fold base-learner predictions, which is a strictly
more expressive combiner than a fixed 50/50 average.

Base learners: Random Forest, XGBoost, a lightweight LSTM (per-sample feature
vector treated as a length-n_features sequence with a single channel).
Meta-learner: Logistic Regression on the 3 base learners' out-of-fold
probabilities (standard stacking, avoids leakage).
"""

import gc
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return torch.sigmoid(out).squeeze(-1)


class TorchLSTMClassifier:
    """Minimal sklearn-style wrapper around a small LSTM for tabular flow features."""

    def __init__(self, epochs: int = 12, batch_size: int = 256, lr: float = 1e-3, random_state: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = self.scaler.fit_transform(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)

        n_features = X.shape[1]
        self.model = _LSTMNet().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        X_t = torch.from_numpy(X).unsqueeze(-1)  # (n, seq_len, 1)
        y_t = torch.from_numpy(y)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            print(f"[stacked_hybrid][LSTM] epoch {epoch + 1}/{self.epochs} - loss {total_loss / len(dataset):.4f}")
        return self

    def predict_proba(self, X, batch_size: int = 512):
        X = self.scaler.transform(np.asarray(X, dtype=np.float32))
        X_t = torch.from_numpy(X).unsqueeze(-1)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                preds.append(self.model(X_t[i:i + batch_size].to(self.device)).cpu().numpy())
        p1 = np.concatenate(preds)
        p0 = 1 - p1
        return np.stack([p0, p1], axis=1)


class StackedHybrid:
    """RF + XGBoost + a deep base learner -> trained Logistic Regression meta-learner.

    deep="lstm": the LSTM above (the paper's future-work suggestion).
    deep="ft":   FT-Transformer, a tabular deep model that does not assume a feature order.
    """

    def __init__(self, deep: str = "lstm", n_folds: int = 3, random_state: int = 42):
        assert deep in ("lstm", "ft")
        self.deep = deep
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_jobs = 1 if os.name == "nt" else -1  # single-threaded only on the Windows dev laptop
        self.meta_learner = LogisticRegression(max_iter=1000)

    def _base_learners(self, n_trees):
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=self.random_state, n_jobs=self.n_jobs)
        xgb = XGBClassifier(n_estimators=n_trees, random_state=self.random_state, n_jobs=self.n_jobs,
                            eval_metric="logloss")
        if self.deep == "lstm":
            deep = TorchLSTMClassifier(random_state=self.random_state)
        else:
            from ft_transformer import FTTransformerClassifier
            deep = FTTransformerClassifier(random_state=self.random_state)
        return [rf, xgb, deep]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        oof_preds = np.zeros((len(y), 3))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"[stacked_hybrid:{self.deep}] Fold {fold + 1}/{self.n_folds} - out-of-fold predictions...", flush=True)
            learners = self._base_learners(n_trees=100)
            for j, model in enumerate(learners):
                model.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, j] = model.predict_proba(X[val_idx])[:, 1]
            del learners
            gc.collect()

        print(f"[stacked_hybrid:{self.deep}] Training meta-learner on out-of-fold predictions...", flush=True)
        self.meta_learner.fit(oof_preds, y)
        print(f"[stacked_hybrid:{self.deep}] meta-learner weights (rf, xgb, {self.deep}): "
              f"{np.round(self.meta_learner.coef_[0], 3).tolist()}", flush=True)

        print(f"[stacked_hybrid:{self.deep}] Refitting base learners on full training set...", flush=True)
        self.final_learners = self._base_learners(n_trees=200)
        for model in self.final_learners:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        base = np.stack([m.predict_proba(X)[:, 1] for m in self.final_learners], axis=1)
        return self.meta_learner.predict_proba(base)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
