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

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = self.scaler.fit_transform(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)

        n_features = X.shape[1]
        self.model = _LSTMNet()
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
                preds.append(self.model(X_t[i:i + batch_size]).numpy())
        p1 = np.concatenate(preds)
        p0 = 1 - p1
        return np.stack([p0, p1], axis=1)


class StackedHybrid:
    """RF + XGBoost + LSTM base learners -> trained Logistic Regression meta-learner."""

    def __init__(self, n_folds: int = 3, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.rf_final = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
        self.xgb_final = XGBClassifier(
            n_estimators=200, random_state=random_state, n_jobs=1,
            eval_metric="logloss", use_label_encoder=False,
        )
        self.lstm_final = TorchLSTMClassifier(random_state=random_state)
        self.meta_learner = LogisticRegression(max_iter=1000)

    def _new_base_learners(self):
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)
        xgb = XGBClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=1,
            eval_metric="logloss", use_label_encoder=False,
        )
        lstm = TorchLSTMClassifier(random_state=self.random_state)
        return rf, xgb, lstm

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        oof_preds = np.zeros((n, 3))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"[stacked_hybrid] Fold {fold + 1}/{self.n_folds} - generating out-of-fold predictions...")
            rf, xgb, lstm = self._new_base_learners()
            rf.fit(X[train_idx], y[train_idx])
            xgb.fit(X[train_idx], y[train_idx])
            lstm.fit(X[train_idx], y[train_idx])

            oof_preds[val_idx, 0] = rf.predict_proba(X[val_idx])[:, 1]
            oof_preds[val_idx, 1] = xgb.predict_proba(X[val_idx])[:, 1]
            oof_preds[val_idx, 2] = lstm.predict_proba(X[val_idx])[:, 1]

            # Repeatedly constructing RF/XGBoost/LSTM models in one long-lived
            # process fragments Windows' heap over many folds; force a cleanup
            # pass between folds to keep the process stable.
            del rf, xgb, lstm
            gc.collect()

        print("[stacked_hybrid] Training meta-learner on out-of-fold predictions...")
        self.meta_learner.fit(oof_preds, y)

        print("[stacked_hybrid] Refitting base learners on full training set...")
        self.rf_final.fit(X, y)
        self.xgb_final.fit(X, y)
        self.lstm_final.fit(X, y)
        return self

    def _base_predictions(self, X):
        X = np.asarray(X)
        p_rf = self.rf_final.predict_proba(X)[:, 1]
        p_xgb = self.xgb_final.predict_proba(X)[:, 1]
        p_lstm = self.lstm_final.predict_proba(X)[:, 1]
        return np.stack([p_rf, p_xgb, p_lstm], axis=1)

    def predict_proba(self, X):
        base_preds = self._base_predictions(X)
        return self.meta_learner.predict_proba(base_preds)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
