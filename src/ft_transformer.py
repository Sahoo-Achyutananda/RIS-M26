"""
FT-Transformer (Gorishniy et al., 2021, "Revisiting Deep Learning Models for
Tabular Data"): every numeric feature becomes a token via its own learned
linear embedding, a [CLS] token is prepended, and a Transformer encoder mixes
the tokens; the classification head reads the [CLS] output.

Simplified vs the reference: standard PyTorch pre-norm encoder layers (GELU FFN)
instead of ReGLU, one shared dropout rate. Quantile-normal input scaling and
AdamW follow the paper's recommendations.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer


class _FTTransformerNet(nn.Module):
    def __init__(self, n_features, d_token, n_blocks, n_heads, dropout):
        super().__init__()
        bound = d_token ** -0.5
        self.weight = nn.Parameter(torch.empty(n_features, d_token).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_features, d_token).uniform_(-bound, bound))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token).uniform_(-bound, bound))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4 // 3 * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias           # (batch, n_features, d_token)
        h = torch.cat([self.cls.expand(x.size(0), -1, -1), tokens], dim=1)
        return self.head(self.encoder(h)[:, 0]).squeeze(-1)          # logits


class FTTransformerClassifier:
    """sklearn-style wrapper: fit / predict_proba / predict, with early stopping on a validation split."""

    def __init__(self, d_token=64, n_blocks=3, n_heads=8, dropout=0.1, lr=1e-4, weight_decay=1e-5,
                 batch_size=512, max_epochs=40, patience=5, val_fraction=0.1, random_state=42):
        self.d_token, self.n_blocks, self.n_heads, self.dropout = d_token, n_blocks, n_heads, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.val_fraction = max_epochs, patience, val_fraction
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tensor(self, X):
        return torch.as_tensor(self.scaler.transform(np.asarray(X, dtype=np.float64)), dtype=torch.float32)

    def _loss(self, X_t, y_t, criterion):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(X_t), 4096):
                xb, yb = X_t[i:i + 4096].to(self.device), y_t[i:i + 4096].to(self.device)
                total += criterion(self.model(xb), yb).item() * len(xb)
        return total / len(X_t)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        gpu = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "none"
        print(f"[ft_transformer] device: {self.device} (GPU: {gpu}, CPU threads: {torch.get_num_threads()})", flush=True)
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float32)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state
        )
        self.scaler = QuantileTransformer(
            output_distribution="normal", n_quantiles=min(1000, len(X_tr)),
            subsample=min(len(X_tr), 200_000), random_state=self.random_state,
        ).fit(X_tr)
        X_tr_t, X_val_t = self._tensor(X_tr), self._tensor(X_val)
        y_tr_t, y_val_t = torch.as_tensor(y_tr), torch.as_tensor(y_val)

        self.model = _FTTransformerNet(X.shape[1], self.d_token, self.n_blocks, self.n_heads, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=self.batch_size, shuffle=True
        )

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(xb), yb).backward()
                optimizer.step()
            val_loss = self._loss(X_val_t, y_val_t, criterion)
            print(f"[ft_transformer] epoch {epoch + 1}: val_loss {val_loss:.4f}", flush=True)
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"[ft_transformer] early stop at epoch {epoch + 1}")
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        X_t = self._tensor(X)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), 4096):
                out.append(torch.sigmoid(self.model(X_t[i:i + 4096].to(self.device))).cpu().numpy())
        p1 = np.concatenate(out)
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
