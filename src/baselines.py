"""
Representative baseline models (see plan's Scope Adjustment): standalone RF,
standalone XGBoost, Decision Tree, Logistic Regression, and a simple MLP as the
deep-learning baseline -- enough to show the hybrid beats individual learners
without re-running the paper's full 13-model comparison.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_baseline_models(random_state: int = 42) -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=4),
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=random_state, n_jobs=4,
            eval_metric="logloss", use_label_encoder=False,
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=random_state),
    }
