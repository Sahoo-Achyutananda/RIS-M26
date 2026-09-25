"""
Phase: class balancing via SMOTE-Tomek, matching the paper's approach of
producing a clean 50/50 Normal/Attack split for training.
Only applied to the training split -- never to the held-out test set.
"""

from imblearn.combine import SMOTETomek


def balance_train_set(X_train, y_train, random_state: int = 42):
    print(f"[balance] Before SMOTE-Tomek: {y_train.value_counts().to_dict()}")
    smt = SMOTETomek(random_state=random_state, n_jobs=4)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    print(f"[balance] After SMOTE-Tomek: {y_res.value_counts().to_dict()}")
    return X_res, y_res
