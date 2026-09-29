"""
Phase 2: how much of the ~99% is real once the test is fair?

Runs the checks promised at the end of Phase 1. Each experiment writes CSVs to
results/<dataset>/phase2/ (the cross-dataset test writes to results/cross_dataset/phase2/).

  protocols  The same models under three orders of steps, repeated over several seeds:
               paper  - pick columns on all data, balance all data, then split (the study)
               phase1 - pick columns on all data, split, balance the learning part (our Phase 1)
               fair   - split first; pick columns and balance with the learning part only
             The fair runs use the larger model set and feed the significance tests.
  official   NSL-KDD only: learn on KDDTrain+, test on KDDTest+.
  holdout    Leave one attack family out of learning; is it still caught at test time?
  shortcut   Remove the most important columns and retrain; does accuracy survive?
  cross      Learn on CIC-IDS2017 and test on CSE-CIC-IDS2018, and the reverse.

    python phase2.py --dataset nsl_kdd --experiments protocols,official,holdout,shortcut --seeds 5
    python phase2.py --cross

Needs dataset_processed.csv written by the current preprocess.py (with the
attack_type and source columns); re-run preprocessing if it is older.
"""

import argparse
import os
import time

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from balance import balance_train_set
from config import CIC2017_TO_2018, DATA_PROCESSED, RESULTS_DIR, get_config
from evaluate import compute_metrics
from feature_selection import choose_features
from preprocess import META_COLUMNS

N_JOBS = 1 if os.name == "nt" else -1
HYBRID = "Averaging Hybrid (paper)"
TEST_SIZE = 0.3

# ---------------------------------------------------------------- attack families

NSL_FAMILIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm",
            "processtable", "mailbomb"],
    "Probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L": ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient",
            "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
            "named", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"],
}
NSL_TYPE_TO_FAMILY = {t: fam for fam, types in NSL_FAMILIES.items() for t in types}


def family_of(dataset: str, attack_type: str) -> str:
    t = str(attack_type).strip().lower()
    if t in ("benign", "normal"):
        return "Benign"
    if dataset == "nsl_kdd":
        return NSL_TYPE_TO_FAMILY.get(t, "Other")
    if dataset.startswith("unsw"):
        return {"backdoors": "Backdoor"}.get(t, t.capitalize())
    # CIC-IDS2017 / CSE-CIC-IDS2018 labels, e.g. "DoS Hulk", "DDOS attack-HOIC", "Brute Force -XSS"
    if t.startswith("ddos"):
        return "DDoS"
    if t.startswith("dos"):
        return "DoS"
    for key, fam in [("heartbleed", "Heartbleed"), ("portscan", "PortScan"), ("infilt", "Infiltration"),
                     ("web", "Web attack"), ("xss", "Web attack"), ("sql", "Web attack"),
                     ("patator", "Brute force"), ("brute", "Brute force")]:
        if key in t:
            return fam
    return "Bot" if t.startswith("bot") else "Other"

# ------------------------------------------------------------------------- models


class AveragedPair:
    """The paper's hybrid built from an already-fitted RF and XGBoost (mean of probabilities).

    Same models and settings as hybrid_baseline.AveragingHybrid, without training them twice.
    """

    def __init__(self, rf, xgb):
        self.rf, self.xgb = rf, xgb

    def predict_proba(self, X):
        return (self.rf.predict_proba(X) + self.xgb.predict_proba(X)) / 2.0

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(seed: int, full: bool, deep: bool) -> dict:
    """core = RF, XGBoost (+ the hybrid made from them); full adds DT, LightGBM, CatBoost."""
    models = {}
    if full:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=seed)
    models["Random Forest"] = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS)
    models["XGBoost"] = XGBClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS, eval_metric="logloss")
    if full:
        try:
            from lightgbm import LGBMClassifier
            models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS, verbose=-1)
        except ImportError:
            print("[phase2] lightgbm not installed, skipping LightGBM", flush=True)
        try:
            from catboost import CatBoostClassifier
            models["CatBoost"] = CatBoostClassifier(iterations=300, random_seed=seed, verbose=0,
                                                    thread_count=N_JOBS, allow_writing_files=False)
        except ImportError:
            print("[phase2] catboost not installed, skipping CatBoost", flush=True)
    if deep:
        from ft_transformer import FTTransformerClassifier
        from stacked_hybrid import StackedHybrid
        models["FT-Transformer"] = FTTransformerClassifier(random_state=seed)
        models["Stacked RF+XGB+FT"] = StackedHybrid(deep="ft", random_state=seed)
    return models


def train_and_predict(models: dict, X_train, y_train, X_test) -> dict:
    """Fit every model; returns name -> (hard predictions, attack probabilities, seconds)."""
    out = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        out[name] = (np.asarray(model.predict(X_test)).astype(int).ravel(),
                     model.predict_proba(X_test)[:, 1], time.time() - start)
        print(f"[phase2]   {name}: {out[name][2]:.1f}s", flush=True)
    if "Random Forest" in models and "XGBoost" in models:
        hybrid = AveragedPair(models["Random Forest"], models["XGBoost"])
        out[HYBRID] = (hybrid.predict(X_test), hybrid.predict_proba(X_test)[:, 1],
                       out["Random Forest"][2] + out["XGBoost"][2])
    return out


def metric_rows(preds: dict, y_test, **tags) -> list:
    rows = []
    for name, (pred, score, secs) in preds.items():
        m = compute_metrics(y_test, pred, score)
        rows.append({**tags, "Model": name, **m, "TrainSeconds": round(secs, 1), "NTest": len(y_test)})
    return rows


def family_rows(preds: dict, test_df: pd.DataFrame, **tags) -> list:
    """Share of each attack family's test rows flagged as attack (for Benign: false-alarm rate)."""
    rows = []
    fam = test_df["family"].to_numpy()
    for name, (pred, _, _) in preds.items():
        for f in sorted(set(fam)):
            mask = fam == f
            rows.append({**tags, "Model": name, "Family": f, "N": int(mask.sum()),
                         "FlaggedAsAttack": float(pred[mask].mean())})
    return rows

# --------------------------------------------------------------------------- data


def load(dataset: str) -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, dataset, "dataset_processed.csv")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} has no {missing} column(s): re-run preprocess.py for {dataset} first.")
    df["family"] = df["attack_type"].map(lambda t: family_of(dataset, t))
    print(f"[phase2] Loaded {dataset}: {df.shape}; families:\n{df['family'].value_counts().to_string()}", flush=True)
    return df


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ["label", "family"] + META_COLUMNS]


def as_float(X):
    # SMOTE casts synthetic rows back to the input dtype, so integer columns would be
    # rounded. Phase 1 read its features from a float CSV; match that.
    return X.astype("float64")


def prepare(train_df, test_df, feats, seed):
    """Balance the learning part only; return numpy arrays ready for the models."""
    X_bal, y_bal = balance_train_set(as_float(train_df[feats]), train_df["label"], random_state=seed)
    return X_bal.to_numpy(), y_bal.to_numpy(), as_float(test_df[feats]).to_numpy(), test_df["label"].to_numpy()


def fair_features(train_df, k):
    return choose_features(train_df[feature_columns(train_df)], train_df["label"], k)


def save(rows_or_df, out_dir, name):
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"[phase2] Saved {path} ({len(df)} rows)", flush=True)
    return df

# -------------------------------------------------------------------- statistics


def mcnemar_p(correct_a, correct_b):
    """Exact McNemar test: do two models disagree in a lopsided way on the same test rows?"""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def corrected_ttest_p(diffs, test_frac=TEST_SIZE):
    """Nadeau & Bengio's corrected resampled t-test over repeated random splits."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    t = d.mean() / np.sqrt((1 / n + test_frac / (1 - test_frac)) * d.var(ddof=1))
    return float(2 * stats.t.sf(abs(t), n - 1))


def compare_to_hybrid(correct: dict, acc: dict, seeds) -> list:
    """For each model: hybrid's accuracy edge per seed, McNemar per seed, corrected t-test overall."""
    rows = []
    others = sorted({m for (_, m) in correct if m != HYBRID})
    for m in others:
        diffs = []
        for s in seeds:
            b, c, p = mcnemar_p(correct[(s, HYBRID)], correct[(s, m)])
            diffs.append(acc[(s, HYBRID)] - acc[(s, m)])
            rows.append({"Model": m, "Seed": s, "HybridOnlyRight": b, "ModelOnlyRight": c,
                         "McNemarP": p, "HybridMinusModelAcc": diffs[-1]})
        rows.append({"Model": m, "Seed": "all", "HybridMinusModelAcc": float(np.mean(diffs)),
                     "CorrectedTTestP": corrected_ttest_p(diffs)})
    return rows

# -------------------------------------------------------------------- experiments


def exp_protocols(df, seeds, k, deep, out_dir):
    feats_all = choose_features(df[feature_columns(df)], df["label"], k)  # the study's (leaky) choice
    rows, fam_rows, correct, acc = [], [], {}, {}
    for seed in seeds:
        print(f"\n[phase2] protocols | seed {seed}", flush=True)
        train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=df["label"])

        print("[phase2] paper: balance all data, then split", flush=True)
        X_bal, y_bal = balance_train_set(as_float(df[feats_all]), df["label"], random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(X_bal.to_numpy(), y_bal.to_numpy(), test_size=TEST_SIZE,
                                              random_state=seed, stratify=y_bal)
        preds = train_and_predict(make_models(seed, full=False, deep=False), Xtr, ytr, Xte)
        rows += metric_rows(preds, yte, Protocol="paper", Seed=seed)

        print("[phase2] phase1: columns from all data, balance learning part", flush=True)
        Xtr, ytr, Xte, yte = prepare(train_df, test_df, feats_all, seed)
        preds = train_and_predict(make_models(seed, full=False, deep=False), Xtr, ytr, Xte)
        rows += metric_rows(preds, yte, Protocol="phase1", Seed=seed)

        print("[phase2] fair: columns and balancing from the learning part only", flush=True)
        feats = fair_features(train_df, k)
        Xtr, ytr, Xte, yte = prepare(train_df, test_df, feats, seed)
        preds = train_and_predict(make_models(seed, full=True, deep=deep), Xtr, ytr, Xte)
        rows += metric_rows(preds, yte, Protocol="fair", Seed=seed)
        fam_rows += family_rows(preds, test_df, Seed=seed)
        for name, (pred, _, _) in preds.items():
            correct[(seed, name)] = pred == yte
            acc[(seed, name)] = float(np.mean(pred == yte))

    res = save(rows, out_dir, "protocols.csv")
    summary = res.groupby(["Protocol", "Model"])[["Accuracy", "F1", "FPR", "FNR", "AUC"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    save(summary.reset_index(), out_dir, "protocols_summary.csv")
    fam = save(fam_rows, out_dir, "family_recall.csv")
    save(fam.groupby(["Model", "Family"])[["N", "FlaggedAsAttack"]].mean().reset_index(), out_dir,
         "family_recall_summary.csv")
    save(compare_to_hybrid(correct, acc, seeds), out_dir, "significance.csv")


def exp_official(df, seeds, k, deep, out_dir):
    train_df = df[df["source"].str.startswith("KDDTrain")]
    test_df = df[df["source"].str.startswith("KDDTest")]
    print(f"\n[phase2] official | learn {len(train_df)} rows (KDDTrain+), test {len(test_df)} rows (KDDTest+)", flush=True)
    feats = fair_features(train_df, k)
    seen = set(train_df["attack_type"])
    rows, type_rows = [], []
    for seed in seeds:
        print(f"[phase2] official | seed {seed}", flush=True)
        Xtr, ytr, Xte, yte = prepare(train_df, test_df, feats, seed)
        preds = train_and_predict(make_models(seed, full=True, deep=deep), Xtr, ytr, Xte)
        rows += metric_rows(preds, yte, Split="official", Seed=seed)
        types = test_df["attack_type"].to_numpy()
        for name, (pred, _, _) in preds.items():
            for t in sorted(set(types)):
                mask = types == t
                type_rows.append({"Seed": seed, "Model": name, "AttackType": t, "Family": family_of("nsl_kdd", t),
                                  "SeenInTraining": t in seen, "N": int(mask.sum()),
                                  "FlaggedAsAttack": float(pred[mask].mean())})
    save(rows, out_dir, "official.csv")
    types = save(type_rows, out_dir, "official_types.csv")
    attacks = types[types["AttackType"] != "benign"].copy()
    attacks["Caught"] = attacks["N"] * attacks["FlaggedAsAttack"]
    agg = attacks.groupby(["Model", "SeenInTraining"])[["N", "Caught"]].sum().reset_index()
    agg["Recall"] = agg["Caught"] / agg["N"]  # both summed over seeds
    agg["N"] = agg["N"] / len(seeds)
    save(agg.drop(columns="Caught"), out_dir, "official_seen_vs_new.csv")


def exp_holdout(df, seed, k, out_dir, min_rows=20):
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=df["label"])
    families = [f for f in sorted(set(df["family"]) - {"Benign"})
                if (test_df["family"] == f).sum() >= min_rows and (train_df["family"] == f).sum() >= min_rows]
    print(f"\n[phase2] holdout | families: {families}", flush=True)

    def recall_on(preds, fam_test, fam):
        mask = fam_test == fam
        return {name: float(pred[mask].mean()) for name, (pred, _, _) in preds.items()}

    fam_test = test_df["family"].to_numpy()
    Xtr, ytr, Xte, yte = prepare(train_df, test_df, fair_features(train_df, k), seed)
    base = train_and_predict(make_models(seed, full=False, deep=False), Xtr, ytr, Xte)
    rows = []
    for fam in families:
        print(f"[phase2] holdout | hiding {fam}", flush=True)
        reduced = train_df[train_df["family"] != fam]
        Xtr, ytr, Xte, yte = prepare(reduced, test_df, fair_features(reduced, k), seed)
        preds = train_and_predict(make_models(seed, full=False, deep=False), Xtr, ytr, Xte)
        seen_r, unseen_r = recall_on(base, fam_test, fam), recall_on(preds, fam_test, fam)
        benign = fam_test == "Benign"
        for name in preds:
            rows.append({"Family": fam, "Model": name, "NTest": int((fam_test == fam).sum()),
                         "RecallWhenSeen": seen_r[name], "RecallWhenHidden": unseen_r[name],
                         "FPRWhenHidden": float(preds[name][0][benign].mean())})
    save(rows, out_dir, "holdout.csv")


def exp_shortcut(df, seed, k, out_dir, drops=(0, 1, 2, 3, 5, 10)):
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=df["label"])
    feats = fair_features(train_df, k)
    Xtr, ytr, Xte, yte = prepare(train_df, test_df, feats, seed)
    ranker = XGBClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS, eval_metric="logloss").fit(Xtr, ytr)
    order = [feats[i] for i in np.argsort(ranker.feature_importances_)[::-1]]
    print(f"\n[phase2] shortcut | most important columns: {order[:10]}", flush=True)
    rows = []
    for n_drop in drops:
        if n_drop >= len(feats):
            break
        keep = [i for i, f in enumerate(feats) if f not in order[:n_drop]]
        print(f"[phase2] shortcut | dropping top {n_drop}", flush=True)
        preds = train_and_predict(make_models(seed, full=False, deep=False), Xtr[:, keep], ytr, Xte[:, keep])
        rows += metric_rows(preds, yte, Dropped=n_drop, DroppedColumns="; ".join(order[:n_drop]))
    save(rows, out_dir, "shortcut.csv")


def exp_cross(seed, k, out_dir):
    a = load("cic_ids2017").rename(columns=CIC2017_TO_2018)
    b = load("cse_cic_ids2018")
    common = sorted(set(feature_columns(a)) & set(feature_columns(b)))
    print(f"\n[phase2] cross | {len(common)} shared columns", flush=True)
    rows, fam_rows = [], []
    for (src_name, src), (dst_name, dst) in [(("CIC-IDS2017", a), ("CSE-CIC-IDS2018", b)),
                                             (("CSE-CIC-IDS2018", b), ("CIC-IDS2017", a))]:
        print(f"[phase2] cross | learn {src_name} -> test {dst_name}", flush=True)
        train_df, test_df = train_test_split(src, test_size=TEST_SIZE, random_state=seed, stratify=src["label"])
        feats = choose_features(train_df[common], train_df["label"], k)
        Xtr, ytr, Xte, yte = prepare(train_df, test_df, feats, seed)
        models = make_models(seed, full=True, deep=False)
        same = train_and_predict(models, Xtr, ytr, Xte)  # also fits the models
        X_dst = as_float(dst[feats]).to_numpy()
        fitted = dict(models, **{HYBRID: AveragedPair(models["Random Forest"], models["XGBoost"])})
        other = {name: (np.asarray(m.predict(X_dst)).astype(int).ravel(), m.predict_proba(X_dst)[:, 1], same[name][2])
                 for name, m in fitted.items()}
        for test_name, t, preds in [(f"{src_name} (same dataset)", test_df, same), (dst_name, dst, other)]:
            rows += metric_rows(preds, t["label"].to_numpy(), Train=src_name, Test=test_name)
            fam_rows += family_rows(preds, t, Train=src_name, Test=test_name)
    save(rows, out_dir, "cross.csv")
    save(fam_rows, out_dir, "cross_family_recall.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset key from config.py (not needed with --cross)")
    parser.add_argument("--experiments", default="protocols,holdout,shortcut",
                        help="comma-separated: protocols, official (NSL-KDD only), holdout, shortcut")
    parser.add_argument("--seeds", type=int, default=5, help="number of repeated random splits")
    parser.add_argument("--deep", action="store_true", help="also run FT-Transformer and the stacked FT team (slow; GPU)")
    parser.add_argument("--cross", action="store_true", help="run the CIC-IDS2017 <-> CSE-CIC-IDS2018 test")
    args = parser.parse_args()

    seeds = [42 + i for i in range(args.seeds)]  # seed 42 is the Phase 1 split
    if args.cross:
        exp_cross(seeds[0], 30, os.path.join(RESULTS_DIR, "cross_dataset", "phase2"))
        return
    if not args.dataset:
        parser.error("--dataset is required unless --cross is given")

    cfg = get_config(args.dataset)
    k = cfg["n_features_to_select"]
    out_dir = os.path.join(RESULTS_DIR, args.dataset, "phase2")
    df = load(args.dataset)
    wanted = [e.strip() for e in args.experiments.split(",") if e.strip()]
    for exp in wanted:
        start = time.time()
        if exp == "protocols":
            exp_protocols(df, seeds, k, args.deep, out_dir)
        elif exp == "official":
            if args.dataset != "nsl_kdd":
                print("[phase2] official split exists only for NSL-KDD; skipping", flush=True)
                continue
            exp_official(df, seeds, k, args.deep, out_dir)
        elif exp == "holdout":
            exp_holdout(df, seeds[0], k, out_dir)
        elif exp == "shortcut":
            exp_shortcut(df, seeds[0], k, out_dir)
        else:
            raise SystemExit(f"unknown experiment {exp!r}")
        print(f"[phase2] {exp} finished in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
