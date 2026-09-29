"""
Collects the Phase 2 CSVs of every dataset into one summary (printed, and saved as
results/phase2_summary.md) plus figures for the report (results/phase2_figures/).

    python phase2_report.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

DATASETS = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15",
            "cse_cic_ids2018": "CSE-CIC-IDS2018", "cic_ids2017": "CIC-IDS2017"}
HYBRID = "Averaging Hybrid (paper)"
FIG_DIR = os.path.join(RESULTS_DIR, "phase2_figures")  # inside results/ so it is in the Kaggle zip
# Nemenyi critical values q_0.05 for k = 2..10 models (Demsar, 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def read(ds, name):
    path = os.path.join(RESULTS_DIR, ds, "phase2", name)
    return pd.read_csv(path) if os.path.exists(path) else None


def pct(m, s=None):
    return f"{100 * m:.2f}" if s is None or np.isnan(s) else f"{100 * m:.2f} ± {100 * s:.2f}"


def section(title, df):
    return f"\n## {title}\n\n{df.to_markdown(index=False)}\n" if df is not None and len(df) else ""


def protocols_table():
    rows = []
    for ds, name in DATASETS.items():
        s = read(ds, "protocols_summary.csv")
        if s is None:
            continue
        for proto in ["paper", "phase1", "fair"]:
            r = s[(s["Protocol"] == proto) & (s["Model"] == HYBRID)]
            if len(r):
                r = r.iloc[0]
                rows.append({"Dataset": name, "Protocol": proto, "Accuracy %": pct(r.Accuracy_mean, r.Accuracy_std),
                             "FPR %": pct(r.FPR_mean, r.FPR_std), "FNR %": pct(r.FNR_mean, r.FNR_std)})
    return pd.DataFrame(rows)


def models_table():
    rows = []
    for ds, name in DATASETS.items():
        s = read(ds, "protocols_summary.csv")
        if s is None:
            continue
        for _, r in s[s["Protocol"] == "fair"].sort_values("Accuracy_mean", ascending=False).iterrows():
            rows.append({"Dataset": name, "Model": r.Model, "Accuracy %": pct(r.Accuracy_mean, r.Accuracy_std),
                         "F1 %": pct(r.F1_mean, r.F1_std), "FPR %": pct(r.FPR_mean, r.FPR_std),
                         "AUC": f"{r.AUC_mean:.4f}"})
    return pd.DataFrame(rows)


def significance_table():
    rows = []
    for ds, name in DATASETS.items():
        s = read(ds, "significance.csv")
        if s is None:
            continue
        per_seed = s[s["Seed"] != "all"]
        overall = s[s["Seed"] == "all"].set_index("Model")
        for m, g in per_seed.groupby("Model"):
            rows.append({"Dataset": name, "Hybrid vs": m,
                         "Mean accuracy edge (pts)": f"{100 * overall.loc[m, 'HybridMinusModelAcc']:+.3f}",
                         "Hybrid-only / model-only right (per seed)":
                             ", ".join(f"{int(a)}/{int(b)}" for a, b in zip(g.HybridOnlyRight, g.ModelOnlyRight)),
                         "Seeds with McNemar p<0.05": f"{int((g.McNemarP < 0.05).sum())}/{len(g)}",
                         "Corrected t-test p": f"{overall.loc[m, 'CorrectedTTestP']:.3f}"})
    return pd.DataFrame(rows)


def friedman_summary():
    """Friedman test over (dataset, seed) blocks on fair-protocol accuracy, with Nemenyi critical difference."""
    blocks = []
    for ds in DATASETS:
        p = read(ds, "protocols.csv")
        if p is not None:
            f = p[p["Protocol"] == "fair"].pivot_table(index="Seed", columns="Model", values="Accuracy")
            f.index = [f"{ds}-{i}" for i in f.index]
            blocks.append(f)
    if not blocks:
        return None, ""
    acc = pd.concat(blocks).dropna(axis=1)  # only models run on every dataset
    ranks = acc.rank(axis=1, ascending=False).mean().sort_values()
    k, n = acc.shape[1], acc.shape[0]
    stat, p = stats.friedmanchisquare(*[acc[c] for c in acc.columns])
    cd = Q05.get(k, float("nan")) * np.sqrt(k * (k + 1) / (6 * n))
    table = pd.DataFrame({"Model": ranks.index, "Average rank (1 = best)": ranks.values.round(2)})
    note = (f"Friedman test over {n} blocks (dataset x seed), {k} models: chi2 = {stat:.2f}, p = {p:.2g}. "
            f"Nemenyi critical difference at 0.05: {cd:.2f} rank units.")
    return table, note


def family_table():
    rows = []
    for ds, name in DATASETS.items():
        f = read(ds, "family_recall_summary.csv")
        if f is None:
            continue
        h = f[f["Model"] == HYBRID]
        for _, r in h.iterrows():
            label = "false-alarm rate" if r.Family == "Benign" else "caught"
            rows.append({"Dataset": name, "Family": r.Family, "Test rows": int(round(r.N)),
                         f"Hybrid: % flagged": f"{100 * r.FlaggedAsAttack:.2f}", "Meaning": label})
    return pd.DataFrame(rows)


def official_tables():
    o = read("nsl_kdd", "official.csv")
    if o is None:
        return None, None
    s = o.groupby("Model")[["Accuracy", "F1", "FPR", "FNR"]].agg(["mean", "std"])
    t = pd.DataFrame({"Model": s.index,
                      "Accuracy %": [pct(s.loc[m, ("Accuracy", "mean")], s.loc[m, ("Accuracy", "std")]) for m in s.index],
                      "FPR %": [pct(s.loc[m, ("FPR", "mean")]) for m in s.index],
                      "FNR %": [pct(s.loc[m, ("FNR", "mean")]) for m in s.index]})
    sv = read("nsl_kdd", "official_seen_vs_new.csv")
    if sv is not None:
        sv = sv.assign(**{"Attack types": sv.SeenInTraining.map({True: "seen in learning file", False: "new in test"}),
                          "Recall %": (100 * sv.Recall).round(2), "Test rows": sv.N.round().astype(int)})
        sv = sv[["Model", "Attack types", "Test rows", "Recall %"]]
    return t.sort_values("Accuracy %", ascending=False), sv


def holdout_table():
    rows = []
    for ds, name in DATASETS.items():
        h = read(ds, "holdout.csv")
        if h is None:
            continue
        for _, r in h[h["Model"] == HYBRID].iterrows():
            rows.append({"Dataset": name, "Hidden family": r.Family, "Test rows": r.NTest,
                         "Caught when seen %": f"{100 * r.RecallWhenSeen:.1f}",
                         "Caught when hidden %": f"{100 * r.RecallWhenHidden:.1f}"})
    return pd.DataFrame(rows)


def shortcut_table_and_figure():
    rows, curves = [], {}
    for ds, name in DATASETS.items():
        s = read(ds, "shortcut.csv")
        if s is None:
            continue
        h = s[s["Model"] == HYBRID].sort_values("Dropped")
        curves[name] = (h.Dropped.to_numpy(), 100 * h.Accuracy.to_numpy())
        for _, r in h.iterrows():
            rows.append({"Dataset": name, "Columns removed": r.Dropped, "Accuracy %": f"{100 * r.Accuracy:.2f}",
                         "FPR %": f"{100 * r.FPR:.2f}",
                         "Removed": r.DroppedColumns if isinstance(r.DroppedColumns, str) else ""})
    if curves:
        plt.figure(figsize=(6, 4))
        for name, (x, y) in curves.items():
            plt.plot(x, y, marker="o", label=name)
        plt.xlabel("Most important columns removed")
        plt.ylabel("Hybrid accuracy (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, "phase2_shortcut.png"), dpi=200)
        plt.close()
    return pd.DataFrame(rows)


def protocols_figure():
    data = {}
    for ds, name in DATASETS.items():
        s = read(ds, "protocols_summary.csv")
        if s is not None:
            h = s[s["Model"] == HYBRID].set_index("Protocol")
            data[name] = [100 * (1 - h.loc[p, "Accuracy_mean"]) if p in h.index else np.nan
                          for p in ["paper", "phase1", "fair"]]
    if not data:
        return
    x = np.arange(len(data))
    plt.figure(figsize=(7, 4))
    for i, (label, color) in enumerate([("Study's order", "#4C78A8"), ("Our Phase 1", "#F58518"),
                                        ("Fair order", "#54A24B")]):
        plt.bar(x + (i - 1) * 0.27, [v[i] for v in data.values()], 0.27, label=label, color=color)
    plt.xticks(x, list(data))
    plt.ylabel("Hybrid error (100 - accuracy, %)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, "phase2_protocols.png"), dpi=200)
    plt.close()


def cross_table():
    c = pd.read_csv(os.path.join(RESULTS_DIR, "cross_dataset", "phase2", "cross.csv")) \
        if os.path.exists(os.path.join(RESULTS_DIR, "cross_dataset", "phase2", "cross.csv")) else None
    if c is None:
        return None
    c = c[c["Model"].isin([HYBRID, "XGBoost", "Random Forest", "LightGBM"])]
    return pd.DataFrame({"Learned on": c.Train, "Tested on": c.Test, "Model": c.Model,
                         "Accuracy %": (100 * c.Accuracy).round(2), "Recall %": (100 * c.Recall).round(2),
                         "FPR %": (100 * c.FPR).round(2)})


def main():
    out = ["# Phase 2 summary\n"]
    out.append(section("Order of steps (hybrid, mean ± std over seeds)", protocols_table()))
    out.append(section("Fair order: all models (mean ± std over seeds)", models_table()))
    out.append(section("Is the hybrid really better? (fair order)", significance_table()))
    ranks, note = friedman_summary()
    if ranks is not None:
        out.append(section("Ranking across all datasets and seeds", ranks) + f"\n{note}\n")
    out.append(section("Share of each attack family caught (hybrid, fair order, mean over seeds)", family_table()))
    official, seen_new = official_tables()
    out.append(section("NSL-KDD official test (KDDTrain+ -> KDDTest+)", official))
    out.append(section("NSL-KDD official test: seen vs new attack types", seen_new))
    out.append(section("Hidden attack family (hybrid)", holdout_table()))
    out.append(section("Removing the most important columns (hybrid)", shortcut_table_and_figure()))
    out.append(section("Learn on one dataset, test on the other", cross_table()))
    protocols_figure()
    text = "".join(out)
    path = os.path.join(RESULTS_DIR, "phase2_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved {path}; figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
