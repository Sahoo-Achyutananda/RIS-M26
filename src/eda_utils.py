"""Helpers for the per-dataset EDA notebooks (raw data, before any cleaning)."""

import numpy as np
import pandas as pd

from config import get_config


def _reader(path, cfg, chunksize, usecols=None):
    kw = dict(low_memory=False, encoding=cfg.get("encoding", "utf-8"), chunksize=chunksize)
    if cfg["has_header"]:
        kw["header"] = 0
    else:
        kw.update(header=None, names=cfg["column_names"])
    if usecols is not None:
        kw["usecols"] = usecols
    return pd.read_csv(path, **kw)


def label_counts(dataset: str, chunksize: int = 50_000) -> pd.DataFrame:
    """Exact multi-class label counts per raw file (reads only the label column)."""
    cfg = get_config(dataset)
    label_col = cfg["label_column"]
    rows = {}
    for path in cfg["raw_files"]:
        counts = pd.Series(dtype="int64")
        keep = (lambda c: str(c).strip() == label_col) if cfg["has_header"] else [label_col]
        for chunk in _reader(path, cfg, chunksize, usecols=keep):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            counts = counts.add(chunk[label_col].astype(str).str.strip().value_counts(), fill_value=0)
        rows[path.replace("\\", "/").split("/")[-1]] = counts.astype("int64")
    return pd.DataFrame(rows).fillna(0).astype("int64").T


def load_sample(dataset: str, frac: float = 1.0, seed: int = 42, chunksize: int = 20_000) -> pd.DataFrame:
    """Uniform random sample of every raw file, concatenated, with a `_source` column. No cleaning."""
    cfg = get_config(dataset)
    parts = []
    for path in cfg["raw_files"]:
        for chunk in _reader(path, cfg, chunksize):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            if frac < 1.0:
                chunk = chunk.sample(frac=frac, random_state=seed)
            chunk["_source"] = path.replace("\\", "/").split("/")[-1]
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True, sort=False)


def to_binary(labels: pd.Series, benign_values) -> pd.Series:
    benign = {str(v).strip().lower() for v in benign_values}
    return labels.astype(str).str.strip().str.lower().map(lambda v: 0 if v in benign else 1)


def quality_table(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    """Per-column data-quality summary. Text values are coerced to numbers so 'Infinity' strings are counted."""
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        was_text = s.dtype == object
        num = pd.to_numeric(s, errors="coerce")
        non_numeric = int(num.isna().sum() - s.isna().sum())
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing_%": round(100 * s.isna().mean(), 3),
            "non_numeric_text": non_numeric if was_text else 0,
            "inf": int(np.isinf(num).sum()),
            "negative": int((num < 0).sum()),
            "nunique": int(s.nunique(dropna=True)),
            "min": num.replace([np.inf, -np.inf], np.nan).min(),
            "max": num.replace([np.inf, -np.inf], np.nan).max(),
        })
    return pd.DataFrame(rows).set_index("column")


def numeric_frame(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    """Numeric view for analysis only (coerce, inf -> NaN -> 0). This is NOT the cleaning step."""
    cols = [c for c in df.columns if c not in exclude]
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)
