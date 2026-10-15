"""Compute the task metrics from predictions.csv. Do not edit the numbers by hand.

Required columns: sample_id, uniprot, y, pred, label_type, distance, fold, model.
Distance is angstroms from the mutated residue to the ligand.
A fold with fewer than 30 finite rows is unstable and is left out of the mean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_N = 30


def pearson(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < MIN_N or float(np.std(y)) < 1e-12 or float(np.std(p)) < 1e-12:
        return None
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < MIN_N or float(np.std(y)) < 1e-12 or float(np.std(p)) < 1e-12:
        return None
    yr = pd.Series(y).rank().to_numpy()
    pr = pd.Series(p).rank().to_numpy()
    return float(np.corrcoef(yr, pr)[0, 1])


def rmse(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return None
    return float(np.sqrt(np.mean((p - y) ** 2)))


def fold_stats(frame):
    rows = []
    for fold, part in frame.groupby("fold"):
        n = int((np.isfinite(part["y"].to_numpy()) & np.isfinite(part["pred"].to_numpy())).sum())
        stable = n >= MIN_N
        rows.append({
            "fold": int(fold) if str(fold).isdigit() else str(fold),
            "n": n,
            "n_uniprot": int(part["uniprot"].nunique()) if len(part) else 0,
            "stable": stable,
            "pearson": pearson(part["y"], part["pred"]) if stable else None,
            "spearman": spearman(part["y"], part["pred"]) if stable else None,
            "rmse": rmse(part["y"], part["pred"]) if n else None,
        })
    return rows


def mean_of(rows, key):
    vals = [row[key] for row in rows if row["stable"] and row[key] is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def distance_bin(value):
    if not np.isfinite(value):
        return "missing"
    if value < 5.0:
        return "lt5"
    if value <= 15.0:
        return "5to15"
    return "gt15"


def slice_metrics(frame):
    n = int((np.isfinite(frame["y"]) & np.isfinite(frame["pred"])).sum()) if len(frame) else 0
    return {
        "n": n,
        "n_uniprot": int(frame["uniprot"].nunique()) if len(frame) else 0,
        "pearson": pearson(frame["y"], frame["pred"]) if n >= MIN_N else None,
        "spearman": spearman(frame["y"], frame["pred"]) if n >= MIN_N else None,
        "rmse": rmse(frame["y"], frame["pred"]) if n else None,
        "stable": n >= MIN_N,
    }


def top_protein(frame, base_mean):
    """Protein whose removal lowers the fold-mean Pearson the most."""
    proteins = sorted(frame["uniprot"].dropna().unique())
    best_name = None
    best_drop = None
    best_mean = None
    for name in proteins:
        held = frame.loc[frame["uniprot"] != name]
        rows = fold_stats(held)
        new_mean = mean_of(rows, "pearson")
        if base_mean is None or new_mean is None:
            drop = None
        else:
            drop = float(base_mean - new_mean)
        if drop is None:
            continue
        if best_drop is None or drop > best_drop:
            best_drop = drop
            best_name = str(name)
            best_mean = new_mean
    return best_name, best_mean, best_drop


def one_model(frame):
    rows = fold_stats(frame)
    base = mean_of(rows, "pearson")
    protein, without, drop = top_protein(frame, base)
    frame = frame.copy()
    frame["dist_bin"] = frame["distance"].map(distance_bin)
    bins = {}
    for name in ("lt5", "5to15", "gt15", "missing"):
        bins[name] = slice_metrics(frame.loc[frame["dist_bin"] == name])
    reached = base is not None and base >= 0.30 and without is not None and without >= 0.30
    if base is None:
        status = "未计算"
    elif base < 0.30:
        status = "未达到 0.30"
    elif without is None or without < 0.30:
        status = "被单个蛋白抬高，未达到稳定的 0.30"
    else:
        status = "达到 0.30"
    return {
        "status": status,
        "pearson_mean_stable_folds": base,
        "spearman_mean_stable_folds": mean_of(rows, "spearman"),
        "rmse_mean_stable_folds": mean_of(rows, "rmse"),
        "n_stable_folds": int(sum(row["stable"] for row in rows)),
        "folds": rows,
        "n_test_rows": int(len(frame)),
        "n_test_uniprot": int(frame["uniprot"].nunique()),
        "protein_removed": protein,
        "pearson_mean_without_that_protein": without,
        "pearson_drop": drop,
        "distance": bins,
        "reached_stable_0_30": reached,
    }


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "predictions.csv")
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else "metrics.json")
    frame = pd.read_csv(src)
    needed = ["sample_id", "uniprot", "y", "pred", "label_type", "distance", "fold", "model"]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SystemExit(f"predictions.csv missing columns: {missing}")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame["pred"] = pd.to_numeric(frame["pred"], errors="coerce")
    frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
    frame["uniprot"] = frame["uniprot"].astype(str).str.upper()
    out = {"source": str(src), "min_n_per_fold": MIN_N, "models": {}}
    for model, part in frame.groupby("model"):
        out["models"][str(model)] = one_model(part)
    names = list(out["models"])
    if "baseline" in out["models"] and len(names) > 1:
        other = [name for name in names if name != "baseline"][0]
        b = out["models"]["baseline"]["pearson_mean_stable_folds"]
        m = out["models"][other]["pearson_mean_stable_folds"]
        out["pearson_gain_vs_baseline"] = None if b is None or m is None else float(m - b)
        out["compared_model"] = other
    dest.write_text(json.dumps(out, indent=2))
    print(dest.read_text())


if __name__ == "__main__":
    main()
