"""Try a coarser grouping and unused rotamer or tip columns.

The clash model groups z-scores by wild-type complex. One candidate
groups the same three nudges by UniProt. Other candidates add one
cached column whose training sign agrees in every fold. A test file is
written only when validation Pearson rises by at least 0.01 and the
worst fold does not fall by more than 0.02.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import train_gate
import train_znudge

MIN_GAIN = 0.01
WORST_DROP = 0.02
ENSEMBLE = (
    "d_lig_0p5", "d_lig_1p0", "d_lig_2p0", "d_lig_4p0",
    "d_tot_0p5", "d_tot_1p0", "d_tot_2p0", "d_tot_4p0",
    "d_lig_min", "d_tot_min", "d_ct_ens", "d_att_ens",
)
EXTRA_FILES = (
    (train_gate.OUT / "geom_features.csv", ("polar_min", "arom")),
    (train_gate.OUT / "tip_distance.csv", ("tip_delta", "tip_mut")),
    (train_gate.OUT / "tip_shell.csv", ("d_tip_min", "d_tip_n_45", "d_tip_polar_45")),
)
ON_FRAME = ("d_hydro", "d_charge", "d_vol", "esm650_llr", "esm_llr", "rsa")


def within_z(frame, column, group_col):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    groups = frame[group_col].astype(str).to_numpy()
    mean = pd.Series(values).groupby(groups).transform("mean").to_numpy()
    std = pd.Series(values).groupby(groups).transform("std").to_numpy()
    z = (values - mean) / np.where(std == 0.0, np.nan, std)
    return np.where(np.isfinite(z), z, 0.0)


def apply_nudges(frame, base, group_col):
    pred = np.asarray(base, dtype=float).copy()
    for weight, column in ((0.5, "d_hb_ens"), (1.0, "prox"), (-0.5, "x_clash")):
        spread = float(np.std(pred)) + 1e-8
        pred = pred + weight * within_z(frame, column, group_col) * spread
    return pred


def train_sign(train, column):
    z = within_z(train, column, "complex")
    y = pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=float)
    groups = train["complex"].astype(str).to_numpy()
    centered = y - pd.Series(y).groupby(groups).transform("mean").to_numpy()
    ok = np.isfinite(centered) & np.isfinite(z)
    if int(ok.sum()) < 30 or float(np.std(z[ok])) == 0.0 or float(np.std(centered[ok])) == 0.0:
        return 0.0
    score = train_gate.pearson(centered[ok], z[ok])
    if score is None or not np.isfinite(score) or score == 0.0:
        return 0.0
    return 1.0 if score > 0.0 else -1.0


def add_columns(frame):
    ens = pd.concat([
        pd.read_csv("/root/autodl-tmp/project4/cache/ensemble_features.csv"),
        pd.read_csv(train_gate.OUT / "ensemble_extra.csv"),
    ], ignore_index=True)
    ens["sample_id"] = ens["graph_id"].astype(str)
    ens = ens.drop_duplicates("sample_id", keep="last")
    present = [col for col in ENSEMBLE if col in ens.columns]
    print("ensemble", present, flush=True)
    if present:
        frame = frame.merge(ens[["sample_id"] + present], on="sample_id", how="left")
    found = list(present)
    for path, columns in EXTRA_FILES:
        if not Path(path).exists():
            print("missing", path, flush=True)
            continue
        extra = pd.read_csv(path)
        extra["sample_id"] = extra["sample_id"].astype(str)
        keep = [col for col in columns if col in extra.columns]
        if keep:
            frame = frame.merge(extra[["sample_id"] + keep], on="sample_id", how="left")
            found.extend(keep)
    for column in ON_FRAME:
        if column in frame.columns:
            found.append(column)
    return frame, found


def scores_of(pairs):
    out = []
    for _val, pred in pairs:
        score = train_gate.pearson(_val["y"], pred)
        out.append(np.nan if score is None else score)
    return out


def main():
    frame, vectors = train_znudge.prepare()
    frame, columns = add_columns(frame)
    base_cache = []
    current_pairs = []
    uniprot_pairs = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        val = part.loc[part["split"] == "val"]
        base = train_znudge.base_predict(train, val, vectors)
        base_cache.append((train, val, base))
        current_pairs.append((val, apply_nudges(val, base, "complex")))
        uniprot_pairs.append((val, apply_nudges(val, base, "uniprot")))
        print("packed", fold, flush=True)
    base_scores = scores_of(current_pairs)
    base_mean = float(np.nanmean(base_scores))
    base_worst = float(np.nanmin(base_scores))
    print("current", "val_mean", round(base_mean, 4), "worst", round(base_worst, 4), flush=True)
    table = [("current", "complex", None, 0.0, base_mean, base_worst)]
    uni_scores = scores_of(uniprot_pairs)
    uni_mean = float(np.nanmean(uni_scores))
    uni_worst = float(np.nanmin(uni_scores))
    table.append(("uniprot", "uniprot", None, 0.0, uni_mean, uni_worst))
    print("uniprot", "val_mean", round(uni_mean, 4), "worst", round(uni_worst, 4), flush=True)
    for column in columns:
        fold_signs = [train_sign(train, column) for train, _val, _base in base_cache]
        print(column, "train_signs", fold_signs, flush=True)
        if column not in frame.columns or len(set(fold_signs)) != 1 or fold_signs[0] == 0.0:
            continue
        sign = fold_signs[0]
        for magnitude in (0.25, 0.5, 1.0):
            weight = sign * magnitude
            folded = []
            for (_train, val, base), (_val, pred) in zip(base_cache, current_pairs):
                z = within_z(val, column, "complex")
                nudged = pred + weight * z * (float(np.std(pred)) + 1e-8)
                folded.append((val, nudged))
            got = scores_of(folded)
            mean = float(np.nanmean(got))
            worst = float(np.nanmin(got))
            table.append((column, "complex", column, weight, mean, worst))
            print(column, f"w{weight:g}", "val_mean", round(mean, 4), "worst", round(worst, 4), flush=True)
    eligible = [
        item for item in table
        if item[0] == "current" or (item[4] >= base_mean + MIN_GAIN and item[5] >= base_worst - WORST_DROP)
    ]
    eligible.sort(key=lambda item: -item[4])
    name, group_col, column, weight, mean, worst = eligible[0]
    print("CHOSEN", name, weight, "val_mean", round(mean, 4), "worst", round(worst, 4), flush=True)
    if name == "current":
        print("clash model remains best on validation; test file unchanged", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        base = train_znudge.base_predict(train, test, vectors)
        pred = apply_nudges(test, base, group_col)
        if column is not None:
            z = within_z(test, column, "complex")
            pred = pred + weight * z * (float(np.std(pred)) + 1e-8)
        if not np.isfinite(pred).all():
            raise SystemExit(f"non-finite test prediction in fold {fold}")
        for i, rec in enumerate(test.itertuples(index=False)):
            dist = rec.distance
            rows.append({
                "sample_id": rec.sample_id,
                "uniprot": rec.uniprot,
                "y": float(rec.y),
                "pred": float(pred[i]),
                "label_type": rec.label_type,
                "distance": float(dist) if np.isfinite(dist) else np.nan,
                "fold": int(fold),
                "model": "znudge6",
            })
    saved = pd.read_csv(train_gate.OUT / "predictions.csv")
    saved = saved.loc[saved["model"] == "baseline"].copy()
    preds = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
    dest = train_gate.OUT / "predictions_znudge6.csv"
    preds.to_csv(dest, index=False)
    print("wrote", dest, len(rows), flush=True)
    subprocess.check_call([
        sys.executable, str(train_gate.OUT / "score_metrics.py"),
        str(dest), str(train_gate.OUT / "metrics_znudge6.json"),
    ])


if __name__ == "__main__":
    main()
