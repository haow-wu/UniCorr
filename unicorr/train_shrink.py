"""Shrink unstable UniProt z-scores. Select on three validation folds.

The reference nudge uses weights 1, 1 and -0.5 on d_hb_ens, prox and
x_clash. Discovery folds 0-2 propose a candidate. Folds 3-4 must confirm
it, and the five-fold validation score must not fall. A test file is
written only then.
"""

import subprocess
import sys

import numpy as np
import pandas as pd

import score_metrics
import train_gate
import train_znudge

REF = (("d_hb_ens", 1.0), ("prox", 1.0), ("x_clash", -0.5))
DISCOVER = (0, 1, 2)
CONFIRM = (3, 4)


def pieces(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    groups = frame["uniprot"].astype(str).to_numpy()
    order = pd.Series(values)
    mean = order.groupby(groups).transform("mean").to_numpy()
    std = order.groupby(groups).transform("std").to_numpy()
    count = order.groupby(groups).transform("size").to_numpy(dtype=float)
    return values, mean, std, count


def pooled_scale(train, column):
    values, mean, _std, _count = pieces(train, column)
    centered = values - mean
    centered = centered[np.isfinite(centered)]
    scale = float(np.std(centered)) if len(centered) else 1.0
    return scale if scale > 1e-8 else 1.0


def z_values(frame, column, mode, param, scale):
    values, mean, std, count = pieces(frame, column)
    if mode == "global":
        raw = (values - mean) / scale
    else:
        raw = (values - mean) / np.where(std == 0.0, np.nan, std)
        if mode == "shrink":
            raw = raw * count / (count + float(param))
        elif mode == "clip":
            raw = np.clip(raw, -float(param), float(param))
    return np.where(np.isfinite(raw), raw, 0.0)


def predict(frame, base, mode, param, scales):
    pred = np.asarray(base, dtype=float).copy()
    for column, weight in REF:
        spread = float(np.std(pred)) + 1e-8
        pred = pred + weight * z_values(frame, column, mode, param, scales[column]) * spread
    return pred


def evaluate(parts, folds):
    chunks = []
    for fold in folds:
        val, pred = parts[fold]
        chunk = val[["uniprot", "y", "fold"]].copy()
        chunk["pred"] = pred
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    stats = score_metrics.fold_stats(frame)
    mean = score_metrics.mean_of(stats, "pearson")
    _protein, without, _drop = score_metrics.top_protein(frame, mean)
    fold_r = [row["pearson"] for row in stats]
    worst = None if any(item is None for item in fold_r) else float(min(fold_r))
    stable = -1.0 if mean is None or without is None else float(min(mean, without))
    return stable, mean, without, worst


def main():
    frame, vectors = train_znudge.prepare()
    cache = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        val = part.loc[part["split"] == "val"]
        base = train_znudge.base_predict(train, val, vectors)
        scales = {column: pooled_scale(train, column) for column, _weight in REF}
        cache.append((train, val, base, scales))
        print("packed", fold, flush=True)
    specs = [("own", "own", 0.0)]
    for k in (1.0, 2.0, 4.0, 8.0):
        specs.append((f"shrink{k:g}", "shrink", k))
    for limit in (1.0, 1.5, 2.0):
        specs.append((f"clip{limit:g}", "clip", limit))
    specs.append(("global", "global", 0.0))
    scored = {}
    for name, mode, param in specs:
        parts = []
        for _train, val, base, scales in cache:
            parts.append((val, predict(val, base, mode, param, scales)))
        discover = evaluate(parts, DISCOVER)
        confirm = evaluate(parts, CONFIRM)
        full = evaluate(parts, (0, 1, 2, 3, 4))
        scored[name] = (mode, param, parts, discover, confirm, full)
        print(
            name,
            "d", round(discover[0], 4), round(discover[1], 4), round(discover[3], 4),
            "c", round(confirm[0], 4), round(confirm[1], 4), round(confirm[3], 4),
            "full", round(full[0], 4), round(full[1], 4), round(full[3], 4),
            flush=True,
        )
    ref_d, ref_c, ref_f = scored["own"][3], scored["own"][4], scored["own"][5]
    kept = []
    for name, (mode, param, parts, discover, confirm, full) in scored.items():
        if name == "own":
            continue
        if discover[0] < ref_d[0] + 0.01:
            continue
        if discover[1] < ref_d[1] - 0.005 or discover[3] < ref_d[3]:
            continue
        if confirm[0] < ref_c[0] or confirm[3] < ref_c[3] - 0.01:
            continue
        if full[0] < ref_f[0] + 0.005 or full[3] < ref_f[3]:
            continue
        kept.append((name, mode, param, confirm[0], full))
    if not kept:
        print("CHOSEN own", flush=True)
        print("reference nudge remains best on held-out validation folds; test file unchanged", flush=True)
        return
    kept.sort(key=lambda item: -item[3])
    name, mode, param, _confirm, full = kept[0]
    print("CHOSEN", name, "full_stable", round(full[0], 4), "full_mean", round(full[1], 4), flush=True)
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        base = train_znudge.base_predict(train, test, vectors)
        scales = {column: pooled_scale(train, column) for column, _weight in REF}
        pred = predict(test, base, mode, param, scales)
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
                "model": "shrink",
            })
    saved = pd.read_csv(train_gate.OUT / "predictions.csv")
    saved = saved.loc[saved["model"] == "baseline"].copy()
    preds = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
    dest = train_gate.OUT / "predictions_shrink.csv"
    preds.to_csv(dest, index=False)
    print("wrote", dest, len(rows), flush=True)
    subprocess.check_call([
        sys.executable, str(train_gate.OUT / "score_metrics.py"),
        str(dest), str(train_gate.OUT / "metrics_shrink.json"),
    ])


if __name__ == "__main__":
    main()
