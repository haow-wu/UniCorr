"""Nudge the current model by a within-protein z-score.

Hydrogen-bond geometry correlates with the residual, but a kcal-scale
slope overwhelms a prediction whose spread is about 0.01 kcal/mol.
The z-score is scaled to the prediction spread. The weight is chosen
on validation.
"""

import subprocess
import sys

import numpy as np
import pandas as pd

import train_esm650
import train_esmllr
import train_gate
import train_hbens
import train_interact650
import train_siteres


def prepare():
    frame, vectors = train_gate.load()
    frame = train_esmllr.add_llr(frame)
    frame = train_esm650.add_650(frame)
    frame = train_interact650.add_interaction(frame)
    extra = pd.read_csv(train_gate.OUT / "psichic_extra.csv")
    extra["sample_id"] = extra["sample_id"].astype(str)
    frame = frame.merge(extra[["sample_id", "attn_delta", "attn_mt"]], on="sample_id", how="left")
    frame["attn_x_llr"] = pd.to_numeric(frame["attn_mt"], errors="coerce") * frame["esm650_llr"]
    frame["attn_x_prox"] = pd.to_numeric(frame["attn_mt"], errors="coerce") * frame["prox"]
    thermo = pd.read_csv(train_gate.OUT / "thermompnn_ddg.csv")
    thermo["sample_id"] = thermo["sample_id"].astype(str)
    frame = frame.merge(thermo, on="sample_id", how="left")
    geom = pd.read_csv(train_gate.OUT / "geom_features.csv")
    geom["sample_id"] = geom["sample_id"].astype(str)
    frame = frame.merge(geom[["sample_id", "hb_geom"]], on="sample_id", how="left")
    ens = pd.concat([
        pd.read_csv("/root/autodl-tmp/project4/cache/ensemble_features.csv"),
        pd.read_csv(train_gate.OUT / "ensemble_extra.csv"),
    ], ignore_index=True)
    ens["sample_id"] = ens["graph_id"].astype(str)
    ens = ens.drop_duplicates("sample_id", keep="last")
    frame = frame.merge(ens[["sample_id", "d_hb_ens"]], on="sample_id", how="left")
    return frame, vectors


def base_predict(train, other, vectors):
    train = train_siteres.add_site(train)
    other = train_siteres.add_site(other)
    base = train_hbens.fit_columns(
        train, other, vectors,
        ["attn_delta", "attn_mt", "attn_x_llr", "attn_x_prox"],
    )
    return base + train_siteres.residual(train, other, ["thermompnn_ddg"], 1000.0)


def within_z(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce")
    groups = frame["complex"].astype(str)
    mean = values.groupby(groups).transform("mean")
    std = values.groupby(groups).transform("std")
    z = (values - mean) / std.replace(0, np.nan)
    return z.fillna(0.0).to_numpy(dtype=float)


def main():
    frame, vectors = prepare()
    cached = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        val = part.loc[part["split"] == "val"]
        pred = base_predict(train, val, vectors)
        cached.append((val, pred))
        print("packed", fold, "pred_std", round(float(np.std(pred)), 5), flush=True)
    weights = (0.0, 0.5, 1.0, 2.0, 4.0)
    columns = ("hb_geom", "d_hb_ens")
    table = []
    for column in columns:
        for weight in weights:
            scores = []
            for val, pred in cached:
                z = within_z(val, column)
                nudged = pred + weight * z * (float(np.std(pred)) + 1e-8)
                score = train_gate.pearson(val["y"], nudged)
                scores.append(np.nan if score is None else score)
            mean = float(np.nanmean(scores))
            table.append((column, weight, mean, scores))
            print(column, f"w{weight:g}", "val_mean", round(mean, 4), flush=True)
    table.sort(key=lambda item: -item[2])
    column, weight, mean, _ = table[0]
    print("CHOSEN", column, weight, "val_mean", round(mean, 4), flush=True)
    if weight == 0.0:
        print("current model remains best on validation; test file unchanged", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = base_predict(train, test, vectors)
        z = within_z(test, column)
        nudged = pred + weight * z * (float(np.std(pred)) + 1e-8)
        if not np.isfinite(nudged).all():
            raise SystemExit(f"non-finite test prediction in fold {fold}")
        for i, rec in enumerate(test.itertuples(index=False)):
            dist = rec.distance
            rows.append({
                "sample_id": rec.sample_id,
                "uniprot": rec.uniprot,
                "y": float(rec.y),
                "pred": float(nudged[i]),
                "label_type": rec.label_type,
                "distance": float(dist) if np.isfinite(dist) else np.nan,
                "fold": int(fold),
                "model": "znudge",
            })
    saved = pd.read_csv(train_gate.OUT / "predictions.csv")
    saved = saved.loc[saved["model"] == "baseline"].copy()
    preds = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
    dest = train_gate.OUT / "predictions_znudge.csv"
    preds.to_csv(dest, index=False)
    print("wrote", dest, len(rows), flush=True)
    subprocess.check_call([
        sys.executable, str(train_gate.OUT / "score_metrics.py"),
        str(dest), str(train_gate.OUT / "metrics_znudge.json"),
    ])


if __name__ == "__main__":
    main()
