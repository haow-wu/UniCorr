"""Re-rank substitutions at the same site with ThermoMPNN.

The stability score is centered inside each wild-type residue, so it
cannot change which site is predicted, only which amino acid. The slope
is fit on training proteins. Test predictions are written only when this
beats the attention model on validation.
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


def add_site(frame):
    parsed = frame["mutation"].map(train_esm650.parse_mutation)
    res = [("" if item is None else str(item[1])) for item in parsed]
    out = frame.copy()
    out["site"] = out["complex"].astype(str) + "|" + pd.Series(res, index=out.index).astype(str)
    return out


def center(values, sites):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    return train_gate.center_by_group(values.reshape(-1, 1), np.asarray(sites)).ravel()


def base_predict(train, other, vectors):
    return train_hbens.fit_columns(
        train, other, vectors,
        ["attn_delta", "attn_mt", "attn_x_llr", "attn_x_prox"],
    )


def residual(train, other, columns, alpha):
    """Slope from multi-substitution sites only, shrunk by alpha * variance."""
    y = center(train["y"].to_numpy(dtype=float), train["site"])
    x_tr = np.column_stack([center(train[col].to_numpy(dtype=float), train["site"]) for col in columns])
    x_ot = np.column_stack([center(other[col].to_numpy(dtype=float), other["site"]) for col in columns])
    counts = train["site"].astype(str).value_counts()
    multi = train["site"].astype(str).map(counts).to_numpy() >= 2
    if int(multi.sum()) < 20:
        return np.zeros(len(other), dtype=float)
    x_m = x_tr[multi]
    y_m = y[multi]
    gram = x_m.T @ x_m
    scale = float(np.trace(gram) / max(len(columns), 1))
    penalty = float(alpha) * max(scale, 1e-8) * np.eye(len(columns))
    coef = np.linalg.solve(gram + penalty, x_m.T @ y_m)
    add = x_ot @ coef
    return np.clip(add, -1.0, 1.0)


def main():
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
    frame = add_site(frame)
    options = [("base", None, None)]
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        options.append((f"thermo_a{alpha:g}", ["thermompnn_ddg"], alpha))
        options.append((f"thermo_llr_a{alpha:g}", ["thermompnn_ddg", "esm650_llr"], alpha))
    table = []
    for key, cols, alpha in options:
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred = base_predict(train, val, vectors)
            if cols is not None:
                pred = pred + residual(train, val, cols, alpha)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, mean, cols, alpha))
        shown = [None if not np.isfinite(item) else round(float(item), 4) for item in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[1])
    chosen, mean, cols, alpha = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "base":
        print("base remains best on validation; test file unchanged", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = base_predict(train, test, vectors) + residual(train, test, cols, alpha)
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
                "model": "siteres",
            })
    base = pd.read_csv(train_gate.OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    dest = train_gate.OUT / "predictions_siteres.csv"
    preds.to_csv(dest, index=False)
    print("wrote", dest, len(rows), flush=True)
    subprocess.check_call([
        sys.executable, str(train_gate.OUT / "score_metrics.py"),
        str(dest), str(train_gate.OUT / "metrics_siteres.json"),
    ])


if __name__ == "__main__":
    main()
