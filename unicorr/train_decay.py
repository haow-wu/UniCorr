"""Scale the within-complex residual by distance to the ligand.

The base model is the validation winner that adds ESM-2 masked log-odds.
A distance gate is chosen on validation proteins. Test predictions are
written only when a gate beats the ungated model.
"""

import numpy as np
import pandas as pd

import train_esmllr
import train_gate

PACK = list(train_gate.PACK)
ESM = PACK + [f"pc{i}" for i in range(32)] + ["esm_llr", "esm_prox"]
ALPHA = 300000.0


def gated(pred, groups, heavy, kind, param):
    pred = np.asarray(pred, float)
    heavy = np.asarray(heavy, float)
    heavy = np.where(np.isfinite(heavy), heavy, 30.0)
    if kind == "none":
        return pred
    if kind == "exp":
        weight = np.exp(-heavy / param)
    elif kind == "hinge":
        weight = np.clip(param - heavy, 0.0, None) / param
    else:
        raise KeyError(kind)
    out = np.array(pred, copy=True)
    for gid in pd.unique(groups):
        mask = groups == gid
        center = float(pred[mask].mean())
        out[mask] = center + weight[mask] * (pred[mask] - center)
    return out


def configs():
    return [
        ("ungated", "none", None),
        ("exp5", "exp", 5.0),
        ("exp10", "exp", 10.0),
        ("hinge8", "hinge", 8.0),
        ("hinge12", "hinge", 12.0),
        ("hinge15", "hinge", 15.0),
    ]


def main():
    frame, vectors = train_gate.load()
    frame = train_esmllr.add_llr(frame)
    table = []
    for key, kind, param in configs():
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred, _ = train_gate.fit_predict_fold(train, val, ESM, ALPHA, "between", vectors)
            pred = gated(pred, val["complex"].astype(str).to_numpy(), val["heavy_dist"].to_numpy(float), kind, param)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, kind, param, mean, scores))
        shown = [None if not np.isfinite(s) else round(float(s), 4) for s in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[3])
    chosen, kind, param, mean, _ = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "ungated":
        print("ungated remains best on validation; test file stays predictions_esmllr.csv", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred, _ = train_gate.fit_predict_fold(train, test, ESM, ALPHA, "between", vectors)
        pred = gated(pred, test["complex"].astype(str).to_numpy(), test["heavy_dist"].to_numpy(float), kind, param)
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
                "model": "decay",
            })
    base = pd.read_csv(train_gate.OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(train_gate.OUT / "predictions_decay.csv", index=False)
    print("wrote predictions_decay.csv", len(rows), flush=True)


if __name__ == "__main__":
    main()
