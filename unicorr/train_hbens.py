"""Add the Boltzmann hydrogen-bond term if validation improves.

d_hb_ens is positive inside every validation fold. The interaction ridge
stays the control. Test predictions are written only when a candidate
beats that control on validation proteins.
"""

import numpy as np
import pandas as pd

import train_decay
import train_esm650
import train_esm650h
import train_esmllr
import train_gate
import train_interact650


def add_hb(frame):
    ens = pd.concat([
        pd.read_csv("/root/autodl-tmp/project4/cache/ensemble_features.csv"),
        pd.read_csv(train_gate.OUT / "ensemble_extra.csv"),
    ], ignore_index=True)
    ens["sample_id"] = ens["graph_id"].astype(str)
    ens = ens.drop_duplicates("sample_id", keep="last")
    frame = frame.merge(ens[["sample_id", "d_hb_ens"]], on="sample_id", how="left")
    frame["hb_prox"] = frame["d_hb_ens"] * frame["prox"]
    return frame


def fit_columns(train, other, vectors, extra):
    base = train_esm650h.WIDE + ["llr_x_polar", "llr_x_close"] + extra
    train, other, columns = train_esm650h.with_hidden(train, other, vectors, base, 16)
    pred, _ = train_gate.fit_predict_fold(
        train, other, columns, train_decay.ALPHA, "between", vectors
    )
    return train_decay.gated(
        pred,
        other["complex"].astype(str).to_numpy(),
        other["heavy_dist"].to_numpy(dtype=float),
        "exp",
        10.0,
    )


def main():
    frame, vectors = train_gate.load()
    frame = train_esmllr.add_llr(frame)
    frame = train_esm650.add_650(frame)
    frame = train_interact650.add_interaction(frame)
    frame = add_hb(frame)
    finite = int(np.isfinite(pd.to_numeric(frame["d_hb_ens"], errors="coerce")).sum())
    print("d_hb_ens finite", finite, "rows", len(frame), flush=True)
    options = {
        "control": [],
        "hb": ["d_hb_ens"],
        "hb_prox": ["d_hb_ens", "hb_prox"],
    }
    table = []
    for key, extra in options.items():
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred = fit_columns(train, val, vectors, extra)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, mean, scores, extra))
        shown = [None if not np.isfinite(item) else round(float(item), 4) for item in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[1])
    chosen, mean, _scores, extra = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control":
        print("control remains best on validation; test file unchanged", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = fit_columns(train, test, vectors, extra)
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
                "model": "hbens",
            })
    base = pd.read_csv(train_gate.OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(train_gate.OUT / "predictions_hbens.csv", index=False)
    print("wrote predictions_hbens.csv", len(rows), flush=True)


if __name__ == "__main__":
    main()
