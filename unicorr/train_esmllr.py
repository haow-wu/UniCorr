"""Add the ESM-2 masked log-odds to the current validation winner.

The log-odds use only the wild-type sequence and the mutation letters.
Validation chooses whether to keep the feature. Test predictions are
written only when the choice beats the shared-penalty control.
"""

import numpy as np
import pandas as pd

import train_gate

PACK = list(train_gate.PACK)


def add_llr(frame):
    scores = pd.read_csv(train_gate.OUT / "esm_llr.csv")
    scores["sample_id"] = scores["sample_id"].astype(str)
    frame = frame.merge(scores[["sample_id", "esm_llr"]], on="sample_id", how="left")
    frame["esm_prox"] = frame["esm_llr"] * frame["prox"]
    finite = int(np.isfinite(pd.to_numeric(frame["esm_llr"], errors="coerce")).sum())
    print("esm_llr_finite_rows", finite, "of", len(frame), flush=True)
    return frame


def configs():
    esm = PACK + [f"pc{i}" for i in range(32)]
    extra = ["esm_llr", "esm_prox"]
    return [
        ("control_esm_a1e5", "between", esm, 100000.0),
        ("plus_llr_a1e5", "between", esm + extra, 100000.0),
        ("plus_llr_a3e5", "between", esm + extra, 300000.0),
        ("pack_llr_a300", "between", PACK + extra, 300.0),
        ("llr_only_a30", "between", ["esm_llr", "heavy_dist"], 30.0),
    ]


def main():
    frame, vectors = train_gate.load()
    frame = add_llr(frame)
    table = []
    for key, mode, cols, alpha in configs():
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred, _ = train_gate.fit_predict_fold(train, val, cols, alpha, mode, vectors)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, mode, cols, alpha, mean, scores))
        shown = [None if not np.isfinite(s) else round(float(s), 4) for s in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[4])
    chosen, mode, cols, alpha, mean, _ = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control_esm_a1e5":
        print("control remains best on validation; test file stays predictions_gate.csv", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred, _ = train_gate.fit_predict_fold(train, test, cols, alpha, mode, vectors)
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
                "model": "esmllr",
            })
    base = pd.read_csv(train_gate.OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(train_gate.OUT / "predictions_esmllr.csv", index=False)
    print("wrote predictions_esmllr.csv", len(rows), flush=True)


if __name__ == "__main__":
    main()
