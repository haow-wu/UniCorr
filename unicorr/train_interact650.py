"""One low-capacity interaction, plus a check that Kd and Ki signs agree.

esm650 log-odds times proximity to a polar ligand atom is not in the
current linear model. Validation decides whether to add it.
"""

import numpy as np
import pandas as pd

import train_decay
import train_esm650h
import train_gate


def sign_check():
    cohort = pd.read_csv(train_gate.OUT / "cohort_rows.csv")
    for unit, part in cohort.groupby(cohort["label_type"].astype(str)):
        y = pd.to_numeric(part["ddg_from_k"], errors="coerce")
        ref = pd.to_numeric(part["ddg_exp"], errors="coerce")
        mask = y.notna() & ref.notna()
        if int(mask.sum()) < 30 or y[mask].std() < 1e-8 or ref[mask].std() < 1e-8:
            print("sign", unit, "n", int(mask.sum()), "pearson", None, flush=True)
            continue
        corr = float(np.corrcoef(y[mask], ref[mask])[0, 1])
        print("sign", unit, "n", int(mask.sum()), "pearson", round(corr, 4), flush=True)


def add_interaction(frame):
    rot = pd.concat([
        pd.read_csv("/root/autodl-tmp/project4/cache/rotamer_features.csv"),
        pd.read_csv(train_gate.OUT / "rotamer_extra.csv"),
    ], ignore_index=True)
    rot["sample_id"] = rot["graph_id"].astype(str)
    rot = rot.drop_duplicates("sample_id", keep="last")
    near = pd.to_numeric(rot.get("sh_O_35"), errors="coerce").fillna(0)
    near = near + pd.to_numeric(rot.get("sh_O_45"), errors="coerce").fillna(0)
    near = near + pd.to_numeric(rot.get("sh_N_35"), errors="coerce").fillna(0)
    near = near + pd.to_numeric(rot.get("sh_N_45"), errors="coerce").fillna(0)
    rot = rot.assign(near_polar=near.to_numpy())
    frame = frame.merge(rot[["sample_id", "near_polar"]], on="sample_id", how="left")
    frame["llr_x_polar"] = frame["esm650_llr"] * frame["near_polar"]
    frame["llr_x_close"] = frame["esm650_llr"] * (pd.to_numeric(frame["heavy_dist"], errors="coerce") <= 5).astype(float)
    return frame


def fit_extra(train, other, vectors):
    extra = ["llr_x_polar", "llr_x_close"]
    train, other, columns = train_esm650h.with_hidden(
        train, other, vectors, train_esm650h.WIDE + extra, 16
    )
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
    sign_check()
    frame, vectors = train_gate.load()
    import train_esm650
    import train_esmllr
    frame = train_esmllr.add_llr(frame)
    frame = train_esm650.add_650(frame)
    frame = add_interaction(frame)
    # Distance mix of each test fold, labels not used.
    for fold in range(5):
        test = frame.loc[(frame["fold"] == fold) & (frame["split"] == "test")]
        heavy = pd.to_numeric(test["heavy_dist"], errors="coerce")
        print(
            f"test_fold {fold} n {len(test)} close5 {int((heavy <= 5).sum())} "
            f"mid {int(((heavy > 5) & (heavy <= 15)).sum())} far {int((heavy > 15).sum())}",
            flush=True,
        )
    options = ("control", "interact")
    table = []
    for key in options:
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            if key == "control":
                pred = train_esm650h.predict(train, val, vectors, 16, "exp", 10.0)
            else:
                pred = fit_extra(train, val, vectors)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, mean, scores))
        shown = [None if not np.isfinite(item) else round(float(item), 4) for item in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[1])
    chosen, mean, _ = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control":
        print("control remains best on validation; test file stays predictions_esm650h.csv", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = fit_extra(train, test, vectors)
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
                "model": "interact",
            })
    base = pd.read_csv(train_gate.OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(train_gate.OUT / "predictions_interact650.csv", index=False)
    print("wrote predictions_interact650.csv", len(rows), flush=True)


if __name__ == "__main__":
    main()
