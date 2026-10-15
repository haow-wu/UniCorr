"""Add the ESM-2 650M masked-site hidden vector.

The 650M log-odds used so far are one contrast of the final logits.
This keeps the 1280-dimensional hidden state at the masked site, reduces
it with a train-only PCA, and keeps the vector only when validation
improves on the current model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from transformers import EsmForMaskedLM, EsmTokenizer

import train_decay
import train_esm650
import train_gate

OUT = Path("/root/autodl-tmp/project4/unseen_kd")
SEQ = Path("/root/autodl-tmp/project/feature/sequence")
HIDDEN = OUT / "esm650_hidden.npz"
WIDE = train_decay.ESM + ["esm650_llr", "esm650_prox"]


def save_hidden():
    if HIDDEN.exists():
        pack = np.load(HIDDEN, allow_pickle=True)
        if len(pack["sample_id"]) >= 500 and np.isfinite(pack["hidden"]).mean() > 0.9:
            print("reuse", HIDDEN, pack["hidden"].shape, flush=True)
            return
    snapshot = train_esm650.find_snapshot()
    print("snapshot", snapshot, flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(snapshot, local_files_only=True).to(device)
    model.eval()
    cohort = pd.read_csv(OUT / "cohort_rows.csv")
    cohort["sample_id"] = cohort["SAMPLE_ID"].astype(str)
    ids, rows = [], []
    width = None
    for count, rec in enumerate(cohort.drop_duplicates("sample_id").itertuples(index=False), start=1):
        parsed = train_esm650.parse_mutation(rec.mutation)
        seq = train_esm650.read_seq(SEQ / rec.sample_id / "wt_seq.txt")
        mt_seq = train_esm650.read_seq(SEQ / rec.sample_id / "mt_seq.txt")
        vector = None
        if parsed is not None and seq:
            wt, resnum, mut = parsed
            index = train_esm650.site_index(seq, mt_seq, wt, resnum, mut)
            if index is not None:
                piece, local = train_esm650.window(seq, index)
                tokens = tokenizer(piece, return_tensors="pt")
                pos = local + 1
                tokens["input_ids"][0, pos] = tokenizer.mask_token_id
                tokens = {key: value.to(device) for key, value in tokens.items()}
                with torch.no_grad():
                    out = model(**tokens, output_hidden_states=True)
                vector = out.hidden_states[-1][0, pos].detach().float().cpu().numpy()
                width = vector.shape[0]
        ids.append(rec.sample_id)
        rows.append(vector)
        if count % 50 == 0:
            print("hidden", count, flush=True)
    width = 1280 if width is None else width
    matrix = np.vstack([
        np.full(width, np.nan, dtype=np.float32) if row is None else row.astype(np.float32)
        for row in rows
    ])
    np.savez_compressed(HIDDEN, sample_id=np.array(ids), hidden=matrix)
    print("wrote", HIDDEN, matrix.shape, "finite_rows", int(np.isfinite(matrix).all(axis=1).sum()), flush=True)


def hidden_table():
    pack = np.load(HIDDEN, allow_pickle=True)
    return {str(i): np.asarray(row, float) for i, row in zip(pack["sample_id"], pack["hidden"])}


def with_hidden(train, other, vectors, base_columns, k):
    table = hidden_table()

    def stack(frame):
        rows = []
        width = len(next(iter(table.values())))
        for gid in frame["sample_id"].astype(str):
            vec = table.get(gid)
            rows.append(np.full(width, np.nan) if vec is None else vec)
        return np.vstack(rows)

    train, other, columns = train_gate.with_pcs(train, other, vectors, base_columns)
    imp = SimpleImputer(strategy="median")
    pca = PCA(n_components=k, random_state=42)
    z_tr = pca.fit_transform(imp.fit_transform(stack(train)))
    z_ot = pca.transform(imp.transform(stack(other)))
    names = [f"h{i}" for i in range(k)]
    train = train.copy()
    other = other.copy()
    for i, name in enumerate(names):
        train[name] = z_tr[:, i]
        other[name] = z_ot[:, i]
    return train, other, columns + names


def predict(train, other, vectors, k, kind, param):
    if k == 0:
        return train_esm650.predict_fold(train, other, WIDE, kind, param, vectors)
    train, other, columns = with_hidden(train, other, vectors, WIDE, k)
    pred, _ = train_gate.fit_predict_fold(train, other, columns, train_decay.ALPHA, "between", vectors)
    return train_decay.gated(
        pred,
        other["complex"].astype(str).to_numpy(),
        other["heavy_dist"].to_numpy(dtype=float),
        kind,
        param,
    )


def main():
    save_hidden()
    frame, vectors = train_gate.load()
    import train_esmllr
    frame = train_esmllr.add_llr(frame)
    frame = train_esm650.add_650(frame)
    options = [
        ("control_exp10", 0, "exp", 10.0),
        ("h16_exp10", 16, "exp", 10.0),
        ("h32_exp10", 32, "exp", 10.0),
        ("h32_hinge15", 32, "hinge", 15.0),
    ]
    table = []
    for key, k, kind, param in options:
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred = predict(train, val, vectors, k, kind, param)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, k, kind, param, mean, scores))
        shown = [None if not np.isfinite(item) else round(float(item), 4) for item in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[4])
    chosen, k, kind, param, mean, _ = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control_exp10":
        print("control remains best on validation; test file stays predictions_esm650.csv", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = predict(train, test, vectors, k, kind, param)
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
                "model": "esm650h",
            })
    base = pd.read_csv(OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(OUT / "predictions_esm650h.csv", index=False)
    print("wrote predictions_esm650h.csv", len(rows), flush=True)


if __name__ == "__main__":
    main()
