"""ESM-2 650M masked log-odds, then a validation choice against the 0.244 model.

The score is logit(wild type) minus logit(mutant) on the wild-type sequence.
The distance gate and whether to keep the 650M score are chosen on validation.
Test predictions are written only when that choice beats the current model.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmForMaskedLM, EsmTokenizer

import train_decay
import train_esmllr
import train_gate

OUT = Path("/root/autodl-tmp/project4/unseen_kd")
SEQ = Path("/root/autodl-tmp/project/feature/sequence")
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
BASE = train_decay.ESM
ALPHA = train_decay.ALPHA


def find_snapshot():
    roots = [
        Path("/root/autodl-tmp/cache/huggingface/hub"),
        Path("/root/.cache/huggingface/hub"),
    ]
    for root in roots:
        matches = list(root.glob("models--facebook--esm2_t33_650M_UR50D/snapshots/*"))
        matches = [path for path in matches if (path / "config.json").exists()]
        if matches:
            return matches[0]
    raise SystemExit("esm2 650M snapshot not found")


def read_seq(path):
    if not path.exists():
        return ""
    return "".join(ch for ch in path.read_text(errors="ignore") if ch.isalpha()).upper()


def parse_mutation(text):
    raw = str(text).strip().upper().replace(" ", "")
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", raw)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    match = re.fullmatch(r"([A-Z]{3})(\d+)([A-Z]{3})", raw)
    if match and match.group(1) in AA3 and match.group(3) in AA3:
        return AA3[match.group(1)], int(match.group(2)), AA3[match.group(3)]
    return None


def site_index(seq, mt_seq, wt, resnum, mut):
    if 1 <= resnum <= len(seq) and seq[resnum - 1] == wt:
        return resnum - 1
    if mt_seq and len(mt_seq) == len(seq):
        hits = [i for i, (a, b) in enumerate(zip(seq, mt_seq)) if a == wt and b == mut and a != b]
        if len(hits) == 1:
            return hits[0]
    hits = [i for i, a in enumerate(seq) if a == wt]
    if len(hits) == 1:
        return hits[0]
    return None


def window(seq, index, limit=1022):
    if len(seq) <= limit:
        return seq, index
    start = max(0, min(index - limit // 2, len(seq) - limit))
    return seq[start:start + limit], index - start


def save_scores():
    dest = OUT / "esm650_llr.csv"
    if dest.exists():
        frame = pd.read_csv(dest)
        finite = int(np.isfinite(pd.to_numeric(frame["esm650_llr"], errors="coerce")).sum())
        if finite >= 500:
            print("reuse", dest, "finite", finite, flush=True)
            return
    snapshot = find_snapshot()
    print("snapshot", snapshot, flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(snapshot, local_files_only=True).to(device)
    model.eval()
    token_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    cohort = pd.read_csv(OUT / "cohort_rows.csv")
    cohort["sample_id"] = cohort["SAMPLE_ID"].astype(str)
    rows = []
    for count, rec in enumerate(cohort.drop_duplicates("sample_id").itertuples(index=False), start=1):
        parsed = parse_mutation(rec.mutation)
        seq = read_seq(SEQ / rec.sample_id / "wt_seq.txt")
        mt_seq = read_seq(SEQ / rec.sample_id / "mt_seq.txt")
        score = np.nan
        if parsed is not None and seq:
            wt, resnum, mut = parsed
            index = site_index(seq, mt_seq, wt, resnum, mut)
            if index is not None and wt in token_id and mut in token_id:
                piece, local = window(seq, index)
                tokens = tokenizer(piece, return_tensors="pt")
                pos = local + 1
                tokens["input_ids"][0, pos] = tokenizer.mask_token_id
                tokens = {key: value.to(device) for key, value in tokens.items()}
                with torch.no_grad():
                    logits = model(**tokens).logits[0, pos]
                score = float(logits[token_id[wt]] - logits[token_id[mut]])
        rows.append({"sample_id": rec.sample_id, "esm650_llr": score})
        if count % 50 == 0:
            print("scored", count, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest, index=False)
    print("wrote", dest, "finite", int(np.isfinite(frame["esm650_llr"]).sum()), flush=True)


def add_650(frame):
    scores = pd.read_csv(OUT / "esm650_llr.csv")
    scores["sample_id"] = scores["sample_id"].astype(str)
    frame = frame.merge(scores[["sample_id", "esm650_llr"]], on="sample_id", how="left")
    frame["esm650_prox"] = frame["esm650_llr"] * frame["prox"]
    return frame


def predict_fold(train, other, columns, kind, param, vectors):
    pred, _ = train_gate.fit_predict_fold(train, other, columns, ALPHA, "between", vectors)
    return train_decay.gated(
        pred,
        other["complex"].astype(str).to_numpy(),
        other["heavy_dist"].to_numpy(dtype=float),
        kind,
        param,
    )


def main():
    save_scores()
    frame, vectors = train_gate.load()
    frame = train_esmllr.add_llr(frame)
    frame = add_650(frame)
    wide = BASE + ["esm650_llr", "esm650_prox"]
    options = [
        ("control_hinge15", BASE, "hinge", 15.0),
        ("wide_ungated", wide, "none", None),
        ("wide_hinge15", wide, "hinge", 15.0),
        ("wide_exp10", wide, "exp", 10.0),
    ]
    table = []
    for key, columns, kind, param in options:
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred = predict_fold(train, val, columns, kind, param, vectors)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, columns, kind, param, mean, scores))
        shown = [None if not np.isfinite(item) else round(float(item), 4) for item in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[4])
    chosen, columns, kind, param, mean, _ = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control_hinge15":
        print("control remains best on validation; test file stays predictions_decay.csv", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = predict_fold(train, test, columns, kind, param, vectors)
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
                "model": "esm650",
            })
    base = pd.read_csv(OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(OUT / "predictions_esm650.csv", index=False)
    print("wrote predictions_esm650.csv", len(rows), flush=True)


if __name__ == "__main__":
    main()
