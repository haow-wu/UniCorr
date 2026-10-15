"""Ligand-conditioned score: PSICHIC(mutant, ligand) minus PSICHIC(wild type, ligand).

Uses the multitask checkpoint, not the PDBBind-finetuned one. The frozen
UniProt split is not changed. Test predictions are written only when the
new column beats the interaction ridge on validation proteins.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

REPO = Path("/root/autodl-tmp/project4/third_party/PSICHIC")
PROD = REPO / "PSICHIC-prod"
OUT = Path("/root/autodl-tmp/project4/unseen_kd")
SEQ = Path("/root/autodl-tmp/project/feature/sequence")
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROD))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_psichic")


def read_seq(path):
    if not path.exists():
        return ""
    return "".join(path.read_text().split())


def canonical(text):
    mol = Chem.MolFromSmiles(str(text))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_samples():
    cohort = pd.read_csv(OUT / "cohort_rows.csv")
    cohort["sample_id"] = cohort["SAMPLE_ID"].astype(str)
    cohort["uniprot"] = cohort["uniprot"].astype(str).str.upper()
    cohort["mutation_key"] = cohort["mutation"].astype(str).str.upper().str.replace(" ", "", regex=False)
    cohort["pdb_key"] = cohort["pdb"].astype(str).str.upper().str[:4]
    plat = pd.read_csv(OUT / "platinum_flat_file.csv", low_memory=False)
    plat["uniprot"] = plat["mut.uniprot"].astype(str).str.upper()
    plat["mutation_key"] = plat["mutation"].astype(str).str.upper().str.replace(" ", "", regex=False)
    plat["pdb_key"] = plat["affin.pdb_id"].astype(str).str.upper().str[:4]
    plat = plat[["uniprot", "mutation_key", "pdb_key", "lig.canonical_smiles"]].drop_duplicates(
        ["uniprot", "mutation_key", "pdb_key"]
    )
    frame = cohort.merge(plat, on=["uniprot", "mutation_key", "pdb_key"], how="left")
    rows = []
    for rec in frame.drop_duplicates("sample_id").to_dict("records"):
        sample_id = str(rec["sample_id"])
        wt = read_seq(SEQ / sample_id / "wt_seq.txt").upper()
        mt = read_seq(SEQ / sample_id / "mt_seq.txt").upper()
        smiles = canonical(rec.get("lig.canonical_smiles"))
        rows.append({
            "sample_id": sample_id,
            "wt_seq": wt,
            "mt_seq": mt,
            "smiles": smiles if smiles else "",
        })
    table = pd.DataFrame(rows)
    print(
        "samples", len(table),
        "smiles", int(table["smiles"].astype(bool).sum()),
        "wt", int(table["wt_seq"].astype(bool).sum()),
        "mt", int(table["mt_seq"].astype(bool).sum()),
        flush=True,
    )
    return table


def protein_features(sequences):
    cache_path = OUT / "psichic_prot_cache.pt"
    cached = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        print("protein cache", len(cached), flush=True)
    missing = [seq for seq in sequences if seq and seq not in cached]
    if not missing:
        return cached
    from esm_batched.esm_batched import batched_protein_init
    from utils.protein_init import contact_map, seq_feature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("esm sequences", len(missing), "device", device, flush=True)
    fresh = batched_protein_init(
        missing, device=device, seq_feature_fn=seq_feature, contact_map_fn=contact_map
    )
    for seq, feat in fresh.items():
        feat["seq_feat"] = feat["seq_feat"].float().cpu()
        feat["token_representation"] = feat["token_representation"].float().cpu()
        feat["edge_weight"] = feat["edge_weight"].float().cpu()
        feat["edge_index"] = feat["edge_index"].long().cpu()
        cached[seq] = feat
    torch.save(cached, cache_path)
    print("saved protein cache", len(cached), flush=True)
    return cached


def wait_weights():
    path = REPO / "trained_weights" / "multitask_PSICHIC" / "model.pt"
    deadline = time.time() + 40 * 60
    while time.time() < deadline:
        if path.exists() and path.stat().st_size >= 23000000:
            print("weights", path.stat().st_size, flush=True)
            return path
        print("waiting for model.pt", flush=True)
        time.sleep(30)
    raise SystemExit("运行失败: multitask model.pt 未下载完成")


def score_pairs(table, proteins):
    from inference import _post_transform_ligand, _score_batch, load_model
    from utils.ligand_init import smiles2graph

    wait_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ligands = {}
    for smiles in sorted(set(table["smiles"])):
        if not smiles:
            continue
        try:
            ligands[smiles] = _post_transform_ligand(smiles2graph(smiles))
        except Exception as exc:
            print("ligand fail", smiles[:40], type(exc).__name__, exc, flush=True)
    print("ligands", len(ligands), flush=True)
    model = load_model("multitask_PSICHIC", device)
    out_path = OUT / "psichic_delta.csv"
    done = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        for rec in prev.itertuples(index=False):
            if np.isfinite(rec.psichic_delta):
                done[str(rec.sample_id)] = rec
        print("resume", len(done), flush=True)

    pending = []
    for rec in table.itertuples(index=False):
        if rec.sample_id in done:
            continue
        if not rec.smiles or rec.smiles not in ligands or not rec.wt_seq or not rec.mt_seq:
            pending.append((rec.sample_id, np.nan, np.nan))
            continue
        if rec.wt_seq not in proteins or rec.mt_seq not in proteins:
            pending.append((rec.sample_id, np.nan, np.nan))
            continue
        pending.append(rec)

    rows = []
    batch = []
    owners = []

    def flush():
        if not batch:
            return
        reg, _cls, _mcls = _score_batch(
            batch, ligands, proteins, model, False, False, True, device
        )
        for (sample_id, which), value in zip(owners, reg.tolist()):
            rows.append((sample_id, which, float(value)))
        batch.clear()
        owners.clear()

    real = [item for item in pending if hasattr(item, "wt_seq")]
    skipped = [item for item in pending if not hasattr(item, "wt_seq")]
    for count, rec in enumerate(real, start=1):
        batch.append((rec.sample_id, rec.wt_seq, rec.smiles, rec.smiles))
        owners.append((rec.sample_id, "wt"))
        batch.append((rec.sample_id, rec.mt_seq, rec.smiles, rec.smiles))
        owners.append((rec.sample_id, "mt"))
        if len(batch) >= 4:
            flush()
        if count % 25 == 0:
            print("scored", count, "/", len(real), flush=True)
    flush()
    wide = {}
    for sample_id, which, value in rows:
        wide.setdefault(sample_id, {})[which] = value
    records = []
    for sample_id, payload in wide.items():
        wt = payload.get("wt", np.nan)
        mt = payload.get("mt", np.nan)
        records.append({
            "sample_id": sample_id,
            "psichic_wt": wt,
            "psichic_mt": mt,
            "psichic_delta": mt - wt if np.isfinite(wt) and np.isfinite(mt) else np.nan,
        })
    for sample_id, _wt, _mt in skipped:
        records.append({
            "sample_id": sample_id,
            "psichic_wt": np.nan,
            "psichic_mt": np.nan,
            "psichic_delta": np.nan,
        })
    for sample_id, rec in done.items():
        records.append({
            "sample_id": sample_id,
            "psichic_wt": rec.psichic_wt,
            "psichic_mt": rec.psichic_mt,
            "psichic_delta": rec.psichic_delta,
        })
    result = pd.DataFrame(records).drop_duplicates("sample_id")
    result.to_csv(out_path, index=False)
    finite = int(np.isfinite(pd.to_numeric(result["psichic_delta"], errors="coerce")).sum())
    print("wrote", out_path, "rows", len(result), "finite", finite, flush=True)
    return result


def maybe_train(delta):
    import train_decay
    import train_esm650
    import train_esm650h
    import train_esmllr
    import train_gate
    import train_hbens
    import train_interact650
    import train_stable

    frame, vectors = train_gate.load()
    frame = train_esmllr.add_llr(frame)
    frame = train_esm650.add_650(frame)
    frame = train_interact650.add_interaction(frame)
    frame = frame.merge(delta[["sample_id", "psichic_delta"]], on="sample_id", how="left")
    print("merged finite", int(np.isfinite(frame["psichic_delta"]).sum()), flush=True)
    val_signs = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        val = part.loc[part["split"] == "val"]
        tr = train_stable.within_score(train, "psichic_delta")
        va = train_stable.within_score(val, "psichic_delta")
        val_signs.append(va)
        print("within", fold, "train", tr, "val", va, flush=True)
    defined = [item for item in val_signs if item is not None]
    if len(defined) < 4 or not (all(item > 0 for item in defined) or all(item < 0 for item in defined)):
        print("psichic delta rejected: validation signs disagree", flush=True)
        return
    options = {"control": [], "psichic": ["psichic_delta"]}
    table = []
    for key, extra in options.items():
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred = train_hbens.fit_columns(train, val, vectors, extra)
            score = train_gate.pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, mean, extra))
        shown = [None if not np.isfinite(item) else round(float(item), 4) for item in scores]
        print(key, "val_mean", round(mean, 4), "folds", shown, flush=True)
    table.sort(key=lambda item: -item[1])
    chosen, mean, extra = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control":
        print("control remains best on validation; test file unchanged", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred = train_hbens.fit_columns(train, test, vectors, extra)
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
                "model": "psichic",
            })
    base = pd.read_csv(OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    dest = OUT / "predictions_psichic.csv"
    preds.to_csv(dest, index=False)
    print("wrote", dest, len(rows), flush=True)
    import subprocess
    subprocess.check_call([
        sys.executable, str(OUT / "score_metrics.py"),
        str(dest), str(OUT / "metrics_psichic.json"),
    ], cwd=str(OUT))


def main():
    table = load_samples()
    sequences = sorted(set(table["wt_seq"]) | set(table["mt_seq"]))
    sequences = [seq for seq in sequences if seq]
    proteins = protein_features(sequences)
    scored = score_pairs(table, proteins)
    maybe_train(scored)


if __name__ == "__main__":
    main()
