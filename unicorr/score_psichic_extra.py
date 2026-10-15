"""Extract PSICHIC residue attention at the mutated site and class scores.

Reuses the protein cache from score_psichic.py. Writes psichic_extra.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch

REPO = Path("/root/autodl-tmp/project4/third_party/PSICHIC")
PROD = REPO / "PSICHIC-prod"
OUT = Path("/root/autodl-tmp/project4/unseen_kd")
SEQ = Path("/root/autodl-tmp/project/feature/sequence")
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROD))

import train_esm650
from inference import _post_transform_ligand, _run_forward, _unbatch_nodes, build_data_object, load_model
from utils.ligand_init import smiles2graph


def read_seq(path):
    if not path.exists():
        return ""
    return "".join(path.read_text().split()).upper()


def canonical(text):
    mol = Chem.MolFromSmiles(str(text))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mutation_index(sample_id, mutation, wt_seq, mt_seq):
    parsed = train_esm650.parse_mutation(mutation)
    if parsed is None or not wt_seq:
        return None
    wt, resnum, mut = parsed
    return train_esm650.site_index(wt_seq, mt_seq, wt, resnum, mut)


def main():
    cache_path = OUT / "psichic_prot_cache.pt"
    proteins = torch.load(cache_path, map_location="cpu", weights_only=False)
    print("protein cache", len(proteins), flush=True)
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
    samples = []
    for rec in frame.drop_duplicates("sample_id").to_dict("records"):
        sid = str(rec["sample_id"])
        wt = read_seq(SEQ / sid / "wt_seq.txt")
        mt = read_seq(SEQ / sid / "mt_seq.txt")
        smiles = canonical(rec.get("lig.canonical_smiles"))
        index = mutation_index(sid, rec["mutation"], wt, mt)
        samples.append({
            "sample_id": sid,
            "wt_seq": wt,
            "mt_seq": mt,
            "smiles": smiles or "",
            "site": index if index is not None else -1,
        })
    table = pd.DataFrame(samples)
    ligands = {}
    for smiles in sorted(set(table["smiles"])):
        if not smiles:
            continue
        try:
            ligands[smiles] = _post_transform_ligand(smiles2graph(smiles))
        except Exception as exc:
            print("ligand fail", smiles[:40], type(exc).__name__, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("multitask_PSICHIC", device)
    rows = []
    batch_pairs = []
    meta = []

    def flush():
        if not batch_pairs:
            return
        data_objects = []
        keep_meta = []
        for item, (sid, which, site, smiles, seq) in zip(batch_pairs, meta):
            mol = ligands.get(smiles)
            prot = proteins.get(seq)
            if mol is None or prot is None:
                rows.append({
                    "sample_id": sid, "which": which,
                    "reg": np.nan, "nb": np.nan, "attn": np.nan, "site": site,
                })
                continue
            data_objects.append(build_data_object(mol, prot))
            keep_meta.append((sid, which, site))
        batch_pairs.clear()
        meta.clear()
        if not data_objects:
            return
        batch = Batch.from_data_list(
            data_objects, follow_batch=["mol_x", "clique_x", "prot_node_aa"]
        ).to(device)
        reg_pred, _cls, mcls_pred, attention_dict = _run_forward(
            model, batch, False, save_cluster=True
        )
        reg = np.atleast_1d(reg_pred.squeeze().detach().cpu().float().numpy())
        mcls = torch.softmax(mcls_pred, dim=-1).detach().cpu().float().numpy()
        prot_batch_idx = attention_dict["protein_residue_index"]
        residue_scores = _unbatch_nodes(attention_dict["residue_final_score"], prot_batch_idx)
        for i, (sid, which, site) in enumerate(keep_meta):
            attn = np.nan
            scores = residue_scores[i].detach().cpu().float().numpy().reshape(-1)
            if 0 <= site < len(scores):
                attn = float(scores[site])
            rows.append({
                "sample_id": sid,
                "which": which,
                "reg": float(reg[i]),
                "nb": float(mcls[i, -1]) if mcls.ndim == 2 else np.nan,
                "attn": attn,
                "site": site,
            })

    for count, rec in enumerate(table.itertuples(index=False), start=1):
        if not rec.smiles or rec.site < 0 or not rec.wt_seq or not rec.mt_seq:
            rows.append({"sample_id": rec.sample_id, "which": "wt", "reg": np.nan, "nb": np.nan, "attn": np.nan, "site": rec.site})
            rows.append({"sample_id": rec.sample_id, "which": "mt", "reg": np.nan, "nb": np.nan, "attn": np.nan, "site": rec.site})
            continue
        batch_pairs.append(1)
        meta.append((rec.sample_id, "wt", int(rec.site), rec.smiles, rec.wt_seq))
        batch_pairs.append(1)
        meta.append((rec.sample_id, "mt", int(rec.site), rec.smiles, rec.mt_seq))
        if len(batch_pairs) >= 4:
            flush()
        if count % 50 == 0:
            print("extra", count, flush=True)
    flush()
    long = pd.DataFrame(rows)
    wide = []
    for sample_id, part in long.groupby("sample_id"):
        get = part.set_index("which")
        wt = get.loc["wt"] if "wt" in get.index else None
        mt = get.loc["mt"] if "mt" in get.index else None
        wide.append({
            "sample_id": sample_id,
            "psichic_wt": float(wt["reg"]) if wt is not None else np.nan,
            "psichic_mt": float(mt["reg"]) if mt is not None else np.nan,
            "nb_wt": float(wt["nb"]) if wt is not None else np.nan,
            "nb_mt": float(mt["nb"]) if mt is not None else np.nan,
            "attn_wt": float(wt["attn"]) if wt is not None else np.nan,
            "attn_mt": float(mt["attn"]) if mt is not None else np.nan,
            "site": int(wt["site"]) if wt is not None else -1,
        })
    out = pd.DataFrame(wide)
    out["nb_delta"] = out["nb_mt"] - out["nb_wt"]
    out["attn_delta"] = out["attn_mt"] - out["attn_wt"]
    out["psichic_delta"] = out["psichic_mt"] - out["psichic_wt"]
    out.to_csv(OUT / "psichic_extra.csv", index=False)
    print(
        "wrote", len(out),
        "nb", int(np.isfinite(out["nb_delta"]).sum()),
        "attn", int(np.isfinite(out["attn_delta"]).sum()),
        flush=True,
    )


if __name__ == "__main__":
    main()
