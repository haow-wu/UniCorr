"""Score ThermoMPNN stability ΔΔG for each cohort mutation on the WT backbone.

ThermoMPNN is protein-only. Validation decides whether the score helps the
binding model. Split and labels are not changed.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path("/root/autodl-tmp/project4/third_party/ThermoMPNN")
OUT = Path("/root/autodl-tmp/project4/unseen_kd")
DATA = Path("/root/autodl-tmp/project/data/Single_Substitution")
SEQ = Path("/root/autodl-tmp/project/feature/sequence")
MPNN_WT = Path("/root/autodl-tmp/project4/third_party/mpnn/v_48_020.pt")

sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import train_esm650
from datasets import Mutation
from protein_mpnn_utils import alt_parse_PDB
from train_thermompnn import TransferModelPL


def protein_only_pdb(src, dest):
    """Write ATOM records only so ligand HETATM does not confuse the parser."""
    lines = []
    with open(src) as handle:
        for line in handle:
            if line.startswith("ATOM"):
                lines.append(line)
            elif line.startswith("TER") or line.startswith("END"):
                lines.append(line)
    if not lines:
        return False
    if not any(line.startswith("END") for line in lines):
        lines.append("END\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines))
    return True


def site_in_pdb(pdb_dict, wt, resnum, mut, wt_seq, mt_seq):
    seq = pdb_dict["seq"].replace("-", "")
    # alt_parse keeps gaps as '-'; index into the tensor uses the gapped seq.
    full = pdb_dict["seq"]
    # Prefer sequence alignment via our site_index on ungapped seq, then map.
    index = train_esm650.site_index(wt_seq, mt_seq, wt, resnum, mut)
    if index is not None and index < len(seq) and seq[index] == wt:
        # Map ungapped index to gapped index
        seen = -1
        for i, aa in enumerate(full):
            if aa != "-":
                seen += 1
            if seen == index:
                return i
    # Fallback: PDB resseq match if available
    if "resseq" in pdb_dict or "residue_numbers" in pdb_dict:
        pass
    hits = [i for i, aa in enumerate(full) if aa == wt]
    if len(hits) == 1:
        return hits[0]
    # Try 0-based resnum-1 if in range
    if 1 <= resnum <= len(full) and full[resnum - 1] == wt:
        return resnum - 1
    return None


def load_model():
    weight = ROOT / "models" / "thermoMPNN_default.pt"
    vanilla_dir = ROOT / "vanilla_model_weights"
    vanilla_dir.mkdir(parents=True, exist_ok=True)
    target = vanilla_dir / "v_48_020.pt"
    if not target.exists() and MPNN_WT.exists():
        if target.exists() or target.is_symlink():
            pass
        else:
            os.symlink(str(MPNN_WT), str(target))
    config = {
        "training": {
            "num_workers": 2,
            "learn_rate": 0.001,
            "epochs": 100,
            "lr_schedule": True,
        },
        "model": {
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True,
            "lr_schedule": True,
        },
        "platform": {
            "thermompnn_dir": str(ROOT),
            "accel": "gpu",
            "cache_dir": str(OUT / "thermompnn_cache"),
        },
    }
    cfg = OmegaConf.create(config)
    model = TransferModelPL.load_from_checkpoint(str(weight), cfg=cfg).model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().to(device)


def main():
    sys.path.insert(0, "/root/autodl-tmp/project4/src")
    from structure_io import find_wt_complex

    cohort = pd.read_csv(OUT / "cohort_rows.csv")
    cohort["sample_id"] = cohort["SAMPLE_ID"].astype(str)
    model = load_model()
    dest = OUT / "thermompnn_ddg.csv"
    done = {}
    if dest.exists():
        prev = pd.read_csv(dest)
        for rec in prev.itertuples(index=False):
            if np.isfinite(rec.thermompnn_ddg):
                done[str(rec.sample_id)] = float(rec.thermompnn_ddg)
        print("resume", len(done), flush=True)

    rows = []
    tmp_dir = OUT / "thermompnn_pdb"
    for count, rec in enumerate(cohort.drop_duplicates("sample_id").itertuples(index=False), start=1):
        sid = str(rec.sample_id)
        if sid in done:
            rows.append({"sample_id": sid, "thermompnn_ddg": done[sid]})
            continue
        parsed = train_esm650.parse_mutation(rec.mutation)
        value = np.nan
        if parsed is not None:
            wt, resnum, mut = parsed
            src = find_wt_complex(str(DATA / sid))
            if src is not None:
                local = tmp_dir / f"{sid}.pdb"
                if protein_only_pdb(src, local):
                    try:
                        pdb = alt_parse_PDB(str(local), None)
                        wt_seq = train_esm650.read_seq(SEQ / sid / "wt_seq.txt")
                        mt_seq = train_esm650.read_seq(SEQ / sid / "mt_seq.txt")
                        pos = site_in_pdb(pdb[0], wt, resnum, mut, wt_seq, mt_seq)
                        if pos is not None and pdb[0]["seq"][pos] == wt:
                            mutation = Mutation(position=pos, wildtype=wt, mutation=mut, ddG=None, pdb=pdb[0]["name"])
                            with torch.no_grad():
                                pred, _ = model(pdb, [mutation])
                            value = float(pred[0]["ddG"].detach().cpu().item())
                    except Exception as exc:
                        print("fail", sid, type(exc).__name__, exc, flush=True)
        rows.append({"sample_id": sid, "thermompnn_ddg": value})
        if count % 25 == 0:
            frame = pd.DataFrame(rows)
            frame.to_csv(dest, index=False)
            finite = int(np.isfinite(pd.to_numeric(frame["thermompnn_ddg"], errors="coerce")).sum())
            print("thermompnn", count, "finite", finite, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest, index=False)
    print("wrote", dest, "finite", int(np.isfinite(frame["thermompnn_ddg"]).sum()), flush=True)


if __name__ == "__main__":
    main()
