"""Build one feature row per wild-type complex.

The mutation is applied by rebuilding the side chain on the crystallographic
backbone. Mutant coordinates from the database are never read.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from rotlib import (
    ATOM_CHARGE,
    best_rotamer,
    reconstruction_rmsd,
    score_sidechain,
)
from structure_io import find_wt_complex, locate_residue, parse_complex

DATA = os.environ.get("MDR_DATA", "/root/autodl-tmp/project/data")
OUT_DIR = os.environ.get("PROJECT4", "/root/autodl-tmp/project4")
SPLIT_ROOT = os.environ.get("MDR_SPLITS", os.path.join(DATA, "processed"))

SHELL_ELEMENTS = ("C", "N", "O", "S", "X")
SHELL_EDGES = (3.5, 4.5, 6.0)
PHYSICS = (
    "d_rep", "d_att", "d_elec", "d_rep_prot", "d_att_prot",
    "d_clash", "d_contact", "d_hbond", "d_min_dist",
    "x_rep", "x_att", "x_elec", "x_clash", "x_contact", "x_hbond", "x_min_dist",
)
SHELL_COLUMNS = [f"sh_{elem}_{int(edge * 10)}" for edge in SHELL_EDGES for elem in SHELL_ELEMENTS]


def graph_ids():
    found = []
    for split in ("uniprot", "random"):
        for fold in range(5):
            for part in ("train", "val", "test"):
                path = os.path.join(SPLIT_ROOT, f"kfold_{split}", f"fold_{fold}", f"{part}.csv")
                if not os.path.exists(path):
                    continue
                frame = pd.read_csv(path, usecols=["graph_id"])
                found.extend(frame["graph_id"].astype(str).tolist())
    return sorted(set(found))


def _shell(cb, lig_xyz, lig_elem):
    counts = {name: 0.0 for name in SHELL_COLUMNS}
    if len(lig_xyz) == 0:
        return counts
    dist = np.linalg.norm(lig_xyz - cb[None, :], axis=1)
    for edge in SHELL_EDGES:
        low = 0.0 if edge == SHELL_EDGES[0] else SHELL_EDGES[SHELL_EDGES.index(edge) - 1]
        mask = (dist >= low) & (dist < edge)
        for elem in lig_elem[mask]:
            key = elem if elem in SHELL_ELEMENTS else "X"
            counts[f"sh_{key}_{int(edge * 10)}"] += 1.0
    return counts


def _env_atoms(protein, index, cb):
    xyz, elem, charge = [], [], []
    for other, rec in enumerate(protein):
        if other == index:
            continue
        atoms = rec["atoms"]
        coords = np.vstack(list(atoms.values()))
        if float(np.linalg.norm(coords - cb[None, :], axis=1).min()) > 6.0:
            continue
        for name, coord in atoms.items():
            if float(np.linalg.norm(coord - cb)) > 6.0:
                continue
            xyz.append(coord)
            element = "C"
            if name.startswith(("OG", "OD", "OE", "OH", "O")):
                element = "O"
            elif name.startswith(("N", "ND", "NE", "NH", "NZ")):
                element = "N"
            elif name.startswith("S"):
                element = "S"
            elem.append(element)
            charge.append(ATOM_CHARGE.get((rec["resname"], name), 0.0))
    if not xyz:
        return (
            np.zeros((0, 3)),
            np.array([], dtype=object),
            np.zeros((0,)),
        )
    return np.vstack(xyz), np.array(elem, dtype=object), np.asarray(charge, dtype=np.float64)


def featurize_one(graph_id):
    row = {"graph_id": graph_id, "ok": 0, "reason": ""}
    for name in list(PHYSICS) + SHELL_COLUMNS + ["heavy_dist", "ca_dist", "rebuild_rmsd", "n_rot_wt", "n_rot_mut"]:
        row[name] = np.nan
    sample_dir = os.path.join(DATA, "Single_Substitution", graph_id)
    path = find_wt_complex(sample_dir)
    if path is None:
        row["reason"] = "missing_wt"
        return row
    try:
        protein, ligand = parse_complex(path)
    except Exception as exc:
        row["reason"] = f"parse:{type(exc).__name__}"
        return row
    lig_elem = np.array([item[0] for item in ligand], dtype=object)
    lig_xyz = np.vstack([item[1] for item in ligand])
    # The split table carries the residue number. Read it from any fold file
    # via the closure stored on the function by the parent process.
    meta = META.get(graph_id)
    if meta is None:
        row["reason"] = "no_meta"
        return row
    index = locate_residue(protein, meta["wt_aa"], meta["resnum"], lig_xyz)
    if index is None:
        row["reason"] = "residue_not_found"
        return row
    rec = protein[index]
    atoms = rec["atoms"]
    n, ca, c = atoms["N"], atoms["CA"], atoms["C"]
    cb = atoms.get("CB", ca)
    row["ca_dist"] = float(np.linalg.norm(lig_xyz - ca[None, :], axis=1).min())
    residue_xyz = np.vstack(list(atoms.values()))
    row["heavy_dist"] = float(np.linalg.norm(residue_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1).min())
    row["rebuild_rmsd"] = reconstruction_rmsd(meta["wt_aa"], atoms)
    prot_xyz, prot_elem, prot_q = _env_atoms(protein, index, cb)
    crystal = {name: xyz for name, xyz in atoms.items() if name not in ("N", "CA", "C", "O", "OXT")}
    if meta["wt_aa"] == "G":
        crystal = {}
    wt_score, n_wt = best_rotamer(
        meta["wt_aa"], n, ca, c, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q
    )
    mut_score, n_mut = best_rotamer(
        meta["mut_aa"], n, ca, c, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q
    )
    xray = score_sidechain(meta["wt_aa"], crystal, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q)
    row.update(_pack(mut_score, wt_score, xray))
    row.update(_shell(cb, lig_xyz, lig_elem))
    row["n_rot_wt"] = n_wt
    row["n_rot_mut"] = n_mut
    row["ok"] = 1
    row["reason"] = ""
    return row


def _pack(mut, wt, xray):
    return {
        "d_rep": mut["rep_lig"] - wt["rep_lig"],
        "d_att": mut["att_lig"] - wt["att_lig"],
        "d_elec": mut["elec_lig"] - wt["elec_lig"],
        "d_rep_prot": mut["rep_prot"] - wt["rep_prot"],
        "d_att_prot": mut["att_prot"] - wt["att_prot"],
        "d_clash": mut["clash"] - wt["clash"],
        "d_contact": mut["contact"] - wt["contact"],
        "d_hbond": mut["hbond"] - wt["hbond"],
        "d_min_dist": mut["min_dist"] - wt["min_dist"],
        "x_rep": mut["rep_lig"] - xray["rep_lig"],
        "x_att": mut["att_lig"] - xray["att_lig"],
        "x_elec": mut["elec_lig"] - xray["elec_lig"],
        "x_clash": mut["clash"] - xray["clash"],
        "x_contact": mut["contact"] - xray["contact"],
        "x_hbond": mut["hbond"] - xray["hbond"],
        "x_min_dist": mut["min_dist"] - xray["min_dist"],
    }


def _init_worker(meta):
    global META
    META = meta


def load_meta():
    meta = {}
    for split in ("uniprot", "random"):
        for fold in range(5):
            for part in ("train", "val", "test"):
                path = os.path.join(SPLIT_ROOT, f"kfold_{split}", f"fold_{fold}", f"{part}.csv")
                if not os.path.exists(path):
                    continue
                frame = pd.read_csv(path)
                for rec in frame.itertuples(index=False):
                    meta[str(rec.graph_id)] = {
                        "wt_aa": str(rec.wt_aa).strip().upper(),
                        "mut_aa": str(rec.mut_aa).strip().upper(),
                        "resnum": int(rec.resnum),
                    }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cache = os.path.join(OUT_DIR, "cache")
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, "rotamer_features.csv")
    partial = os.path.join(cache, "rotamer_partial.csv")
    ids = graph_ids()
    done = set()
    if args.limit:
        ids = ids[: args.limit]
    elif os.path.exists(partial):
        old = pd.read_csv(partial, usecols=["graph_id"])
        done = set(old["graph_id"].astype(str))
        ids = [item for item in ids if item not in done]
    meta = load_meta()
    print(f"samples {len(ids)} already {len(done)}", flush=True)
    if not ids and os.path.exists(partial):
        frame = pd.read_csv(partial)
        frame.to_csv(dest, index=False)
        print(f"wrote {dest} rows {len(frame)} ok {int(frame.ok.sum())}", flush=True)
        return
    rows = []
    header = not os.path.exists(partial) or bool(args.limit)
    seen_ok = 0

    def flush():
        nonlocal header, rows
        if not rows or args.limit:
            return
        pd.DataFrame(rows).to_csv(partial, mode="a", header=header, index=False)
        header = False
        rows = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(meta,)) as pool:
        for count, row in enumerate(pool.map(featurize_one, ids, chunksize=8), start=1):
            rows.append(row)
            seen_ok += int(row["ok"])
            if count % 200 == 0 or count == len(ids):
                print(f"done {count}/{len(ids)} ok {seen_ok}", flush=True)
                flush()
    if args.limit:
        frame = pd.DataFrame(rows)
        frame.to_csv(dest, index=False)
    else:
        flush()
        frame = pd.read_csv(partial)
        frame.to_csv(dest, index=False)
    print(f"wrote {dest} rows {len(frame)} ok {int(frame.ok.sum())}", flush=True)
    if (frame.ok == 0).any():
        reasons = frame.loc[frame.ok == 0, "reason"].value_counts().head(8)
        print(reasons.to_string(), flush=True)


if __name__ == "__main__":
    main()
