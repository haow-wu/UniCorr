"""Boltzmann rotamer free energies on the wild-type backbone.

Mutant crystal coordinates are not read. Temperatures are a fixed list.
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

from featurize import graph_ids, load_meta
from rotlib import ensemble_feature_names, ensemble_state
from structure_io import find_wt_complex, locate_residue, parse_complex

DATA = os.environ.get("MDR_DATA", "/root/autodl-tmp/project/data")
OUT_DIR = os.environ.get("PROJECT4", "/root/autodl-tmp/project4")
FEATURES = ensemble_feature_names()


def _env_atoms(protein, index, cb):
    from rotlib import ATOM_CHARGE

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
        return np.zeros((0, 3)), np.array([], dtype=object), np.zeros((0,))
    return np.vstack(xyz), np.array(elem, dtype=object), np.asarray(charge, dtype=np.float64)


def featurize_one(graph_id):
    row = {"graph_id": graph_id, "ok": 0, "reason": ""}
    for name in FEATURES:
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
    meta = META.get(graph_id)
    if meta is None:
        row["reason"] = "no_meta"
        return row
    lig_elem = np.array([item[0] for item in ligand], dtype=object)
    lig_xyz = np.vstack([item[1] for item in ligand])
    index = locate_residue(protein, meta["wt_aa"], meta["resnum"], lig_xyz)
    if index is None:
        row["reason"] = "residue_not_found"
        return row
    atoms = protein[index]["atoms"]
    n_xyz, ca, c_xyz = atoms["N"], atoms["CA"], atoms["C"]
    cb = atoms.get("CB", ca)
    prot_xyz, prot_elem, prot_q = _env_atoms(protein, index, cb)
    wild = ensemble_state(
        meta["wt_aa"], n_xyz, ca, c_xyz, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q
    )
    mutant = ensemble_state(
        meta["mut_aa"], n_xyz, ca, c_xyz, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q
    )
    for key, value in mutant.items():
        row[f"d_{key}"] = float(value) - float(wild[key])
    row["ok"] = 1
    row["reason"] = ""
    return row


def _init_worker(meta):
    global META
    META = meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cache = os.path.join(OUT_DIR, "cache")
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, "ensemble_features.csv")
    partial = os.path.join(cache, "ensemble_partial.csv")
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

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(meta,)
    ) as pool:
        for count, row in enumerate(pool.map(featurize_one, ids, chunksize=4), start=1):
            rows.append(row)
            seen_ok += int(row["ok"])
            if count % 200 == 0 or count == len(ids):
                print(f"done {count}/{len(ids)} ok {seen_ok}", flush=True)
                flush()
    if args.limit:
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"wrote {dest} rows {len(rows)}", flush=True)
        return
    flush()
    frame = pd.read_csv(partial)
    frame.to_csv(dest, index=False)
    print(f"wrote {dest} rows {len(frame)} ok {int(frame.ok.sum())}", flush=True)
    if (frame.ok == 0).any():
        reasons = frame.loc[frame.ok == 0, "reason"].value_counts().head(8)
        print(reasons.to_string(), flush=True)


if __name__ == "__main__":
    main()
