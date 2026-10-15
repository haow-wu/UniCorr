"""Wild-type complex parser. Reads coordinates only; no labels."""

from __future__ import annotations

import os

import numpy as np

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "MSE": "M",
}
WATER = {"HOH", "WAT", "H2O", "DOD"}
METAL = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO",
    "NI", "CU", "ZN", "CD", "HG", "AL", "CR", "PT", "PD", "AG", "AU",
}


def pad_pdb_line(line):
    line = line.rstrip("\n")
    if len(line) < 78:
        line = line + " " * (78 - len(line))
    return line


def element_of(atom_name, element_field):
    element = element_field.strip().upper()
    if element:
        return element
    name = atom_name.strip().upper()
    if name.startswith(("CL", "BR")):
        return name[:2]
    letters = "".join(ch for ch in name if ch.isalpha())
    return letters[:1] if letters else "X"


def parse_complex(path):
    residues = {}
    het_groups = {}
    with open(path, "r", errors="ignore") as handle:
        for raw in handle:
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            line = pad_pdb_line(raw)
            if line[16] not in (" ", "A"):
                continue
            resname = line[17:20].strip().upper()
            atom = line[12:16].strip().upper()
            element = element_of(atom, line[76:78])
            if element in ("H", "D") or atom.startswith("H"):
                continue
            try:
                resseq = int(line[22:26])
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                )
            except ValueError:
                continue
            chain = line[21].strip()
            icode = line[26].strip()
            if raw.startswith("ATOM"):
                if resname not in AA3:
                    continue
                key = (chain, resseq, icode)
                rec = residues.setdefault(key, {
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "aa": AA3[resname],
                    "atoms": {},
                })
                rec["atoms"][atom] = xyz
            else:
                if resname in WATER or resname in METAL or element in METAL:
                    continue
                het_groups.setdefault((resname, chain, resseq, icode), []).append((element, xyz))
    protein = [rec for rec in residues.values() if "CA" in rec["atoms"] and "N" in rec["atoms"] and "C" in rec["atoms"]]
    protein.sort(key=lambda rec: (rec["chain"], rec["resseq"], rec["icode"]))
    if not het_groups:
        raise ValueError("no ligand heavy atoms")
    ligand_key = max(het_groups, key=lambda key: len(het_groups[key]))
    ligand = het_groups[ligand_key]
    return protein, ligand


def find_wt_complex(sample_dir):
    if not os.path.isdir(sample_dir):
        return None
    for name in sorted(os.listdir(sample_dir)):
        lower = name.lower()
        if name.startswith("WT_") and lower.endswith(".pdb") and "complex" in lower:
            return os.path.join(sample_dir, name)
    return None


def locate_residue(protein, wt_aa, resnum, ligand_xyz):
    numbered = [i for i, rec in enumerate(protein) if rec["resseq"] == resnum]
    matched = [i for i in numbered if protein[i]["aa"] == wt_aa]
    pool = matched or numbered
    if not pool:
        return None
    if len(pool) == 1 or len(ligand_xyz) == 0:
        return pool[0]
    center = ligand_xyz.mean(axis=0)
    return min(pool, key=lambda i: float(np.linalg.norm(protein[i]["atoms"]["CA"] - center)))
