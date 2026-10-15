"""Freeze the Kd/Ki single-missense cohort before any model is trained.

MdrDB has no assay-type column (Yang et al., Commun. Chem. 2023). Rows are
kept only when they join the Platinum flat file (Pires, Blundell, Ascher,
Nucleic Acids Res. 2015) on UniProt, wild-type PDB, and mutation string, and
Platinum affin.unit is KD or Ki. Km is excluded. GDSC, DepMap, KinaseMD, RET,
TKI, and AIMMS are not in this cohort: those sources are cell-line IC50,
ΔpIC50, or an assay type that this table does not identify as Kd or Ki.

DeltaDeltaG uses MdrDB DDG.EXP when the row joins. The sign check against
RT ln(K_mut/K_wt) is written to cohort_sign_check.txt and is not a model score.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

MDR = os.environ.get(
    "MDR_TSV",
    "/root/autodl-tmp/project/data/MdrDB_CoreSet_release_v1.0.2022.tsv",
)
PLATINUM = os.environ.get(
    "PLATINUM_CSV",
    "/root/autodl-tmp/project4/unseen_kd/platinum_flat_file.csv",
)
OUT = Path(os.environ.get("UNSEEN_OUT", "/root/autodl-tmp/project4/unseen_kd"))
RT = 0.00198720425864083 * 298.15  # kcal/mol at 25 C


def norm_mut(text):
    return str(text).strip().upper().replace(" ", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plat = pd.read_csv(PLATINUM, low_memory=False)
    plat["unit"] = plat["affin.unit"].astype(str).str.strip()
    plat = plat.loc[plat["unit"].isin(["KD", "Ki"])].copy()
    plat = plat.loc[plat["mut.is_single_point"].astype(str).str.upper().eq("YES")].copy()
    plat["uniprot"] = plat["mut.uniprot"].astype(str).str.strip().str.upper()
    plat["pdb"] = plat["affin.pdb_id"].astype(str).str.strip().str.upper()
    plat["mutation"] = plat["mutation"].map(norm_mut)
    plat["k_wt"] = pd.to_numeric(plat["affin.k_wt"], errors="coerce")
    plat["k_mt"] = pd.to_numeric(plat["affin.k_mt"], errors="coerce")
    usable = (plat["k_wt"] > 0) & (plat["k_mt"] > 0)
    plat.loc[usable, "ddg_from_k"] = RT * np.log(plat.loc[usable, "k_mt"] / plat.loc[usable, "k_wt"])
    print("platinum_kd_ki_single", len(plat), flush=True)

    use = [
        "SAMPLE_ID", "DATASET", "TYPE", "UNIPROT_ID", "MUTATION",
        "DDG.EXP", "PDB_ID", "SMILES",
    ]
    # PDB column name varies; detect it.
    header = pd.read_csv(MDR, sep="\t", nrows=0).columns.tolist()
    pdb_col = next((c for c in header if c.upper() in {"PDB_ID", "PDB", "WT_PDB", "PDBID"}), None)
    if pdb_col is None:
        pdb_like = [c for c in header if "PDB" in c.upper()]
        print("PDB_COLUMNS", pdb_like, flush=True)
        pdb_col = pdb_like[0] if pdb_like else None
    cols = [c for c in use if c in header and c != "PDB_ID"]
    if pdb_col and pdb_col not in cols:
        cols.append(pdb_col)
    mdr = pd.read_csv(MDR, sep="\t", usecols=cols, low_memory=False)
    mdr = mdr.loc[mdr["DATASET"].astype(str).str.upper().eq("PLATINUM")].copy()
    mdr = mdr.loc[mdr["TYPE"].astype(str).str.contains("Single", na=False)].copy()
    mdr["uniprot"] = mdr["UNIPROT_ID"].astype(str).str.strip().str.upper()
    mdr["mutation"] = mdr["MUTATION"].map(norm_mut)
    if pdb_col:
        mdr["pdb"] = mdr[pdb_col].astype(str).str.strip().str.upper().str[:4]
    else:
        mdr["pdb"] = ""
    mdr["ddg_exp"] = pd.to_numeric(mdr["DDG.EXP"], errors="coerce")
    print("mdr_platinum_single", len(mdr), "pdb_col", pdb_col, flush=True)

    keys = ["uniprot", "mutation", "pdb"]
    joined = mdr.merge(
        plat[keys + ["unit", "ddg_from_k", "k_wt", "k_mt"]],
        on=keys,
        how="inner",
    )
    print("joined_on_uniprot_mutation_pdb", len(joined), flush=True)
    if joined.empty:
        joined = mdr.merge(
            plat[["uniprot", "mutation", "unit", "ddg_from_k", "k_wt", "k_mt", "pdb"]],
            on=["uniprot", "mutation"],
            how="inner",
            suffixes=("", "_plat"),
        )
        print("joined_on_uniprot_mutation_only", len(joined), flush=True)
    both = joined["ddg_exp"].notna() & joined["ddg_from_k"].notna()
    if both.any():
        corr = np.corrcoef(joined.loc[both, "ddg_exp"], joined.loc[both, "ddg_from_k"])[0, 1]
        med = float((joined.loc[both, "ddg_exp"] - joined.loc[both, "ddg_from_k"]).median())
    else:
        corr, med = float("nan"), float("nan")
    report = OUT / "cohort_sign_check.txt"
    report.write_text(
        "source_mdr=Yang et al. Commun Chem 2023 CoreSet v1.0.2022 DATASET=Platinum TYPE contains Single\n"
        "source_platinum=Pires Blundell Ascher Nucleic Acids Res 2015 flat file affin.unit in {KD, Ki}; Km excluded\n"
        f"pdb_column={pdb_col}\n"
        f"n_joined={len(joined)}\n"
        f"n_with_both_ddg={int(both.sum())}\n"
        f"pearson_mdr_vs_RT_ln={corr}\n"
        f"median_mdr_minus_RT_ln_kcal={med}\n"
        "RT=0.00198720425864083*298.15 kcal/mol. Positive ddg_from_k means K_mut>K_wt, weaker binding.\n"
    )
    print(report.read_text(), flush=True)
    keep = joined.drop_duplicates("SAMPLE_ID").copy()
    keep = keep.loc[keep["ddg_from_k"].notna()].copy()
    keep["y"] = keep["ddg_from_k"]
    keep["label_type"] = keep["unit"]
    keep.to_csv(OUT / "cohort_rows.csv", index=False)
    print("wrote", OUT / "cohort_rows.csv", "rows", len(keep), "y_from", "RT*ln(K_mt/K_wt)", flush=True)


if __name__ == "__main__":
    main()
