"""Distance-gated substitution model on top of the within-complex fit.

Validation chooses the configuration. The previous winner
(between + packing features, ridge alpha 300) is included as a control.
Test predictions are written only for the chosen configuration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("/root/autodl-tmp/project4/unseen_kd")
CACHE = Path("/root/autodl-tmp/project4/cache")
DATA = Path("/root/autodl-tmp/project/data")
MIN_N = 30
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
VOL = {
    "A": 67, "R": 148, "N": 96, "D": 91, "C": 86, "Q": 114, "E": 109,
    "G": 48, "H": 118, "I": 124, "L": 124, "K": 135, "M": 124, "F": 135,
    "P": 90, "S": 73, "T": 93, "W": 163, "Y": 141, "V": 105,
}
CHG = {aa: 0.0 for aa in AA}
CHG.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5})
PACK = [
    "heavy_dist", "prox", "ca_dist", "rsa", "bind", "hb_flag",
    "x_clash", "x_rep", "x_contact", "x_hbond", "x_min_dist", "x_att",
    "d_hbond", "d_clash", "d_rep", "d_contact",
]
CHEM = [
    "heavy_dist", "prox", "d_hydro", "d_charge", "d_vol",
    "prox_hydro", "prox_charge", "prox_vol",
]
LIGAND = ["lig_mw", "lig_logp", "lig_donors", "lig_acceptors", "lig_tpsa", "lig_rot"]
BETWEEN2 = ["mu_heavy", "mu_prox", "mu_rsa", "mu_bind"]


def pearson(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < MIN_N or np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return None
    return float(np.corrcoef(y, p)[0, 1])


def parse_mutation(text):
    import re
    raw = str(text).strip().upper().replace(" ", "")
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", raw)
    if match:
        return match.group(1), match.group(3)
    match = re.fullmatch(r"([A-Z]{3})(\d+)([A-Z]{3})", raw)
    if match and match.group(1) in AA3 and match.group(3) in AA3:
        return AA3[match.group(1)], AA3[match.group(3)]
    return None, None


def concat_cache(name, extra):
    base = pd.read_csv(CACHE / name)
    extra_df = pd.read_csv(OUT / extra)
    frame = pd.concat([base, extra_df], ignore_index=True)
    frame["sample_id"] = frame["graph_id"].astype(str)
    return frame.drop_duplicates("sample_id", keep="last")


def load():
    splits = pd.read_csv(OUT / "splits.csv")
    cohort = pd.read_csv(OUT / "cohort_rows.csv")
    cohort["sample_id"] = cohort["SAMPLE_ID"].astype(str)
    splits["sample_id"] = splits["sample_id"].astype(str)
    keep = [c for c in cohort.columns if c not in {"uniprot", "split", "fold"}]
    frame = splits.merge(cohort[keep], on="sample_id", how="left")
    rot = concat_cache("rotamer_features.csv", "rotamer_extra.csv")
    physics = [
        "heavy_dist", "ca_dist", "d_hbond", "d_clash", "d_rep", "d_contact",
        "x_clash", "x_rep", "x_contact", "x_hbond", "x_min_dist", "x_att",
    ]
    frame = frame.merge(rot[["sample_id"] + physics], on="sample_id", how="left")
    anm = pd.concat([
        pd.read_csv(DATA / "processed" / "anm_distance.csv"),
        pd.read_csv(OUT / "anm_extra.csv"),
    ], ignore_index=True)
    anm["sample_id"] = anm["graph_id"].astype(str)
    anm = anm.drop_duplicates("sample_id", keep="last")
    frame = frame.merge(anm[["sample_id", "mut_dist"]], on="sample_id", how="left")
    plat = pd.read_csv(OUT / "platinum_flat_file.csv", low_memory=False)
    plat["uniprot"] = plat["mut.uniprot"].astype(str).str.upper()
    plat["mutation_key"] = plat["mutation"].astype(str).str.upper().str.replace(" ", "", regex=False)
    plat["pdb_key"] = plat["affin.pdb_id"].astype(str).str.upper().str[:4]
    keep_plat = [
        "uniprot", "mutation_key", "pdb_key", "mut.distance_to_lig",
        "mut.residue_rsa", "mut.in_binding_site", "mut.residue_hbond",
        "lig.mol_weight", "lig.logp", "lig.num_donors", "lig.num_acceptors",
        "lig.tpsa", "lig.num_rotatable_bonds",
    ]
    plat = plat[keep_plat].drop_duplicates(["uniprot", "mutation_key", "pdb_key"])
    frame["mutation_key"] = frame["mutation"].astype(str).str.upper().str.replace(" ", "", regex=False)
    frame["pdb_key"] = frame["pdb"].astype(str).str.upper().str[:4]
    frame = frame.merge(plat, on=["uniprot", "mutation_key", "pdb_key"], how="left")
    frame["distance"] = pd.to_numeric(frame["mut_dist"], errors="coerce")
    plat_dist = pd.to_numeric(frame["mut.distance_to_lig"], errors="coerce")
    missing = ~np.isfinite(frame["distance"].to_numpy(dtype=float))
    frame.loc[missing, "distance"] = plat_dist[missing]
    rsa = frame["mut.residue_rsa"].astype(str).str.replace("%", "", regex=False)
    frame["rsa"] = pd.to_numeric(rsa, errors="coerce")
    frame["bind"] = frame["mut.in_binding_site"].astype(str).str.upper().eq("YES").astype(float)
    hb = frame["mut.residue_hbond"].astype(str).str.lower()
    frame["hb_flag"] = np.where(frame["mut.residue_hbond"].isna(), np.nan, (~hb.str.contains("no")).astype(float))
    for src, dest in (
        ("lig.mol_weight", "lig_mw"),
        ("lig.logp", "lig_logp"),
        ("lig.num_donors", "lig_donors"),
        ("lig.num_acceptors", "lig_acceptors"),
        ("lig.tpsa", "lig_tpsa"),
        ("lig.num_rotatable_bonds", "lig_rot"),
    ):
        frame[dest] = pd.to_numeric(frame[src], errors="coerce")
    heavy = pd.to_numeric(frame["heavy_dist"], errors="coerce")
    frame["prox"] = np.exp(-heavy / 5.0)
    parsed = frame["mutation"].map(parse_mutation)
    frame["wt_aa"] = [item[0] for item in parsed]
    frame["mut_aa"] = [item[1] for item in parsed]
    frame["d_hydro"] = [KD.get(m, np.nan) - KD.get(w, np.nan) if w and m else np.nan for w, m in zip(frame["wt_aa"], frame["mut_aa"])]
    frame["d_charge"] = [CHG.get(m, np.nan) - CHG.get(w, np.nan) if w and m else np.nan for w, m in zip(frame["wt_aa"], frame["mut_aa"])]
    frame["d_vol"] = [VOL.get(m, np.nan) - VOL.get(w, np.nan) if w and m else np.nan for w, m in zip(frame["wt_aa"], frame["mut_aa"])]
    frame["prox_hydro"] = frame["prox"] * frame["d_hydro"]
    frame["prox_charge"] = frame["prox"] * frame["d_charge"]
    frame["prox_vol"] = frame["prox"] * frame["d_vol"]
    for aa in AA:
        frame[f"g_{aa}"] = frame["prox"] * frame["mut_aa"].eq(aa).astype(float)
    frame["complex"] = frame["uniprot"].astype(str) + "|" + frame["pdb_key"].astype(str)
    pack = np.load(OUT / "esm_delta_kd.npz", allow_pickle=True)
    vectors = {str(i): np.asarray(row, float) for i, row in zip(pack["graph_id"], pack["delta"])}
    print("parsed_aa", int(frame["wt_aa"].notna().sum()), "rows", len(frame), "esm", len(vectors), flush=True)
    return frame.drop_duplicates(["sample_id", "fold", "split"]), vectors


def matrix(df, cols):
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def impute_fit(train_x):
    fill = np.nanmedian(train_x, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return fill


def impute_apply(values, fill):
    out = np.array(values, dtype=float, copy=True)
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(fill, inds[1])
    return out


def center_by_group(values, groups):
    out = np.array(values, dtype=float, copy=True)
    for gid in pd.unique(groups):
        mask = groups == gid
        out[mask] = out[mask] - out[mask].mean(axis=0)
    return out


def esm_matrix(df, vectors):
    rows = []
    for gid in df["sample_id"].astype(str):
        vec = vectors.get(gid)
        rows.append(np.full(640, np.nan) if vec is None else vec)
    return np.vstack(rows)


def with_pcs(train, other, vectors, cols):
    """Append train-fit PCA scores. Returns column names present on both copies."""
    if not any(str(c).startswith("pc") for c in cols):
        return train, other, cols
    k = max(int(str(c)[2:]) for c in cols if str(c).startswith("pc")) + 1
    base_cols = [c for c in cols if not str(c).startswith("pc")]
    imp = SimpleImputer(strategy="median")
    pca = PCA(n_components=k, random_state=42)
    z_tr = pca.fit_transform(imp.fit_transform(esm_matrix(train, vectors)))
    z_ot = pca.transform(imp.transform(esm_matrix(other, vectors)))
    train = train.copy()
    other = other.copy()
    pc_names = [f"pc{i}" for i in range(k)]
    for i, name in enumerate(pc_names):
        train[name] = z_tr[:, i]
        other[name] = z_ot[:, i]
    return train, other, base_cols + pc_names


def make_ridge(alpha):
    model = Pipeline([
        ("sc", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
    return model


def fit_predict_fold(train, other, cols, alpha, mode, vectors):
    train, other, cols = with_pcs(train, other, vectors, cols)
    groups_tr = train["complex"].astype(str).to_numpy()
    groups_ot = other["complex"].astype(str).to_numpy()
    x_tr = impute_apply(matrix(train, cols), impute_fit(matrix(train, cols)))
    x_ot = impute_apply(matrix(other, cols), impute_fit(matrix(train, cols)))
    y_tr = train["y"].to_numpy(dtype=float)
    global_mean = float(np.mean(y_tr))
    x_tr_c = center_by_group(x_tr, groups_tr)
    y_tr_c = center_by_group(y_tr.reshape(-1, 1), groups_tr).ravel()
    model = make_ridge(alpha)
    model.fit(x_tr_c, y_tr_c)
    delta = model.predict(center_by_group(x_ot, groups_ot))
    if mode == "between":
        lig_fill = impute_fit(matrix(train, LIGAND))
        lig_tr = impute_apply(matrix(train, LIGAND), lig_fill)
        lig_ot = impute_apply(matrix(other, LIGAND), lig_fill)
        between_tr = np.hstack([lig_tr, x_tr[:, :1]])
        between_ot = np.hstack([lig_ot, x_ot[:, :1]])
    elif mode == "between2":
        geo = ["heavy_dist", "prox", "rsa", "bind"]
        fill = impute_fit(matrix(train, geo))
        g_tr = impute_apply(matrix(train, geo), fill)
        g_ot = impute_apply(matrix(other, geo), fill)
        mu_tr = pd.DataFrame(g_tr, columns=geo).groupby(groups_tr).transform("mean").to_numpy()
        mu_ot = pd.DataFrame(g_ot, columns=geo).groupby(groups_ot).transform("mean").to_numpy()
        lig_fill = impute_fit(matrix(train, LIGAND))
        between_tr = np.hstack([impute_apply(matrix(train, LIGAND), lig_fill), mu_tr])
        between_ot = np.hstack([impute_apply(matrix(other, LIGAND), lig_fill), mu_ot])
    else:
        raise KeyError(mode)
    b_model = make_ridge(alpha)
    b_model.fit(between_tr, y_tr)
    offset = b_model.predict(between_ot)
    offset = pd.Series(offset).groupby(groups_ot).transform("mean").to_numpy()
    return offset + delta, global_mean


def configs():
    gate = PACK + [f"g_{aa}" for aa in AA]
    esm = PACK + [f"pc{i}" for i in range(32)]
    items = [
        ("control_pack_a300", "between", PACK, 300.0),
        ("b2_pack_a300", "between2", PACK, 300.0),
        ("b2_pack_a3000", "between2", PACK, 3000.0),
        ("chem_a30", "between", CHEM, 30.0),
        ("chem_a300", "between", CHEM, 300.0),
        ("gate_a300", "between", gate, 300.0),
        ("gate_a3000", "between", gate, 3000.0),
        ("gate_a30000", "between", gate, 30000.0),
        ("b2_gate_a3000", "between2", gate, 3000.0),
        ("esm_a3000", "between", esm, 3000.0),
        ("esm_a1e5", "between", esm, 100000.0),
        ("b2_esm_a1e5", "between2", esm, 100000.0),
    ]
    return items


def main():
    frame, vectors = load()
    table = []
    for key, mode, cols, alpha in configs():
        scores = []
        for fold in range(5):
            part = frame.loc[frame["fold"] == fold]
            train = part.loc[part["split"] == "train"]
            val = part.loc[part["split"] == "val"]
            pred, _ = fit_predict_fold(train, val, cols, alpha, mode, vectors)
            score = pearson(val["y"], pred)
            scores.append(np.nan if score is None else score)
        mean = float(np.nanmean(scores))
        table.append((key, mode, cols, alpha, mean, scores))
        print(key, "val_mean", round(mean, 4), "folds", [None if not np.isfinite(s) else round(float(s), 4) for s in scores], flush=True)
    table.sort(key=lambda item: -item[4])
    chosen, mode, cols, alpha, mean, _ = table[0]
    print("CHOSEN", chosen, "val_mean", round(mean, 4), flush=True)
    if chosen == "control_pack_a300":
        print("control remains best on validation; test file is predictions_within.csv", flush=True)
        pd.DataFrame([
            {"config": key, "val_mean": val, "chosen": int(key == chosen)}
            for key, _, _, _, val, _ in table
        ]).to_csv(OUT / "val_selection_gate.csv", index=False)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        pred, _ = fit_predict_fold(train, test, cols, alpha, mode, vectors)
        if not np.isfinite(pred).all():
            raise SystemExit(f"non-finite test prediction in fold {fold}")
        for i, rec in enumerate(test.itertuples(index=False)):
            rows.append({
                "sample_id": rec.sample_id,
                "uniprot": rec.uniprot,
                "y": rec.y,
                "pred": float(pred[i]),
                "label_type": rec.label_type,
                "distance": rec.distance,
                "fold": int(fold),
                "model": "gate",
            })
    base = pd.read_csv(OUT / "predictions.csv")
    base = base.loc[base["model"] == "baseline"].copy()
    preds = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    preds.to_csv(OUT / "predictions_gate.csv", index=False)
    pd.DataFrame([
        {"config": key, "val_mean": val, "chosen": int(key == chosen)}
        for key, _, _, _, val, _ in table
    ]).to_csv(OUT / "val_selection_gate.csv", index=False)
    print("wrote", OUT / "predictions_gate.csv", "test_rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
