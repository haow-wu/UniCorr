"""UniProt-disjoint folds. Written before any test correlation is computed.

Five folds, seed 42. Proteins are assigned to test folds by count, then by
row mass. Validation is 15 percent of the remaining proteins, drawn with
seed 42+1000+fold. A protein never crosses train, val, and test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_FOLDS = 5
VAL_FRAC = 0.15


def assign_test_folds(proteins, weights, k, rng):
    base, rem = divmod(len(proteins), k)
    caps = [base + (1 if i < rem else 0) for i in range(k)]
    items = sorted(proteins, key=lambda name: (-weights[name], name))
    folds = [[] for _ in range(k)]
    rows = [0] * k
    counts = [0] * k
    for name in items:
        candidates = [i for i in range(k) if counts[i] < caps[i]]
        best_rows = min(rows[i] for i in candidates)
        best = [i for i in candidates if rows[i] == best_rows]
        choice = int(rng.choice(best))
        folds[choice].append(name)
        rows[choice] += int(weights[name])
        counts[choice] += 1
    return folds


def main():
    src = Path("cohort_rows.csv")
    frame = pd.read_csv(src)
    frame["uniprot"] = frame["uniprot"].astype(str).str.upper()
    frame["sample_id"] = frame["SAMPLE_ID"].astype(str)
    weights = frame.groupby("uniprot").size().to_dict()
    proteins = list(weights)
    if len(proteins) < N_FOLDS:
        raise SystemExit(f"only {len(proteins)} proteins; need at least {N_FOLDS}")
    rng = np.random.default_rng(SEED)
    test_folds = assign_test_folds(proteins, weights, N_FOLDS, rng)
    rows = []
    for fold in range(N_FOLDS):
        test_ids = set(test_folds[fold])
        remain = [name for j, group in enumerate(test_folds) if j != fold for name in group]
        val_n = max(1, int(round(len(remain) * VAL_FRAC)))
        rng_i = np.random.default_rng(SEED + 1000 + fold)
        order = remain.copy()
        rng_i.shuffle(order)
        val_ids = set(order[:val_n])
        train_ids = set(order[val_n:])
        overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        if overlap:
            raise SystemExit(f"fold {fold} overlap {sorted(overlap)[:5]}")
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            part = frame.loc[frame["uniprot"].isin(ids)]
            for rec in part.itertuples(index=False):
                rows.append({
                    "sample_id": rec.sample_id,
                    "uniprot": rec.uniprot,
                    "split": name,
                    "fold": fold,
                })
    out = pd.DataFrame(rows)
    # One row per sample per fold role. A sample is test in exactly one fold.
    out.to_csv("splits.csv", index=False)
    test = out.loc[out["split"] == "test"]
    train = out.loc[out["split"] == "train"]
    lines = []
    for fold in range(N_FOLDS):
        te = set(test.loc[test["fold"] == fold, "uniprot"])
        tr = set(train.loc[train["fold"] == fold, "uniprot"])
        va = set(out.loc[(out["fold"] == fold) & (out["split"] == "val"), "uniprot"])
        lines.append(
            f"fold {fold} train {len(tr)} val {len(va)} test {len(te)} "
            f"train&test {len(tr & te)} val&test {len(va & te)} train&val {len(tr & va)}"
        )
    Path("overlap_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("splits", len(out), "test_rows", len(test))
    meta = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "val_frac_of_remaining_proteins": VAL_FRAC,
        "n_proteins": len(proteins),
        "n_samples": int(frame["sample_id"].nunique()),
    }
    Path("split_config.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
