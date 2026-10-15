"""Freeze the validation-selected model into the official prediction files.

The model is the k=4 UniProt shrinkage nudge plus a 0.25 within-protein
z-score of llr_x_polar. It is written only after the saved rows match the
frozen split and the baseline labels.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/root/autodl-tmp/project4/unseen_kd")


def main():
    raw = pd.read_csv(OUT / "predictions_specsel.csv")
    splits = pd.read_csv(OUT / "splits.csv")
    model = raw.loc[raw["model"] == "specsel"].copy()
    base = raw.loc[raw["model"] == "baseline"].copy()
    if len(model) != 571 or len(base) != 571:
        raise SystemExit(f"row counts model {len(model)} baseline {len(base)}")
    if not np.isfinite(pd.to_numeric(model["pred"], errors="coerce")).all():
        raise SystemExit("non-finite prediction")
    if not np.isfinite(pd.to_numeric(model["y"], errors="coerce")).all():
        raise SystemExit("non-finite label")
    key = ["sample_id", "fold"]
    merged = model.merge(base, on=key, suffixes=("_m", "_b"))
    if len(merged) != 571:
        raise SystemExit(f"baseline alignment {len(merged)}")
    y_gap = np.max(np.abs(merged["y_m"].to_numpy(float) - merged["y_b"].to_numpy(float)))
    if y_gap > 1e-8:
        raise SystemExit(f"label mismatch {y_gap}")
    if not (merged["uniprot_m"].astype(str) == merged["uniprot_b"].astype(str)).all():
        raise SystemExit("uniprot mismatch")
    test = splits.loc[splits["split"] == "test", ["sample_id", "uniprot", "fold"]].copy()
    test["sample_id"] = test["sample_id"].astype(str)
    model["sample_id"] = model["sample_id"].astype(str)
    joined = model.merge(test, on=["sample_id", "fold"], suffixes=("", "_split"))
    if len(joined) != 571:
        raise SystemExit(f"split alignment {len(joined)}")
    if not (joined["uniprot"].astype(str) == joined["uniprot_split"].astype(str)).all():
        raise SystemExit("split uniprot mismatch")
    for fold in range(5):
        part = splits.loc[splits["fold"] == fold]
        tr = set(part.loc[part["split"] == "train", "uniprot"].astype(str))
        va = set(part.loc[part["split"] == "val", "uniprot"].astype(str))
        te = set(part.loc[part["split"] == "test", "uniprot"].astype(str))
        if tr & te or va & te or tr & va:
            raise SystemExit(f"overlap in fold {fold}")
    labels = sorted(model["label_type"].astype(str).unique())
    print("labels", labels, flush=True)
    print("folds", model.groupby("fold").size().to_dict(), flush=True)
    backup = OUT / "predictions_pre_freeze.csv"
    official = OUT / "predictions.csv"
    if not backup.exists():
        shutil.copyfile(official, backup)
        print("backed up", backup, flush=True)
    frozen = raw.copy()
    frozen.loc[frozen["model"] == "specsel", "model"] = "final"
    frozen.to_csv(official, index=False)
    print("wrote", official, len(frozen), flush=True)
    subprocess.check_call([
        sys.executable, str(OUT / "score_metrics.py"),
        str(official), str(OUT / "metrics.json"),
    ])


if __name__ == "__main__":
    main()
