"""Pick the validation model by the official two-number score.

The score is the worse of the fold-mean Pearson and the Pearson after
removing the single most helpful protein. The already tested d_hbond
weight 0.25 is replaced only when another candidate scores higher.
"""

import subprocess
import sys

import numpy as np
import pandas as pd

import train_gate
import train_shrink
import train_znudge
import train_znudge6


def shrink_pred(frame, base, scales):
    return train_shrink.predict(frame, base, "shrink", 4.0, scales)


def add_z(frame, pred, column, weight):
    if weight == 0.0:
        return pred
    z = train_znudge6.within_z(frame, column, "uniprot")
    return pred + weight * z * (float(np.std(pred)) + 1e-8)


def main():
    frame, vectors = train_znudge.prepare()
    cache = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        val = part.loc[part["split"] == "val"]
        base = train_znudge.base_predict(train, val, vectors)
        scales = {column: train_shrink.pooled_scale(train, column) for column, _w in train_shrink.REF}
        cache.append((val, shrink_pred(val, base, scales)))
        print("packed", fold, flush=True)
    specs = [
        ("dhbond_0.2", (("d_hbond", 0.2),)),
        ("dhbond_0.25", (("d_hbond", 0.25),)),
        ("hb_geom_0.25", (("hb_geom", 0.25),)),
        ("llr_0.25", (("llr_x_polar", 0.25),)),
        ("esm_0.5", (("esm_llr", 0.5),)),
        ("dhbond_geom", (("d_hbond", 0.25), ("hb_geom", 0.1))),
    ]
    scored = {}
    for name, terms in specs:
        parts = []
        for val, pred in cache:
            cur = pred
            for column, weight in terms:
                cur = add_z(val, cur, column, weight)
            parts.append((val, cur))
        full = train_shrink.evaluate(parts, (0, 1, 2, 3, 4))
        scored[name] = (terms, full)
        print(
            name, "stable", round(full[0], 4), "mean", round(full[1], 4),
            "without", round(full[2], 4), "worst", round(full[3], 4), flush=True,
        )
    ref = scored["dhbond_0.25"][1][0]
    ranked = sorted(scored.items(), key=lambda item: (-item[1][1][0], -item[1][1][1]))
    name, (terms, full) = ranked[0]
    print("CHOSEN", name, "stable", round(full[0], 4), flush=True)
    if name == "dhbond_0.25" or full[0] < ref:
        print("tested d_hbond weight remains best on the official validation score", flush=True)
        return
    rows = []
    for fold in range(5):
        part = frame.loc[frame["fold"] == fold]
        train = part.loc[part["split"] == "train"]
        test = part.loc[part["split"] == "test"]
        base = train_znudge.base_predict(train, test, vectors)
        scales = {column: train_shrink.pooled_scale(train, column) for column, _w in train_shrink.REF}
        pred = shrink_pred(test, base, scales)
        for column, weight in terms:
            pred = add_z(test, pred, column, weight)
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
                "model": "specsel",
            })
    saved = pd.read_csv(train_gate.OUT / "predictions.csv")
    saved = saved.loc[saved["model"] == "baseline"].copy()
    preds = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
    dest = train_gate.OUT / "predictions_specsel.csv"
    preds.to_csv(dest, index=False)
    print("wrote", dest, len(rows), flush=True)
    subprocess.check_call([
        sys.executable, str(train_gate.OUT / "score_metrics.py"),
        str(dest), str(train_gate.OUT / "metrics_specsel.json"),
    ])


if __name__ == "__main__":
    main()
