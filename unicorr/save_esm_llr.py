"""Save ESM-2 150M masked log-odds for every Kd/Ki sample.

Score = logit(wild-type amino acid) - logit(mutant amino acid) at the
masked site of the wild-type sequence. No label is used.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmForMaskedLM, EsmTokenizer

OUT = Path("/root/autodl-tmp/project4/unseen_kd")
SEQ = Path("/root/autodl-tmp/project/feature/sequence")
SNAPSHOT = (
    "/root/autodl-tmp/cache/huggingface/hub/models--facebook--esm2_t30_150M_UR50D/"
    "snapshots/a695f6045e2e32885fa60af20c13cb35398ce30c"
)
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_seq(path):
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore")
    return "".join(ch for ch in text if ch.isalpha()).upper()


def parse_mutation(text):
    raw = str(text).strip().upper().replace(" ", "")
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", raw)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    match = re.fullmatch(r"([A-Z]{3})(\d+)([A-Z]{3})", raw)
    if match and match.group(1) in AA3 and match.group(3) in AA3:
        return AA3[match.group(1)], int(match.group(2)), AA3[match.group(3)]
    return None


def site_index(seq, mt_seq, wt, resnum, mut):
    if 1 <= resnum <= len(seq) and seq[resnum - 1] == wt:
        return resnum - 1
    if mt_seq and len(mt_seq) == len(seq):
        hits = [
            i for i, (a, b) in enumerate(zip(seq, mt_seq))
            if a == wt and b == mut and a != b
        ]
        if len(hits) == 1:
            return hits[0]
    hits = [i for i, a in enumerate(seq) if a == wt]
    if len(hits) == 1:
        return hits[0]
    return None


def window(seq, index, limit=1022):
    if len(seq) <= limit:
        return seq, index
    start = max(0, min(index - limit // 2, len(seq) - limit))
    return seq[start:start + limit], index - start


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    tokenizer = EsmTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(SNAPSHOT, local_files_only=True).to(device)
    model.eval()
    token_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    cohort = pd.read_csv(OUT / "cohort_rows.csv")
    cohort["sample_id"] = cohort["SAMPLE_ID"].astype(str)
    rows = []
    for count, rec in enumerate(cohort.drop_duplicates("sample_id").itertuples(index=False), start=1):
        parsed = parse_mutation(rec.mutation)
        seq = read_seq(SEQ / rec.sample_id / "wt_seq.txt")
        mt_seq = read_seq(SEQ / rec.sample_id / "mt_seq.txt")
        score = np.nan
        if parsed is not None and seq:
            wt, resnum, mut = parsed
            index = site_index(seq, mt_seq, wt, resnum, mut)
            if index is not None and wt in token_id and mut in token_id:
                piece, local = window(seq, index)
                tokens = tokenizer(piece, return_tensors="pt")
                pos = local + 1
                tokens["input_ids"][0, pos] = tokenizer.mask_token_id
                tokens = {k: v.to(device) for k, v in tokens.items()}
                with torch.no_grad():
                    logits = model(**tokens).logits[0, pos]
                score = float(logits[token_id[wt]] - logits[token_id[mut]])
        rows.append({"sample_id": rec.sample_id, "esm_llr": score})
        if count % 50 == 0:
            print("scored", count, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "esm_llr.csv", index=False)
    print("wrote", len(frame), "finite", int(np.isfinite(frame["esm_llr"]).sum()), flush=True)


if __name__ == "__main__":
    main()
