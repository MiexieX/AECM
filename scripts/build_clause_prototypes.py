#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import tokenizers_fn


JUNK_PREFIXES = (
    "in comparison",
    "compared to",
    "ap portable",
    "frontal and lateral",
    "pa and lateral",
    "portable chest",
    "single view",
    "views were obtained",
    "the patient is",
    "as compared to",
    "history",
    "indication",
    "comparison",
    "exam",
    "chest radiograph",
)

JUNK_SUBSTRINGS = (
    "portable upright ap view",
    "portable ap view",
    "frontal view of the chest",
    "lateral view of the chest",
    "two views of the chest",
    "comparison is made",
    "comparison made to",
)

NORMAL_PATTERNS = (
    "no pleural effusion",
    "no pneumothorax",
    "lungs are clear",
    "without focal consolidation",
    "heart size is normal",
    "cardiomediastinal silhouette is normal",
    "cardiomediastinal silhouette is within normal limits",
    "cardiac and mediastinal silhouettes are unremarkable",
    "mediastinal and hilar contours are normal",
    "pulmonary vasculature is normal",
    "no acute osseous abnormalities",
    "no acute cardiopulmonary abnormality",
    "no focal consolidation",
)

ABNORMAL_KEYWORDS = (
    "effusion",
    "pneumothorax",
    "edema",
    "opacity",
    "opacities",
    "cardiomegaly",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "fracture",
    "lesion",
    "enlarged",
    "congestion",
    "infiltrate",
    "infiltration",
    "nodule",
    "mass",
    "tube",
    "catheter",
    "line",
)


def load_json_args(path):
    json_str = ""
    with open(path, "r") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    defaults = json.loads(json_str, object_pairs_hook=OrderedDict)
    args = {}
    for key in defaults.keys():
        args.update(defaults[key])
    return args


def is_abnormal(label_item):
    excluded = {"no_finding", "support_devices"}
    polarity = label_item.get("pathology_polarity", {})
    for name, state in polarity.items():
        if name not in excluded and state in {"positive", "uncertain"}:
            return True
    pathologies = label_item.get("pathologies", {})
    for name, value in pathologies.items():
        if name not in excluded and int(value) == 1:
            return True
    return False


def split_into_clauses(report):
    # Keep this simple and deterministic. MIMIC reports are already fairly
    # structured; punctuation-based splitting works well enough for MVP.
    clauses = re.split(r"[.;]", report.lower())
    return [c.strip() for c in clauses if c and c.strip()]


def keep_clause(clause):
    words = clause.split()
    if len(words) < 5:
        return False
    if len(words) > 24:
        return False
    if any(clause.startswith(prefix) for prefix in JUNK_PREFIXES):
        return False
    if any(substr in clause for substr in JUNK_SUBSTRINGS):
        return False
    # Filter clauses dominated by digits / metadata-like text
    alpha_chars = sum(ch.isalpha() for ch in clause)
    if alpha_chars < max(8, len(clause) * 0.45):
        return False
    return True


def clause_type(clause):
    if any(pattern in clause for pattern in NORMAL_PATTERNS):
        return "normal"
    if any(keyword in clause for keyword in ABNORMAL_KEYWORDS):
        return "abnormal"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="config/mimic_cxr/vlci.json")
    parser.add_argument("--dataset_name", type=str, default="mimic_cxr")
    parser.add_argument("--tokenizer", type=str, default="mix")
    parser.add_argument("--pseudo_label_path", type=str, default="data/mimic_cxr_pseudo_labels.json")
    parser.add_argument("--output_path", type=str, default="data/mimic_cxr_clause_prototypes.pt")
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    args = parser.parse_args()

    config = load_json_args(args.c)
    tokenizer = tokenizers_fn[args.tokenizer]({"dataset_name": args.dataset_name, **config})

    with open(config["ann_path"], "r") as f:
        ann = json.load(f)
    with open(args.pseudo_label_path, "r") as f:
        pseudo = json.load(f)["labels"]

    clause_counter = Counter()
    abnormal_counter = Counter()
    normal_counter = Counter()
    other_counter = Counter()
    abnormal_count = 0

    clean_report = getattr(tokenizer, "clean_report", None)
    if clean_report is None and hasattr(tokenizer, "clean_report_mimic_cxr"):
        clean_report = tokenizer.clean_report_mimic_cxr
    if clean_report is None:
        raise RuntimeError("Tokenizer does not expose a report cleaning function.")

    for example in ann["train"]:
        image_id = str(example["id"])
        label_item = pseudo.get(image_id)
        if not label_item or not is_abnormal(label_item):
            continue

        abnormal_count += 1
        cleaned_report = clean_report(example["report"])
        clauses = split_into_clauses(cleaned_report)
        for clause in clauses:
            if keep_clause(clause):
                clause_counter[clause] += 1
                ctype = clause_type(clause)
                if ctype == "abnormal":
                    abnormal_counter[clause] += 1
                elif ctype == "normal":
                    normal_counter[clause] += 1
                else:
                    other_counter[clause] += 1

    selected_clauses = []
    selected_token_ids = []
    selected_counts = []
    seen = set()
    unk_id = tokenizer.get_id_by_token("<unk>") if hasattr(tokenizer, "get_id_by_token") else None

    def maybe_add_clause(clause, freq):
        if clause in seen:
            return False
        token_ids = [tokenizer.get_id_by_token(token) for token in clause.split()]
        if not token_ids:
            return False
        if unk_id is not None and unk_id in token_ids:
            return False
        seen.add(clause)
        selected_clauses.append(clause)
        selected_token_ids.append(token_ids)
        selected_counts.append(freq)
        return True

    abnormal_quota = max(1, int(args.top_k * 0.75))
    normal_quota = max(1, args.top_k - abnormal_quota)
    max_candidates = max(args.top_k * args.candidate_multiplier, args.top_k)

    for clause, freq in abnormal_counter.most_common(max_candidates):
        maybe_add_clause(clause, freq)
        if len(selected_clauses) >= abnormal_quota:
            break

    normal_added = 0
    for clause, freq in normal_counter.most_common(max_candidates):
        if maybe_add_clause(clause, freq):
            normal_added += 1
        if normal_added >= normal_quota:
            break

    for source in (other_counter, clause_counter):
        for clause, freq in source.most_common(max_candidates):
            maybe_add_clause(clause, freq)
            if len(selected_clauses) >= args.top_k:
                break
        if len(selected_clauses) >= args.top_k:
            break

    if len(selected_clauses) > args.top_k:
        selected_clauses = selected_clauses[:args.top_k]
        selected_token_ids = selected_token_ids[:args.top_k]
        selected_counts = selected_counts[:args.top_k]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_data = {
        "clauses": selected_clauses,
        "token_ids": selected_token_ids,
        "counts": selected_counts,
        "meta": {
            "top_k": len(selected_clauses),
            "requested_top_k": args.top_k,
            "dataset": args.dataset_name,
            "abnormal_reports_used": abnormal_count,
        },
    }
    torch.save(save_data, args.output_path)

    print(f"\n>>> Successfully extracted Top-{len(selected_clauses)} clause prototypes")
    print(f">>> Saved to: {args.output_path}")
    print(f">>> Abnormal reports used: {abnormal_count}")
    print(
        f">>> Clause mix: abnormal={sum(clause_type(c) == 'abnormal' for c in selected_clauses)}, "
        f"normal={sum(clause_type(c) == 'normal' for c in selected_clauses)}, "
        f"other={sum(clause_type(c) == 'other' for c in selected_clauses)}"
    )
    print("\n=== Top 10 Preview ===")
    for i, (clause, freq) in enumerate(zip(selected_clauses[:10], selected_counts[:10]), start=1):
        print(f"{i:02d}. [{freq}] ({clause_type(clause)}) {clause}")


if __name__ == "__main__":
    main()
