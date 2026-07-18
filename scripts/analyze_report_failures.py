#!/usr/bin/env python
import argparse
import csv
import os
import re
from collections import Counter


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
    "sternotomy",
    "cabg",
    "subcutaneous gas",
    "lymph node",
    "hilar",
    "mediastinal",
)

HALLUCINATION_TOKENS = (
    "endotracheal tube",
    "nasogastric tube",
    "ng tube",
    "ij central venous catheter",
    "picc",
    "chest tube",
    "cabg",
    "sternotomy",
    "intubated",
)


def count_hits(text, patterns):
    text = text.lower()
    return sum(1 for p in patterns if p in text)


def classify_report(text):
    text = text.lower()
    normal_hits = count_hits(text, NORMAL_PATTERNS)
    abnormal_hits = count_hits(text, ABNORMAL_KEYWORDS)
    if abnormal_hits == 0 and normal_hits > 0:
        return "normal_like"
    if abnormal_hits > 0 and normal_hits == 0:
        return "abnormal_like"
    if abnormal_hits > 0 and normal_hits > 0:
        return "mixed"
    return "other"


def word_len(text):
    return len(re.findall(r"\w+", text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_csv", help="Path to mimic_cxr_reports.csv")
    parser.add_argument("--out_dir", default=None, help="Optional output directory")
    parser.add_argument("--top_repeat", type=int, default=20)
    parser.add_argument("--dump_examples", type=int, default=50)
    args = parser.parse_args()

    with open(args.report_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(args.report_csv), "failure_analysis")
    os.makedirs(out_dir, exist_ok=True)

    pred_col = "generated_report"
    ref_col = "reference_report"

    preds = [row[pred_col].strip() for row in rows]
    pred_counter = Counter(preds)

    summary = []
    normal_cases = []
    abnormal_cases = []
    mixed_cases = []
    hallucination_cases = []
    repeated_cases = []

    for row in rows:
        pred = row[pred_col].strip()
        ref = row[ref_col].strip()
        pred_type = classify_report(pred)
        ref_type = classify_report(ref)
        repeat_count = pred_counter[pred]

        item = {
            "split": row.get("split", ""),
            "image_id": row.get("image_id", ""),
            "pred_type": pred_type,
            "ref_type": ref_type,
            "repeat_count": repeat_count,
            "pred_words": word_len(pred),
            "ref_words": word_len(ref),
            "reference_report": ref,
            "generated_report": pred,
        }
        summary.append(item)

        if pred_type == "normal_like":
            normal_cases.append(item)
        elif pred_type == "abnormal_like":
            abnormal_cases.append(item)
        elif pred_type == "mixed":
            mixed_cases.append(item)

        if repeat_count >= 10:
            repeated_cases.append(item)

        pred_lower = pred.lower()
        ref_lower = ref.lower()
        if any(tok in pred_lower for tok in HALLUCINATION_TOKENS) and not any(
            tok in ref_lower for tok in HALLUCINATION_TOKENS
        ):
            hallucination_cases.append(item)

    def dump_csv(path, items):
        if not items:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            writer.writeheader()
            writer.writerows(items)

    repeated_top = sorted(repeated_cases, key=lambda x: (-x["repeat_count"], x["image_id"]))
    dump_csv(os.path.join(out_dir, "normal_like_predictions.csv"), normal_cases[: args.dump_examples])
    dump_csv(os.path.join(out_dir, "abnormal_like_predictions.csv"), abnormal_cases[: args.dump_examples])
    dump_csv(os.path.join(out_dir, "mixed_predictions.csv"), mixed_cases[: args.dump_examples])
    dump_csv(os.path.join(out_dir, "hallucination_cases.csv"), hallucination_cases[: args.dump_examples])
    dump_csv(os.path.join(out_dir, "repeated_template_cases.csv"), repeated_top[: args.dump_examples])

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"total_rows: {len(rows)}\n")
        f.write(f"unique_predictions: {len(pred_counter)}\n")
        f.write(f"normal_like_predictions: {len(normal_cases)}\n")
        f.write(f"abnormal_like_predictions: {len(abnormal_cases)}\n")
        f.write(f"mixed_predictions: {len(mixed_cases)}\n")
        f.write(f"hallucination_cases: {len(hallucination_cases)}\n")
        f.write(f"repeated_template_cases(repeat>=10): {len(repeated_cases)}\n\n")
        f.write("top repeated predictions:\n")
        for text, count in pred_counter.most_common(args.top_repeat):
            short = text[:220].replace("\n", " ")
            f.write(f"[{count}] {short}\n")

    print(f"Saved failure analysis to: {out_dir}")
    print(f"total_rows={len(rows)} unique_predictions={len(pred_counter)}")
    print(
        f"normal_like={len(normal_cases)} abnormal_like={len(abnormal_cases)} mixed={len(mixed_cases)} "
        f"hallucination_cases={len(hallucination_cases)} repeated_cases={len(repeated_cases)}"
    )


if __name__ == "__main__":
    main()
