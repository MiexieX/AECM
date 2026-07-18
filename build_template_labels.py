#!/usr/bin/env python3
"""Build report-template pseudo labels for BLEU-oriented prompt supervision.

The labels are train-report derived and intentionally coarse. They teach the
prompt module which reporting style is likely needed: normal template,
negated findings, positive findings, support devices, or comparison language.
"""

import argparse
import json
import os
import re
from collections import Counter


TEMPLATES = [
    "normal_template",
    "negation_template",
    "finding_template",
    "device_template",
    "comparison_template",
]

PATTERNS = {
    "normal_template": [
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary disease",
        "no active disease",
        "normal chest",
        "lungs are clear",
        "heart size is normal",
        "within normal limits",
        "unremarkable",
    ],
    "negation_template": [
        " no ",
        "without",
        "negative for",
        "no evidence of",
        "no focal",
        "no pleural",
        "no pneumothorax",
        "no acute",
        "absent",
    ],
    "finding_template": [
        "opacity",
        "opacities",
        "edema",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "pneumothorax",
        "effusion",
        "cardiomegaly",
        "enlarged",
        "nodule",
        "mass",
        "fracture",
        "vascular congestion",
    ],
    "device_template": [
        "tube",
        "line",
        "catheter",
        "picc",
        "pacemaker",
        "port",
        "stent",
        "support device",
        "endotracheal",
        "enteric",
        "nasogastric",
    ],
    "comparison_template": [
        "unchanged",
        "stable",
        "interval",
        "improved",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "worsened",
        "worsening",
        "compared",
        "again",
        "persistent",
    ],
}


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def label_report(report):
    report = normalize_text(report)
    labels = {}
    for name in TEMPLATES:
        labels[name] = int(any(pattern in report for pattern in PATTERNS[name]))

    # Almost every report has either normal/negative/finding language. If a rare
    # report misses all rules, treat it as a generic finding-style report so the
    # sample remains supervised instead of being all-zero.
    if not any(labels.values()):
        labels["finding_template"] = 1
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build template pseudo labels from train reports.")
    parser.add_argument("--ann_path", default="data/mimic_cxr/annotation.json")
    parser.add_argument("--output", default="data/mimic_cxr_template_labels.json")
    args = parser.parse_args()

    with open(args.ann_path, "r") as f:
        data = json.load(f)

    train_items = data.get("train", [])
    labels = {}
    counts = Counter()
    for item in train_items:
        image_id = item.get("id")
        if not image_id:
            continue
        item_labels = label_report(item.get("report", ""))
        labels[str(image_id)] = {
            "study_id": item.get("study_id", image_id),
            "templates": item_labels,
        }
        counts.update({name: value for name, value in item_labels.items() if value})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "templates": TEMPLATES,
                "labels": labels,
                "positive_counts": dict(counts),
            },
            f,
        )

    print(f"Loaded train items: {len(train_items)}")
    print(f"Saved template labels: {len(labels)} -> {args.output}")
    for name in TEMPLATES:
        print(f"  {name}: {counts.get(name, 0)}")


if __name__ == "__main__":
    main()
