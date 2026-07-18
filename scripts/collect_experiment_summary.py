#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv_rows(csv_path):
    try:
        with open(csv_path, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def sort_by_metric(rows, metric):
    def key_fn(row):
        try:
            return float(row.get(metric, "nan"))
        except Exception:
            return float("-inf")
    return sorted(rows, key=key_fn, reverse=True)


def write_rows(rows, out_path):
    if not rows:
        return False
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def collect_train_rows(root: Path):
    rows = []
    for csv_path in root.glob("train/*/mimic_cxr.csv"):
        for row in read_csv_rows(csv_path):
            row["experiment_dir"] = csv_path.parent.name
            row["source_csv"] = str(csv_path)
            rows.append(row)
    return sort_by_metric(rows, "test_BLEU_4")


def collect_eval_rows(root: Path):
    rows = []
    for csv_path in root.glob("test/**/mimic_cxr_clinical.csv"):
        for row in read_csv_rows(csv_path):
            row["result_dir"] = csv_path.parent.name
            row["source_csv"] = str(csv_path)
            rows.append(row)
    return sort_by_metric(rows, "test_BLEU_4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/mimic_cxr")
    args = parser.parse_args()

    root = Path(args.results_root)
    train_rows = collect_train_rows(root)
    eval_rows = collect_eval_rows(root)

    train_out = root / "experiment_summary_train.csv"
    eval_out = root / "experiment_summary_eval.csv"

    if write_rows(train_rows, train_out):
        print(f"Saved train summary to: {train_out}")
    else:
        print("No train experiment CSVs found.")

    if write_rows(eval_rows, eval_out):
        print(f"Saved eval summary to: {eval_out}")
    else:
        print("No eval clinical CSVs found.")


if __name__ == "__main__":
    main()
