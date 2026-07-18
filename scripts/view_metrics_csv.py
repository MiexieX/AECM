#!/usr/bin/env python
import argparse
import csv
import os
import sys


def read_rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def format_value(value):
    if value is None:
        return ""
    text = str(value)
    try:
        num = float(text)
        return f"{num:.6f}"
    except ValueError:
        return text


def print_single_row(row, fieldnames, title=None):
    width = max(len(name) for name in fieldnames)
    if title:
        print(title)
        print("-" * len(title))
    for key in fieldnames:
        print(f"{key:<{width}} : {format_value(row.get(key, ''))}")


def print_table(rows, fieldnames):
    widths = {}
    for key in fieldnames:
        widths[key] = max(len(key), max(len(format_value(r.get(key, ""))) for r in rows))

    header = " | ".join(f"{key:<{widths[key]}}" for key in fieldnames)
    sep = "-+-".join("-" * widths[key] for key in fieldnames)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(f"{format_value(row.get(key, '')):<{widths[key]}}" for key in fieldnames))


def main():
    parser = argparse.ArgumentParser(description="Pretty-print one-row metric CSV files.")
    parser.add_argument("csv_path", help="Path to csv file")
    parser.add_argument(
        "--mode",
        choices=["auto", "vertical", "table"],
        default="auto",
        help="Display mode. auto uses vertical for single-row CSV and table otherwise.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    rows, fieldnames = read_rows(args.csv_path)
    if not rows:
        print(f"CSV has no rows: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    if mode == "auto":
        mode = "vertical" if len(rows) == 1 else "table"

    title = os.path.abspath(args.csv_path)
    if mode == "vertical":
        if len(rows) == 1:
            print_single_row(rows[0], fieldnames, title=title)
        else:
            for i, row in enumerate(rows, start=1):
                print_single_row(row, fieldnames, title=f"{title} [row {i}]")
                if i != len(rows):
                    print()
    else:
        print(title)
        print("-" * len(title))
        print_table(rows, fieldnames)


if __name__ == "__main__":
    main()
