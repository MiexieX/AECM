#!/usr/bin/env python3
"""Offline train-corpus RAG-lite reranking for generated beam candidates.

This is deliberately conservative: it does not use test references. It builds a
train-report n-gram prior and prefers beam candidates that look more like the
training report language while still respecting original beam rank.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict


BAD_ENDINGS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "in", "is",
    "no", "of", "or", "the", "to", "with", "without",
}

BAD_PATTERNS = [
    "no pleural effusion or",
    "there is no pleural",
    "the heart is",
    "the cardiac and",
    "the trachea is",
    "the aorta is",
    "of the thoracic aorta",
]


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text):
    return normalize_text(text).split()


def ngrams(words, n):
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def load_references(top1_csv):
    references = {}
    top1_reports = {}
    with open(top1_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            references[image_id] = row.get("reference_report", "")
            top1_reports[image_id] = row.get("generated_report", "")
    return references, top1_reports


def build_train_ngram_counts(ann_path, max_ngram=4):
    with open(ann_path, "r") as f:
        data = json.load(f)

    counts = {n: Counter() for n in range(1, max_ngram + 1)}
    train_items = data.get("train", [])
    for item in train_items:
        words = tokens(item.get("report", ""))
        if len(words) < 3:
            continue
        for n in range(1, max_ngram + 1):
            counts[n].update(ngrams(words, n))
    return counts, len(train_items)


def train_ngram_score(report, counts, min_count):
    words = tokens(report)
    if not words:
        return -3.0

    weights = {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.50}
    score = 0.0
    used = 0.0
    for n, weight in weights.items():
        grams = ngrams(words, n)
        if not grams:
            continue
        values = []
        for gram in grams:
            count = counts[n].get(gram, 0)
            if count >= min_count:
                values.append(math.log1p(count))
        if values:
            # Reward frequent train-report phrases, especially 3/4-grams.
            score += weight * (sum(values) / len(grams))
        else:
            score -= weight * 0.5
        used += weight
    return score / max(used, 1e-6)


def length_score(report, target_len):
    length = len(tokens(report))
    if length == 0:
        return -2.0
    return -abs(length - target_len) / target_len


def repetition_penalty(report):
    words = tokens(report)
    if len(words) < 4:
        return 0.5
    trigrams = ngrams(words, 3)
    repeated_trigrams = len(trigrams) - len(set(trigrams))
    sentences = [normalize_text(s) for s in re.split(r"[.;]+", str(report)) if normalize_text(s)]
    repeated_sentences = len(sentences) - len(set(sentences))
    return repeated_sentences * 0.7 + repeated_trigrams * 0.08


def fragment_penalty(report):
    report_norm = normalize_text(report)
    penalty = 0.0
    for pattern in BAD_PATTERNS:
        if pattern in report_norm:
            penalty += 0.6
    words = report_norm.split()
    if words and words[-1] in BAD_ENDINGS:
        penalty += 1.0
    return penalty


def score_candidate(row, counts, args):
    report = row.get("candidate_report", "")
    rank = int(row.get("candidate_rank", 0))
    prior_total = float(row.get("total", 0.0) or 0.0)
    ngram = train_ngram_score(report, counts, args.min_ngram_count)
    length = length_score(report, args.target_len)
    repeat = repetition_penalty(report)
    fragment = fragment_penalty(report)

    total = (
        args.prior_weight * prior_total
        - args.rank_weight * rank
        + args.ngram_weight * ngram
        + args.length_weight * length
        - args.repeat_weight * repeat
        - args.fragment_weight * fragment
    )
    return total, {
        "rag_total": total,
        "rag_train_ngram_score": ngram,
        "rag_length_score": length,
        "rag_repeat_penalty": repeat,
        "rag_fragment_penalty": fragment,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", default="data/mimic_cxr/annotation.json")
    parser.add_argument("--candidates_csv", default="results/mimic_cxr/test/rerank_b3/mimic_cxr_rerank_candidates.csv")
    parser.add_argument("--top1_csv", default="results/mimic_cxr/test/rerank_b3/mimic_cxr_top1_reports.csv")
    parser.add_argument("--output_dir", default="results/mimic_cxr/test/rag_ngram_b3")
    parser.add_argument("--task_name", default="rag_ngram_b3")
    parser.add_argument("--prior_weight", type=float, default=0.25)
    parser.add_argument("--rank_weight", type=float, default=0.18)
    parser.add_argument("--ngram_weight", type=float, default=0.42)
    parser.add_argument("--length_weight", type=float, default=0.05)
    parser.add_argument("--repeat_weight", type=float, default=0.08)
    parser.add_argument("--fragment_weight", type=float, default=0.20)
    parser.add_argument("--target_len", type=float, default=55.0)
    parser.add_argument("--min_ngram_count", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f">>> Building train n-gram prior from: {args.ann_path}", flush=True)
    counts, train_size = build_train_ngram_counts(args.ann_path)
    print(f">>> Loaded {train_size} train reports", flush=True)
    print(
        ">>> RAG-lite weights:",
        {
            "prior": args.prior_weight,
            "rank": args.rank_weight,
            "ngram": args.ngram_weight,
            "length": args.length_weight,
            "repeat": args.repeat_weight,
            "fragment": args.fragment_weight,
        },
        flush=True,
    )

    references, top1_reports = load_references(args.top1_csv)
    grouped = defaultdict(list)
    with open(args.candidates_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["image_id"]].append(row)

    rows = []
    candidate_debug = []
    changed = 0
    for image_id, candidates in grouped.items():
        scored = []
        for row in candidates:
            total, parts = score_candidate(row, counts, args)
            scored.append((total, int(row.get("candidate_rank", 0)), row, parts))
            debug_row = dict(row)
            debug_row.update(parts)
            candidate_debug.append(debug_row)

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_total, best_rank, best_row, best_parts = scored[0]
        if best_rank != 0:
            changed += 1
        rows.append({
            "split": best_row.get("split", "test"),
            "image_id": image_id,
            "reference_report": references.get(image_id, ""),
            "generated_report": best_row.get("candidate_report", ""),
            "original_top1_report": top1_reports.get(image_id, ""),
            "selected_rank": best_rank,
            "changed_by_rag_rerank": int(best_rank != 0),
            **best_parts,
        })

    report_path = os.path.join(args.output_dir, "mimic_cxr_reports.csv")
    candidate_path = os.path.join(args.output_dir, "mimic_cxr_rag_ngram_candidates.csv")
    fieldnames = [
        "split", "image_id", "reference_report", "generated_report",
        "original_top1_report", "selected_rank", "changed_by_rag_rerank",
        "rag_total", "rag_train_ngram_score", "rag_length_score",
        "rag_repeat_penalty", "rag_fragment_penalty",
    ]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    debug_fields = list(candidate_debug[0].keys()) if candidate_debug else []
    with open(candidate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fields)
        writer.writeheader()
        writer.writerows(candidate_debug)

    total = len(rows)
    print(f"Saved RAG-lite reports to: {report_path}")
    print(f"Saved candidate debug to: {candidate_path}")
    print(f"rerank changed {changed}/{total} reports ({changed / max(total, 1):.2%})")


if __name__ == "__main__":
    main()
