#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict, OrderedDict

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import tokenizers_fn


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "with", "without", "within", "there", "these", "those",
    "seen", "noted", "again", "appears", "appear", "remains", "remain", "no",
}
BAD_CONNECTORS = {"or", "and"}
NEGATION_TOKENS = {"no", "without", "absent", "negative"}

PATHOLOGY_ALIASES = {
    "enlarged_cardiomediastinum": {"enlarged", "cardiomediastinal", "mediastinal", "silhouette"},
    "cardiomegaly": {"cardiomegaly", "cardiac", "heart", "enlargement"},
    "lung_opacity": {"opacity", "opacities", "airspace", "infiltrate"},
    "lung_lesion": {"lesion", "nodular", "nodule", "mass"},
    "edema": {"edema", "pulmonary", "vascular", "congestion"},
    "consolidation": {"consolidation", "airspace"},
    "pneumonia": {"pneumonia", "infectious", "infiltrate"},
    "atelectasis": {"atelectasis", "subsegmental", "bibasilar", "basilar"},
    "pneumothorax": {"pneumothorax", "apical", "pleural"},
    "pleural_effusion": {"pleural", "effusion", "effusions"},
    "pleural_other": {"pleural", "thickening"},
    "fracture": {"fracture", "fractures", "osseous", "rib"},
    "support_devices": {"tube", "catheter", "line", "device", "devices"},
    "no_finding": {"normal", "clear", "unremarkable"},
}

EXCLUDED_PATHOLOGIES = {"no_finding", "support_devices"}


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


def clean_report(tokenizer, report, dataset_name):
    if hasattr(tokenizer, "clean_report"):
        return tokenizer.clean_report(report)
    if dataset_name == "mimic_cxr":
        return tokenizer.clean_report_mimic_cxr(report)
    return tokenizer.clean_report_iu_xray(report)


def is_abnormal(label_item):
    polarity = label_item.get("pathology_polarity", {})
    for name, state in polarity.items():
        if name in EXCLUDED_PATHOLOGIES:
            continue
        if state in {"positive", "uncertain"}:
            return True
    pathologies = label_item.get("pathologies", {})
    for name, value in pathologies.items():
        if name in EXCLUDED_PATHOLOGIES:
            continue
        if int(value) == 1:
            return True
    return False


def pathology_keywords(label_item):
    keywords = set()
    polarity = label_item.get("pathology_polarity", {})
    for name, state in polarity.items():
        if name in EXCLUDED_PATHOLOGIES or state not in {"positive", "uncertain"}:
            continue
        keywords.update(name.split("_"))
        keywords.update(PATHOLOGY_ALIASES.get(name, set()))
    pathologies = label_item.get("pathologies", {})
    for name, value in pathologies.items():
        if name in EXCLUDED_PATHOLOGIES or int(value) != 1:
            continue
        keywords.update(name.split("_"))
        keywords.update(PATHOLOGY_ALIASES.get(name, set()))
    return keywords


def keep_phrase(tokens, keywords):
    if "." in tokens:
        return False
    if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
        return False
    if any(tok in NEGATION_TOKENS for tok in tokens):
        return False
    if any(tok in BAD_CONNECTORS for tok in tokens):
        return False
    non_stop = [tok for tok in tokens if tok not in STOPWORDS]
    if len(non_stop) < 2:
        return False
    if len(non_stop) / max(len(tokens), 1) < 0.5:
        return False
    return any(tok in keywords for tok in tokens)


def build_dense_token_mask(token_ids, token_texts, idf_map, vocab_size, stopword_weight):
    mask = torch.zeros(vocab_size + 1, dtype=torch.float32)
    weights = []
    for token_id, token in zip(token_ids, token_texts):
        if token_id <= 0 or token == ".":
            continue
        base = stopword_weight if token in STOPWORDS else 1.0
        weight = base * idf_map.get(token, 1.0)
        weights.append(weight)
        mask[token_id] = max(mask[token_id], weight)
    if weights:
        mask = mask / max(weights)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="config/mimic_cxr/vlci.json")
    parser.add_argument("--dataset_name", type=str, default="mimic_cxr")
    parser.add_argument("--tokenizer", type=str, default="mix")
    parser.add_argument("--pseudo_label_path", type=str, default="data/mimic_cxr_pseudo_labels.json")
    parser.add_argument("--output_path", type=str, default="data/mimic_cxr_explicit_cpmi_phrase_bank.pt")
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--min_ngram", type=int, default=3)
    parser.add_argument("--max_ngram", type=int, default=4)
    parser.add_argument("--stopword_weight", type=float, default=0.25)
    args = parser.parse_args()

    config = load_json_args(args.c)
    config["dataset_name"] = args.dataset_name
    config["tokenizer"] = args.tokenizer

    tokenizer = tokenizers_fn[config["tokenizer"]](config)
    with open(config["ann_path"], "r") as f:
        ann = json.load(f)
    with open(args.pseudo_label_path, "r") as f:
        pseudo = json.load(f)["labels"]

    doc_freq = Counter()
    abnormal_reports = []
    total_docs = 0

    for example in ann["train"]:
        report = clean_report(tokenizer, example["report"], config["dataset_name"])
        tokens = [tok for tok in report.split() if tok != "."]
        if not tokens:
            continue
        total_docs += 1
        for token in set(tokens):
            doc_freq[token] += 1

        label_item = pseudo.get(str(example["id"]))
        if not label_item or not is_abnormal(label_item):
            continue
        abnormal_reports.append((tokens, pathology_keywords(label_item)))

    idf = {}
    if doc_freq:
        raw_idf = {
            token: math.log((total_docs + 1) / (freq + 1)) + 1.0
            for token, freq in doc_freq.items()
        }
        max_idf = max(raw_idf.values())
        idf = {token: value / max_idf for token, value in raw_idf.items()}

    phrase_counter = Counter()
    phrase_sources = defaultdict(set)

    for tokens, keywords in abnormal_reports:
        for n in range(args.min_ngram, args.max_ngram + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i + n]
                if not keep_phrase(phrase_tokens, keywords):
                    continue
                phrase = " ".join(phrase_tokens)
                phrase_counter[phrase] += 1
                phrase_sources[phrase].update(keywords)

    unk_id = tokenizer.get_id_by_token("<unk>") if hasattr(tokenizer, "get_id_by_token") else None
    selected_phrases = []
    selected_token_ids = []
    selected_masks = []

    for phrase, _ in phrase_counter.most_common(args.top_k * 4):
        token_texts = phrase.split()
        token_ids = [tokenizer.get_id_by_token(token) for token in token_texts]
        if unk_id is not None and any(token_id == unk_id for token_id in token_ids):
            continue
        mask = build_dense_token_mask(
            token_ids,
            token_texts,
            idf,
            tokenizer.get_vocab_size(),
            args.stopword_weight,
        )
        if mask.sum().item() <= 0:
            continue
        selected_phrases.append(phrase)
        selected_token_ids.append(token_ids)
        selected_masks.append(mask)
        if len(selected_phrases) >= args.top_k:
            break

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(
        {
            "phrases": selected_phrases,
            "token_ids": selected_token_ids,
            "token_mask": torch.stack(selected_masks) if selected_masks else torch.empty(0),
            "meta": {
                "dataset_name": args.dataset_name,
                "tokenizer": args.tokenizer,
                "top_k": args.top_k,
                "min_ngram": args.min_ngram,
                "max_ngram": args.max_ngram,
            },
        },
        args.output_path,
    )
    print(f"Saved Explicit CPMI phrase bank to: {args.output_path}")
    print(f"Num abnormal reports used: {len(abnormal_reports)}")
    print(f"Num phrases saved: {len(selected_phrases)}")
    if selected_phrases:
        print("Top phrases:")
        for phrase in selected_phrases[:10]:
            print("  -", phrase)


if __name__ == "__main__":
    main()
