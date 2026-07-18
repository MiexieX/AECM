import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache


REGION_KEYWORDS = {
    "heart": ["heart", "cardiac", "cardiomegaly", "cardiomediastinal", "cardiomediastinal silhouette"],
    "left lung": [
        "left lung",
        "left hemithorax",
        "left base",
        "left upper lobe",
        "left lower lobe",
        "left apex",
        "left apical",
        "left-sided",
        "left pulmonary",
    ],
    "right lung": [
        "right lung",
        "right hemithorax",
        "right base",
        "right upper lobe",
        "right lower lobe",
        "right apex",
        "right apical",
        "right-sided",
        "right pulmonary",
    ],
    "mediastinum": ["mediastinum", "mediastinal"],
    "hila": ["hilar", "hila", "hilum"],
    "pleura": ["pleural", "effusion", "pneumothorax", "hydropneumothorax"],
    "diaphragm": ["diaphragm", "hemidiaphragm", "subdiaphragmatic"],
    "ribs": ["rib", "ribs", "osseous", "bony", "bone", "fracture", "sternum", "sternal"],
    "spine": ["spine", "thoracic spine", "degenerative", "vertebral", "vertebrae"],
    "trachea": ["trachea", "tracheal", "endotracheal tube"],
    "aortic arch": ["aorta", "aortic", "aortic knob", "thoracic aorta"],
}

# CheXpert-style labels. "no_finding" is retained for report-level pseudo labels,
# but most memory cells are driven by the 13 concrete finding/device categories.
PATHOLOGY_KEYWORDS = {
    "enlarged_cardiomediastinum": [
        "enlarged cardiomediastinum",
        "cardiomediastinal enlargement",
        "widened mediastinum",
        "mediastinal widening",
        "prominent cardiomediastinal",
    ],
    "cardiomegaly": [
        "cardiomegaly",
        "cardiac silhouette is enlarged",
        "enlarged cardiac silhouette",
        "heart size is enlarged",
        "heart is enlarged",
        "mildly enlarged heart",
    ],
    "lung_opacity": [
        "opacity",
        "opacities",
        "airspace opacity",
        "hazy opacity",
        "infiltrate",
        "infiltration",
        "density",
        "densities",
    ],
    "lung_lesion": [
        "nodule",
        "nodules",
        "nodular",
        "mass",
        "lesion",
        "granuloma",
        "granulomas",
    ],
    "edema": [
        "edema",
        "pulmonary edema",
        "vascular congestion",
        "pulmonary vascular congestion",
        "interstitial edema",
    ],
    "consolidation": ["consolidation", "consolidative", "airspace consolidation"],
    "pneumonia": ["pneumonia", "infectious", "infection"],
    "atelectasis": ["atelectasis", "atelectatic", "volume loss"],
    "pneumothorax": ["pneumothorax"],
    "pleural_effusion": ["pleural effusion", "pleural effusions", "effusion", "effusions"],
    "pleural_other": ["pleural thickening", "pleural scarring", "pleural plaque", "pleural disease"],
    "fracture": ["fracture", "fractures", "deformity", "deformities"],
    "support_devices": [
        "tube",
        "catheter",
        "line",
        "picc",
        "central venous",
        "pacemaker",
        "port",
        "stent",
        "endotracheal tube",
        "enteric tube",
        "nasogastric tube",
        "chest tube",
        "support device",
        "support devices",
    ],
    "no_finding": [
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary disease",
        "no active disease",
        "no acute disease",
        "normal chest",
        "lungs are clear",
        "heart size is normal",
        "cardiomediastinal silhouette is normal",
    ],
}

# Keep the 2D memory medically plausible. This is the most important BLEU/F1
# safety guard: a phrase can mention both "heart" and "opacity", but that does
# not mean it should populate the heart x lung_opacity cell.
PATHOLOGY_REGION_ALLOWED = {
    "enlarged_cardiomediastinum": {"heart", "mediastinum"},
    "cardiomegaly": {"heart"},
    "lung_opacity": {"left lung", "right lung", "hila"},
    "lung_lesion": {"left lung", "right lung", "hila"},
    "edema": {"left lung", "right lung", "hila"},
    "consolidation": {"left lung", "right lung"},
    "pneumonia": {"left lung", "right lung"},
    "atelectasis": {"left lung", "right lung"},
    "pneumothorax": {"pleura", "left lung", "right lung"},
    "pleural_effusion": {"pleura", "left lung", "right lung"},
    "pleural_other": {"pleura"},
    "fracture": {"ribs", "spine"},
    "support_devices": {"trachea", "mediastinum", "heart", "left lung", "right lung", "pleura"},
    "no_finding": {"heart", "left lung", "right lung", "mediastinum", "hila", "pleura"},
}

POLARITIES = ("positive", "negative", "uncertain")
POLARITY_RANK = {"absent": 0, "negative": 1, "uncertain": 2, "positive": 3}

BOILERPLATE_PATTERNS = [
    "frontal and lateral views of the chest",
    "ap upright and lateral views of the chest",
    "pa and lateral views of the chest",
    "single portable view of the chest",
    "portable ap upright chest film",
    "comparison is made with the next preceding similar study",
    "as compared to the previous radiograph",
    "in comparison with the study of",
    "analysis is performed in direct comparison",
    "the study is compared to",
]

BAD_ENDINGS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "in",
    "is",
    "no",
    "of",
    "or",
    "the",
    "to",
    "with",
    "without",
}

BAD_MEMORY_PATTERNS = [
    "difficult to assess",
    "limited evaluation",
    "cannot be assessed",
    "poorly evaluated",
    "suboptimally evaluated",
    "not well seen",
    "not well visualized",
]

AMBIGUOUS_OR_PATTERNS = [
    "pneumonia or",
    "consolidation or",
    "edema or",
    "atelectasis or",
    "opacity or",
    "or pneumonia",
    "or consolidation",
    "or edema",
    "or atelectasis",
    "or congestive heart failure",
    "or chf",
]

UNCERTAIN_PATTERNS = [
    "not excluded",
    "not exclude",
    "cannot be excluded",
    "cannot exclude",
    "suspicious",
    "concerning for",
    "raises concern",
]

NEGATIVE_PATTERNS = [
    "clear of",
]

MIN_WORDS = 3
MAX_WORDS = 18


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=None)
def phrase_pattern(phrase):
    return re.compile(r"\b" + re.escape(phrase) + r"\b")


def contains_phrase(text, phrase):
    return phrase_pattern(phrase).search(text) is not None


def split_into_clauses(text):
    clauses = re.split(r"[.,;\n]+", text)
    cleaned = []
    for clause in clauses:
        clause = normalize_text(clause)
        if not clause:
            continue
        words = clause.split()
        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue
        if words[-1] in BAD_ENDINGS:
            continue
        if any(pattern in clause for pattern in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(clause)
    return cleaned


def matched_regions(clause):
    regions = []
    for region, keywords in REGION_KEYWORDS.items():
        if any(contains_phrase(clause, kw) for kw in keywords):
            regions.append(region)

    # Generic lung statements are clinically useful but do not carry laterality.
    # Add both lungs only when no laterality-specific lung region was found.
    if "left lung" not in regions and "right lung" not in regions:
        if contains_phrase(clause, "lungs") or contains_phrase(clause, "pulmonary"):
            regions.extend(["left lung", "right lung"])
    return sorted(set(regions), key=list(REGION_KEYWORDS.keys()).index)


def matched_pathologies(clause):
    matched = []
    for pathology, keywords in PATHOLOGY_KEYWORDS.items():
        for keyword in keywords:
            if contains_phrase(clause, keyword):
                matched.append((pathology, keyword))
                break
    return matched


def classify_polarity(clause, keyword):
    escaped = re.escape(keyword)
    negative_before = (
        r"\b(no|without|absent|absence of|free of|clear of|negative for|no evidence of|no sign of|no signs of)\b"
        r"(.{0,80})\b"
        + escaped
        + r"\b"
    )
    negative_after = (
        r"\b"
        + escaped
        + r"\b(.{0,80})\b(absent|not present|not seen|not identified|not evident|resolved)\b"
    )
    uncertain = (
        r"\b(possible|possibly|probable|probably|questionable|may represent|may reflect|suggest|suggests|"
        r"suspicious for|cannot exclude|could represent|likely)\b"
    )

    is_negative = re.search(negative_before, clause) or re.search(negative_after, clause)
    if is_negative:
        return "negative"
    if re.search(uncertain, clause):
        return "uncertain"
    if any(pattern in clause for pattern in UNCERTAIN_PATTERNS):
        return "uncertain"
    if any(pattern in clause for pattern in NEGATIVE_PATTERNS) and contains_phrase(clause, keyword):
        return "negative"
    if any(pattern in clause for pattern in AMBIGUOUS_OR_PATTERNS):
        return "uncertain"
    return "positive"


def update_best_polarity(current, candidate):
    if POLARITY_RANK[candidate] > POLARITY_RANK[current]:
        return candidate
    return current


def phrase_score(phrase, count, shared_cells):
    length_bonus = 1.0 + 0.08 * min(len(phrase.split()), 14)
    specificity_penalty = math.sqrt(max(1, shared_cells))
    return (count * length_bonus) / specificity_penalty


def compatible_regions(pathology, regions):
    allowed = PATHOLOGY_REGION_ALLOWED.get(pathology)
    if not allowed:
        return regions
    return [region for region in regions if region in allowed]


def is_memory_phrase_safe(phrase, polarity):
    if any(pattern in phrase for pattern in BAD_MEMORY_PATTERNS):
        return False

    # Positive cells are used as high-confidence clinical hints. Keep ambiguous
    # differential phrases out of them so the memory does not inflate false
    # positives during reranking or soft-bias decoding.
    if polarity == "positive" and any(pattern in phrase for pattern in AMBIGUOUS_OR_PATTERNS):
        return False
    if polarity == "positive" and any(pattern in phrase for pattern in UNCERTAIN_PATTERNS):
        return False

    return True


def load_split_items(ann_path, split):
    with open(ann_path, "r") as f:
        data = json.load(f)
    return data.get(split, [])


def build_memory_and_labels(ann_path, output_memory, output_labels, top_k_per_cell, min_count):
    train_items = load_split_items(ann_path, "train")
    print(f"Loaded {len(train_items)} train items from {ann_path}", flush=True)

    memory_counts = {
        region: {
            pathology: {polarity: Counter() for polarity in POLARITIES}
            for pathology in PATHOLOGY_KEYWORDS
        }
        for region in REGION_KEYWORDS
    }
    phrase_cell_sets = defaultdict(set)
    pseudo_labels = {}
    seen_memory_studies = set()
    study_label_cache = {}

    region_names = list(REGION_KEYWORDS.keys())
    pathology_names = list(PATHOLOGY_KEYWORDS.keys())

    for item in train_items:
        image_id = item.get("id")
        report = item.get("report", "")
        study_id = item.get("study_id", image_id)
        cache_key = (study_id, report)

        if cache_key in study_label_cache:
            cached = study_label_cache[cache_key]
            if image_id:
                pseudo_labels[image_id] = {
                    "study_id": study_id,
                    "image_path": item.get("image_path", []),
                    "regions": cached["regions"],
                    "pathologies": cached["pathologies"],
                    "pathology_polarity": cached["pathology_polarity"],
                    "region_pathology_polarity": cached["region_pathology_polarity"],
                }
            continue

        clauses = split_into_clauses(report)

        item_regions = {region: 0 for region in region_names}
        item_pathologies = {pathology: 0 for pathology in pathology_names}
        item_polarities = {pathology: "absent" for pathology in pathology_names}
        item_region_pathology = {
            region: {pathology: "absent" for pathology in pathology_names}
            for region in region_names
        }

        add_to_memory = study_id not in seen_memory_studies
        if add_to_memory:
            seen_memory_studies.add(study_id)

        for clause in clauses:
            regions = matched_regions(clause)
            pathologies = matched_pathologies(clause)
            if not regions and not pathologies:
                continue

            for region in regions:
                item_regions[region] = 1

            for pathology, keyword in pathologies:
                polarity = classify_polarity(clause, keyword)
                item_polarities[pathology] = update_best_polarity(item_polarities[pathology], polarity)
                if polarity == "positive":
                    item_pathologies[pathology] = 1

                if not regions:
                    continue

                filtered_regions = compatible_regions(pathology, regions)
                if not filtered_regions:
                    continue

                for region in filtered_regions:
                    item_region_pathology[region][pathology] = update_best_polarity(
                        item_region_pathology[region][pathology], polarity
                    )
                    if add_to_memory and is_memory_phrase_safe(clause, polarity):
                        memory_counts[region][pathology][polarity][clause] += 1
                        phrase_cell_sets[clause].add((region, pathology, polarity))

        if image_id:
            pseudo_labels[image_id] = {
                "study_id": study_id,
                "image_path": item.get("image_path", []),
                "regions": item_regions,
                "pathologies": item_pathologies,
                "pathology_polarity": item_polarities,
                "region_pathology_polarity": item_region_pathology,
            }
        study_label_cache[cache_key] = {
            "regions": item_regions,
            "pathologies": item_pathologies,
            "pathology_polarity": item_polarities,
            "region_pathology_polarity": item_region_pathology,
        }

    memory = {
        region: {
            pathology: {polarity: [] for polarity in POLARITIES}
            for pathology in pathology_names
        }
        for region in region_names
    }

    non_empty_cells = 0
    total_phrases = 0
    for region in region_names:
        for pathology in pathology_names:
            for polarity in POLARITIES:
                counter = memory_counts[region][pathology][polarity]
                scored = []
                for phrase, count in counter.items():
                    if count < min_count:
                        continue
                    shared_cells = len(phrase_cell_sets[phrase])
                    scored.append((phrase_score(phrase, count, shared_cells), count, phrase))
                scored.sort(key=lambda x: (x[0], x[1], len(x[2].split())), reverse=True)
                phrases = [phrase for _, _, phrase in scored[:top_k_per_cell]]
                memory[region][pathology][polarity] = phrases
                if phrases:
                    non_empty_cells += 1
                    total_phrases += len(phrases)

    os.makedirs(os.path.dirname(output_memory), exist_ok=True)
    with open(output_memory, "w") as f:
        json.dump(
            {
                "regions": region_names,
                "pathologies": pathology_names,
                "polarities": list(POLARITIES),
                "pathology_region_allowed": {
                    pathology: sorted(regions)
                    for pathology, regions in PATHOLOGY_REGION_ALLOWED.items()
                },
                "memory": memory,
            },
            f,
            indent=2,
        )

    with open(output_labels, "w") as f:
        json.dump(
            {
                "regions": region_names,
                "pathologies": pathology_names,
                "polarity_rank": POLARITY_RANK,
                "pathology_region_allowed": {
                    pathology: sorted(regions)
                    for pathology, regions in PATHOLOGY_REGION_ALLOWED.items()
                },
                "labels": pseudo_labels,
            },
            f,
        )

    print(f"Saved 2D memory to: {output_memory}", flush=True)
    print(f"Saved pseudo labels to: {output_labels}", flush=True)
    print(f"Unique studies used for memory: {len(seen_memory_studies)}", flush=True)
    print(f"Pseudo-labeled images: {len(pseudo_labels)}", flush=True)
    print(f"Non-empty region/pathology/polarity cells: {non_empty_cells}", flush=True)
    print(f"Total retained phrases: {total_phrases}", flush=True)

    print("\nPreview:", flush=True)
    preview_count = 0
    for region in region_names:
        for pathology in pathology_names:
            for polarity in POLARITIES:
                phrases = memory[region][pathology][polarity]
                if phrases:
                    print(f"  [{region}][{pathology}][{polarity}] -> {phrases[:3]}", flush=True)
                    preview_count += 1
                    if preview_count >= 10:
                        return


def main():
    parser = argparse.ArgumentParser(
        description="Build an anatomy x pathology x polarity memory bank and pseudo labels from train reports."
    )
    parser.add_argument("--ann_path", default="data/mimic_cxr/annotation.json")
    parser.add_argument("--output_memory", default="data/anatomy_pathology_polarity_memory.json")
    parser.add_argument("--output_labels", default="data/mimic_cxr_pseudo_labels.json")
    parser.add_argument("--top_k_per_cell", type=int, default=30)
    parser.add_argument("--min_count", type=int, default=2)
    args = parser.parse_args()

    build_memory_and_labels(
        ann_path=args.ann_path,
        output_memory=args.output_memory,
        output_labels=args.output_labels,
        top_k_per_cell=args.top_k_per_cell,
        min_count=args.min_count,
    )


if __name__ == "__main__":
    main()
