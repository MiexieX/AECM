"""Standard Clinical Efficacy Scoring for Radiology Reports.

This module uses the standard CheXbert/F1CheXbert methodology to extract
14 clinical observations and compute the micro-averaged CE-P, CE-R, and CE-F1.
"""

import os
from pathlib import Path


def _huggingface_cache_dir():
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    if "XDG_CACHE_HOME" in os.environ:
        return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _has_local_bert_base_uncased():
    snapshots_dir = _huggingface_cache_dir() / "models--bert-base-uncased" / "snapshots"
    if not snapshots_dir.exists():
        return False

    required_files = {"config.json", "tokenizer_config.json", "vocab.txt"}
    for snapshot_dir in snapshots_dir.iterdir():
        if not snapshot_dir.is_dir():
            continue
        snapshot_files = {path.name for path in snapshot_dir.iterdir()}
        if required_files.issubset(snapshot_files):
            return True
    return False


def _has_local_chexbert_checkpoint():
    if "CHEXBERT_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["CHEXBERT_CACHE_DIR"])
    elif "XDG_CACHE_HOME" in os.environ:
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "chexbert"
    else:
        cache_dir = Path.home() / ".cache" / "chexbert"
    return (cache_dir / "chexbert.pth").exists()


def _configure_f1chexbert_runtime():
    # f1chexbert may still try a remote metadata lookup even when the model
    # artifacts are already cached. Switch to offline mode only when both local
    # dependencies are present so evaluation stays deterministic.
    if _has_local_bert_base_uncased() and _has_local_chexbert_checkpoint():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        # A stale mirror endpoint can still appear in warning messages before
        # offline mode is honored by downstream libraries.
        os.environ.pop("HF_ENDPOINT", None)
        print("\n[*] Using cached bert-base-uncased/CheXbert artifacts in offline mode...")


def compute_clinical_metrics(gts, res):
    """Compute micro-averaged CE-P / CE-R / CE-F1 using official standard."""
    # 确保 gts 和 res 的顺序严格对齐
    image_ids = list(gts.keys())
    
    # 展平文本（由于有时一个 image_id 对应一个列表，取其第一条报告）
    gt_reports = [gts[img_id][0] for img_id in image_ids]
    pred_reports = [res.get(img_id, [""])[0] for img_id in image_ids]

    # 初始化打分器 (默认会自动调用 cuda 加速)
    _configure_f1chexbert_runtime()
    try:
        from f1chexbert import F1CheXbert
    except ImportError:
        print("F1CheXbert is not installed. Returning 0.0 for clinical metrics.")
        return {"CE_P": 0.0, "CE_R": 0.0, "CE_F1": 0.0}

    try:
        chexbert_evaluator = F1CheXbert(device="cuda")
    except Exception as exc:
        print(f"F1CheXbert initialization failed: {exc}")
        return {"CE_P": 0.0, "CE_R": 0.0, "CE_F1": 0.0}

    print("\n[*] Computing clinical metrics using standard f1chexbert...")
    
    # f1chexbert 官方 API 规范：必须同时传入 hyps (预测值) 和 refs (真实值)
    accuracy, accuracy_not_averaged, class_report, class_report_5 = chexbert_evaluator(
        hyps=pred_reports, 
        refs=gt_reports
    )

    ce_p, ce_r, ce_f1 = 0.0, 0.0, 0.0

    # f1chexbert 底层调用了 sklearn，class_report 通常是一个字典
    if isinstance(class_report, dict):
        if "micro avg" in class_report:
            ce_p = class_report["micro avg"]["precision"]
            ce_r = class_report["micro avg"]["recall"]
            ce_f1 = class_report["micro avg"]["f1-score"]
    else:
        # 兜底防爆机制：如果 class_report 是一个打印格式的字符串，使用正则强行提取
        import re
        match = re.search(r'micro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', class_report)
        if match:
            ce_p, ce_r, ce_f1 = map(float, match.groups())

    clinical_metrics = {
        "CE_P": float(ce_p),
        "CE_R": float(ce_r),
        "CE_F1": float(ce_f1)
    }

    return clinical_metrics
