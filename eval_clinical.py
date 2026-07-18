#!/usr/bin/env python3
"""Standalone clinical evaluation for radiology report generation.

This script can either:
1. export standard generation metrics together with clinical metrics, or
2. read a pre-generated report CSV and compute clinical metrics.

The clinical score follows the paper's CheXpert-style label logic and
is dependency-free so it can run in the current environment.
"""

import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from metric.clinical import compute_clinical_metrics
from metric.metrics import compute_scores
from models import model_fn
from utils import R2DataLoader, tokenizers_fn


def load_json_args(path):
    json_str = ""
    with open(path, "r") as f:
        for line in f:
            line = line.split("//")[0] + "\n"
            json_str += line
    defaults = json.loads(json_str, object_pairs_hook=OrderedDict)
    dict_args = {}
    for key in defaults.keys():
        dict_args.update(defaults[key])
    return dict_args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 关闭 TF32，确保推理精度与论文严格对齐
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def infer_split(model, dataloader):
    model.eval()
    gts, res = [], []
    with torch.no_grad():
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = model(images, mode="sample")
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            res.extend(reports)
            gts.extend(ground_truths)
            print(f"\rProcessing: [{int((batch_idx + 1) / len(dataloader) * 100)}%]", end="", flush=True)
    print()
    gt_dict = {i: [gt] for i, gt in enumerate(gts)}
    res_dict = {i: [re] for i, re in enumerate(res)}
    metrics = compute_scores(gt_dict, res_dict)
    metrics.update(compute_clinical_metrics(gt_dict, res_dict))
    return metrics, gts, res


def evaluate_from_csv(report_csv, report_split="test"):
    df = pd.read_csv(report_csv)
    if "split" in df.columns:
        if report_split == "all":
            df = df.copy()
        else:
            df = df[df["split"] == report_split].copy()
    
    # 智能匹配 CSV 中的真实标签列和预测列，防止因为列名不一致报错
    gt_col = next((c for c in df.columns if c in ["reference_report", "truth", "gt"]), None)
    pred_col = next((c for c in df.columns if c in ["generated_report", "pred", "res", "gen"]), None)

    if not gt_col or not pred_col:
        raise ValueError(f"Could not find valid GT/Pred columns in CSV. Available columns: {df.columns.tolist()}")

    gts = {i: [str(ref)] for i, ref in enumerate(df[gt_col].fillna("").tolist())}
    res = {i: [str(pred)] for i, pred in enumerate(df[pred_col].fillna("").tolist())}
    metrics = compute_scores(gts, res)
    metrics.update(compute_clinical_metrics(gts, res))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="config/mimic_cxr/vlci.json")
    parser.add_argument("--load_model_path", type=str, default="/home/sjx/work/project/CMCRL-main/pretrained/mimic_cxr_model_best.pth")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--sample_method", type=str, default="beam_search")
    parser.add_argument("--max_seq_length", type=int, default=80)
    parser.add_argument("--task_name", type=str, default="clinical_eval")
    parser.add_argument("--result_dir", type=str, default="results/mimic_cxr/clinical_eval")
    parser.add_argument("--report_csv", type=str, default="")
    parser.add_argument("--report_split", type=str, default="test", choices=["val", "test", "all"])
    
    # 使用 parse_known_args 替代 parse_args，防止 bash 脚本传入多余参数时引发 unrecognized arguments 崩溃
    args_cli, unknown = parser.parse_known_args()

    args = load_json_args(args_cli.c)
    args["cuda"] = args_cli.cuda
    args["seed"] = args_cli.seed
    args["batch_size"] = args_cli.batch_size
    args["beam_size"] = args_cli.beam_size
    args["sample_method"] = args_cli.sample_method
    args["max_seq_length"] = args_cli.max_seq_length
    args["task_name"] = args_cli.task_name
    args["result_dir"] = args_cli.result_dir
    args["record_dir"] = args_cli.result_dir
    args["task"] = "test"
    args["load_model_path"] = args_cli.load_model_path

    torch.cuda.set_device(int(args["cuda"]))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    setup_seed(args["seed"])
    print(args)

    os.makedirs(args["result_dir"], exist_ok=True)
    
    # 如果传入了 CSV，走纯净的离线评测流程（极速）
    if args_cli.report_csv:
        print(f"\n[*] Evaluating from offline CSV: {args_cli.report_csv}")
        metrics = evaluate_from_csv(args_cli.report_csv, report_split=args_cli.report_split)
        log = {"task_name": args["task_name"]}
        log.update({f"test_{k}": v for k, v in metrics.items()})
        print("\n" + "="*50)
        print("🏆 Final Evaluation Scores:")
        for k, v in log.items():
            if isinstance(v, float):
                print(f"{k:<15}: {v:.4f}")
            else:
                print(f"{k:<15}: {v}")
        print("="*50 + "\n")
        
        pd.DataFrame([log]).to_csv(
            os.path.join(args["result_dir"], f"{args['dataset_name']}_clinical.csv"),
            index=False,
        )
        return

    # 否则，走完整的模型推理流程（带 strict=False 兼容后续改架构）
    tokenizer = tokenizers_fn[args["tokenizer"]](args)
    print("count of tokens", len(tokenizer.token2idx))

    val_dataloader = R2DataLoader(args, tokenizer, split="val", shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split="test", shuffle=False)

    model = model_fn[args["model"]](args, tokenizer).cuda()
    state = torch.load(args["load_model_path"], map_location="cuda")
    pretrained_dict = state["state_dict"]
    incompatible = model.load_state_dict(pretrained_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("\nCheckpoint load report (strict=False):")
        print("  missing_keys:", incompatible.missing_keys)
        print("  unexpected_keys:", incompatible.unexpected_keys)
        print()

    val_metrics, _, _ = infer_split(model, val_dataloader)
    test_metrics, _, _ = infer_split(model, test_dataloader)

    log = {"task_name": args["task_name"]}
    log.update({f"val_{k}": v for k, v in val_metrics.items()})
    log.update({f"test_{k}": v for k, v in test_metrics.items()})
    
    print("\n" + "="*50)
    for k, v in log.items():
        if isinstance(v, float):
            print(f"{k:<15}: {v:.4f}")
        else:
            print(f"{k:<15}: {v}")
    print("="*50 + "\n")

    pd.DataFrame([log]).to_csv(
        os.path.join(args["result_dir"], f"{args['dataset_name']}_clinical.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
