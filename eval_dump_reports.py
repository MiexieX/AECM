#!/usr/bin/env python3
"""Dump generated reports and ground-truth reports to CSV.

This script runs one inference pass and stores paired reports so that
clinical metrics can be computed later without regenerating text.
"""

import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_split(model, dataloader, split):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
            images, reports_ids = images.cuda(), reports_ids.cuda()
            output = model(images, mode="sample")
            generated = model.tokenizer.decode_batch(output.cpu().numpy())
            reference = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            for image_id, ref, pred in zip(images_id, reference, generated):
                rows.append({
                    "split": split,
                    "image_id": image_id,
                    "reference_report": ref,
                    "generated_report": pred,
                })
            print(f"\r{split} Processing: [{int((batch_idx + 1) / len(dataloader) * 100)}%]", end="", flush=True)
    print()
    return rows


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
    parser.add_argument("--split", type=str, default="both", choices=["both", "val", "test"])
    parser.add_argument("--task_name", type=str, default="report_dump")
    parser.add_argument("--result_dir", type=str, default="results/mimic_cxr/report_dump")
    parser.add_argument("--model", type=str, default="", help="override model name from config")
    parser.add_argument('--suppress_UNK', type=int, default=0, help='suppress unknown tokens')
    parser.add_argument("--use_anatomical_adapter", type=int, default=None)
    parser.add_argument("--use_memory_prompt", type=int, default=None)
    parser.add_argument("--use_counterfactual_intervention", type=int, default=None)
    parser.add_argument("--ccd_gamma", type=float, default=None)
    parser.add_argument("--ccd_use_confidence_gate", type=int, default=None)
    parser.add_argument("--ccli_anomaly_threshold", type=float, default=None)
    parser.add_argument("--ccli_anomaly_temperature", type=float, default=None)
    parser.add_argument("--ccli_max_alpha", type=float, default=None)
    parser.add_argument("--use_confidence_gated_pathology_prompt", type=int, default=None)
    parser.add_argument("--pathology_gate_threshold", type=float, default=None)
    parser.add_argument("--pathology_gate_temperature", type=float, default=None)
    parser.add_argument("--use_normal_aware_suppression", type=int, default=None)
    parser.add_argument("--normal_suppression_threshold", type=float, default=None)
    parser.add_argument("--normal_suppression_temperature", type=float, default=None)
    parser.add_argument(
        '--length_penalty',
        type=str,
        default="",
        help='length penalty for beam search, e.g. "", "wu_0.7", "wu_1.0", "avg_1.0"',
    )
    args_cli = parser.parse_args()

    args = load_json_args(args_cli.c)
    args["cuda"] = args_cli.cuda
    args["seed"] = args_cli.seed
    args["batch_size"] = args_cli.batch_size
    args["beam_size"] = args_cli.beam_size
    args["sample_method"] = args_cli.sample_method
    args["max_seq_length"] = args_cli.max_seq_length
    args["split"] = args_cli.split
    args["suppress_UNK"] = args_cli.suppress_UNK
    args["length_penalty"] = args_cli.length_penalty
    args["task_name"] = args_cli.task_name
    args["result_dir"] = args_cli.result_dir
    args["record_dir"] = args_cli.result_dir
    args["task"] = "test"
    args["load_model_path"] = args_cli.load_model_path
    if args_cli.model:
        args["model"] = args_cli.model
    if args_cli.use_anatomical_adapter is not None:
        args["use_anatomical_adapter"] = args_cli.use_anatomical_adapter
    if args_cli.use_memory_prompt is not None:
        args["use_memory_prompt"] = args_cli.use_memory_prompt
    if args_cli.use_counterfactual_intervention is not None:
        args["use_counterfactual_intervention"] = args_cli.use_counterfactual_intervention
    if args_cli.ccd_gamma is not None:
        args["ccd_gamma"] = args_cli.ccd_gamma
    if args_cli.ccd_use_confidence_gate is not None:
        args["ccd_use_confidence_gate"] = args_cli.ccd_use_confidence_gate
    if args_cli.ccli_anomaly_threshold is not None:
        args["ccli_anomaly_threshold"] = args_cli.ccli_anomaly_threshold
    if args_cli.ccli_anomaly_temperature is not None:
        args["ccli_anomaly_temperature"] = args_cli.ccli_anomaly_temperature
    if args_cli.ccli_max_alpha is not None:
        args["ccli_max_alpha"] = args_cli.ccli_max_alpha
    if args_cli.use_confidence_gated_pathology_prompt is not None:
        args["use_confidence_gated_pathology_prompt"] = args_cli.use_confidence_gated_pathology_prompt
    if args_cli.pathology_gate_threshold is not None:
        args["pathology_gate_threshold"] = args_cli.pathology_gate_threshold
    if args_cli.pathology_gate_temperature is not None:
        args["pathology_gate_temperature"] = args_cli.pathology_gate_temperature
    if args_cli.use_normal_aware_suppression is not None:
        args["use_normal_aware_suppression"] = args_cli.use_normal_aware_suppression
    if args_cli.normal_suppression_threshold is not None:
        args["normal_suppression_threshold"] = args_cli.normal_suppression_threshold
    if args_cli.normal_suppression_temperature is not None:
        args["normal_suppression_temperature"] = args_cli.normal_suppression_temperature

    torch.cuda.set_device(int(args["cuda"]))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    setup_seed(args["seed"])
    print(args)

    tokenizer = tokenizers_fn[args["tokenizer"]](args)
    print("count of tokens", len(tokenizer.token2idx))

    val_dataloader = None
    test_dataloader = None
    if args_cli.split in ("both", "val"):
        val_dataloader = R2DataLoader(args, tokenizer, split="val", shuffle=False)
    if args_cli.split in ("both", "test"):
        test_dataloader = R2DataLoader(args, tokenizer, split="test", shuffle=False)

    model = model_fn[args["model"]](args, tokenizer).cuda()
    state = torch.load(args["load_model_path"], map_location="cuda")
    pretrained_dict = state["state_dict"]
    incompatible = model.load_state_dict(pretrained_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Checkpoint load report:")
        print("  missing_keys:", incompatible.missing_keys)
        print("  unexpected_keys:", incompatible.unexpected_keys)

    rows = []
    if val_dataloader is not None:
        rows.extend(run_split(model, val_dataloader, "val"))
    if test_dataloader is not None:
        rows.extend(run_split(model, test_dataloader, "test"))

    os.makedirs(args["result_dir"], exist_ok=True)
    out_path = os.path.join(args["result_dir"], f"{args['dataset_name']}_reports.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved report pairs to: {out_path}")


if __name__ == "__main__":
    main()
