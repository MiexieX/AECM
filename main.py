import json
import random
from collections import OrderedDict

import numpy as np
import torch
import argparse
from metric.metrics import compute_scores
# ⚠️ 恢复了 build_optimizer 的导入
from utils import R2DataLoader, tokenizers_fn, build_optimizer, build_lr_scheduler, loss_fn
from trainer import *
import os
from models import model_fn
import warnings

warnings.filterwarnings('ignore')


def load_json_args(path):
    json_str = ''
    with open(path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'  #
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
    # --- 新增以下两行，关闭 Ada 显卡的 TF32 精度截断 ---
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()
    parse.add_argument('--c', type=str, default='config/iu_xray/vlci.json',
                       help='json file of config')
    
    # 巧妙利用 parse_known_args 捕获所有命令行里未预先定义的参数
    opt, unknown = parse.parse_known_args()
    args = load_json_args(opt.c)
    
    # [新增] 预设 val_intervals 默认值为 1，以防 JSON 里没有
    if "val_intervals" not in args:
        args["val_intervals"] = 1
    
    # --- 动态覆盖 JSON 配置的核心逻辑 ---
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]  # 提取参数名，去掉 '--'
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                val = unknown[i+1]
                # 核心：如果这个 key 在 JSON 中存在，自动转换并保持原本的数据类型 (int, float, bool)
                if key in args and args[key] is not None:
                    orig_type = type(args[key])
                    if orig_type == bool:
                        args[key] = str(val).lower() in ['true', '1', 't', 'y', 'yes']
                    else:
                        args[key] = orig_type(val)
                else:
                    # [新增] 对 val_intervals 进行强制类型转换
                    if key == "val_intervals":
                        args[key] = int(val)
                    else:
                        args[key] = val
                i += 2
            else:
                args[key] = True  # 如果只有 flag 没有值，默认设为 True
                i += 1
        else:
            i += 1

    # 保持记录和结果路径同步
    if "result_dir" in args:
        args["record_dir"] = args["result_dir"]

    torch.cuda.set_device(int(args["cuda"]))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # -------------------------------
    # fix random seeds
    # -------------------------------
    if args["seed"] == -1:
        args["seed"] = np.random.randint(0, 23333)
    print(args)
    setup_seed(args["seed"])
    # -------------------------------
    # create tokenizer
    # -------------------------------
    tokenizer = tokenizers_fn[args['tokenizer']](args)
    print('count of tokens', len(tokenizer.token2idx))
    # -------------------------------
    # create data loader
    # -------------------------------
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    # -------------------------------
    # build model architecture
    # -------------------------------
    model = model_fn[args["model"]](args, tokenizer)

    if str(args.get("freeze_innovations", "0")).lower() in ["1", "true", "t", "y", "yes"]:
        frozen = []
        for name, param in model.named_parameters():
            if any(tag in name for tag in [
                "anatomical_adapter",
                "phrase_memory",
                "clause_memory",
                "phrase_logit_bias",
                "explicit_cpmi",
                "memory_prompt",
                "counterfactual_normalizer",
                "counterfactual_intervention",
            ]):
                param.requires_grad = False
                frozen.append(name)
        print(f">>> 🧊 Frozen innovation parameters: {len(frozen)} tensors")

    if str(args.get("train_only_phrase_bias", "0")).lower() in ["1", "true", "t", "y", "yes"]:
        trainable = []
        frozen = []
        phrase_bias_tags = [
            "phrase_logit_bias",
        ]
        for name, param in model.named_parameters():
            if any(tag in name for tag in phrase_bias_tags):
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
                frozen.append(name)
        print(f">>> 🧊 Frozen non-phrase-bias parameters: {len(frozen)} tensors")
        print(f">>> 🔬 Trainable phrase-bias parameters: {len(trainable)} tensors")

    elif str(args.get("train_only_explicit_cpmi", "0")).lower() in ["1", "true", "t", "y", "yes"]:
        trainable = []
        frozen = []
        cpmi_tags = [
            "explicit_cpmi",
        ]
        for name, param in model.named_parameters():
            if any(tag in name for tag in cpmi_tags):
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
                frozen.append(name)
        print(f">>> 🧊 Frozen non-Explicit-CPMI parameters: {len(frozen)} tensors")
        print(f">>> 🔬 Trainable Explicit-CPMI parameters: {len(trainable)} tensors")

    elif str(args.get("train_only_counterfactual", "0")).lower() in ["1", "true", "t", "y", "yes"]:
        trainable = []
        frozen = []
        counterfactual_tags = [
            "counterfactual_normalizer",
            "counterfactual_intervention",
        ]
        for name, param in model.named_parameters():
            if any(tag in name for tag in counterfactual_tags):
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
                frozen.append(name)
        print(f">>> 🧊 Frozen non-counterfactual parameters: {len(frozen)} tensors")
        print(f">>> 🔬 Trainable counterfactual parameters: {len(trainable)} tensors")

    elif str(args.get("train_only_innovations", "0")).lower() in ["1", "true", "t", "y", "yes"]:
        trainable = []
        frozen = []
        innovation_tags = [
            "anatomical_adapter",
            "phrase_memory",
            "clause_memory",
            "phrase_logit_bias",
            "explicit_cpmi",
            "memory_prompt",
            "counterfactual_normalizer",
            "counterfactual_intervention",
        ]
        for name, param in model.named_parameters():
            if any(tag in name for tag in innovation_tags):
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
                frozen.append(name)
        print(f">>> 🧊 Frozen backbone parameters: {len(frozen)} tensors")
        print(f">>> 🔬 Trainable innovation parameters: {len(trainable)} tensors")

    # ⚠️ 删除了重复加载权重的逻辑，统一交给 FinetuneTrainer 处理

    model = model.cuda()
    # -------------------------------
    # get function handles of loss and metrics
    # -------------------------------
    criterion = loss_fn[args["loss_fn"]]
    metrics = compute_scores
    
    # -------------------------------
    # build optimizer, learning rate scheduler
    # -------------------------------
    # ⚠️ 删除了重复的参数分组逻辑，恢复默认构造，由 BaseTrainer 接管分组
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer, len(train_dataloader))

    # -------------------------------
    # build trainer and start to train
    # -------------------------------
    kwarg = {"model": model, "criterion": criterion, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
             "lr_scheduler": lr_scheduler, "train_dataloader": train_dataloader, "val_dataloader": val_dataloader,
             "test_dataloader": test_dataloader}

    if args["task"] == 'finetune':
        trainer = FTrainer(**kwarg)
        trainer.train()
    elif args["task"] == 'pretrain':
        trainer = PTrainer(**kwarg)
        trainer.train()
    else:
        trainer = Trainer(**kwarg)
        trainer.inference()


if __name__ == '__main__':
    main()
