"""
Finetune trainer for report generation
"""
import json
import copy
import pandas as pd
from .BaseTrainer import BaseTrainer
import numpy as np
import torch
import time
from trainer.PretrainTrainer import unpatchify, vis_heatmap
from utils.loss import patchify, compute_cls_loss, compute_distill_loss
import os
import sys
from tqdm import tqdm  # 引入 tqdm 进度条库
from metric.clinical import compute_clinical_metrics


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.best_score = 0.

        pretrained_dict = torch.load(args["load_model_path"], map_location='cuda')['state_dict']
        # if args["task_name"].split('_')[1] == 'en':
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                        k.split('.')[0] != 'decoder' and k.split('.')[0] != 'logit'}
        # print(pretrained_dict.keys())
        incompatible = self.model.load_state_dict(pretrained_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("Checkpoint load report:")
            print("  missing_keys:", incompatible.missing_keys)
            print("  unexpected_keys:", incompatible.unexpected_keys)
        if hasattr(self.model, "initialize_explicit_cpmi"):
            self.model.initialize_explicit_cpmi()

        self.abnormal_sample_weight = float(args.get("abnormal_sample_weight", 1.0))
        self.pathology_loss_weight = float(args.get("pathology_loss_weight", 0.0))
        self.pseudo_label_path = args.get("pseudo_label_path", "data/mimic_cxr_pseudo_labels.json")
        self.pathology_names = []
        self.pseudo_labels = {}
        if self.pathology_loss_weight > 0 or self.abnormal_sample_weight != 1.0:
            self._load_pseudo_labels()
        self.template_loss_weight = float(args.get("template_loss_weight", 0.0))
        self.template_label_path = args.get("template_label_path", "data/mimic_cxr_template_labels.json")
        self.template_names = []
        self.template_labels = {}
        if self.template_loss_weight > 0:
            self._load_template_labels()

        self.distill_loss_weight = float(args.get("distill_loss_weight", 0.0))
        self.distill_temperature = float(args.get("distill_temperature", 2.0))
        self.log_clinical_metrics_during_train = str(
            args.get("log_clinical_metrics_during_train", "0")
        ).lower() in ["1", "true", "t", "y", "yes"]
        self.teacher_model = None
        if self.distill_loss_weight > 0:
            self.teacher_model = copy.deepcopy(self.model)
            if str(args.get("teacher_disable_memory_prompt", "1")).lower() in ["1", "true", "t", "y", "yes"]:
                if hasattr(self.teacher_model, "use_memory_prompt"):
                    self.teacher_model.use_memory_prompt = False
                if hasattr(self.teacher_model, "use_phrase_memory"):
                    self.teacher_model.use_phrase_memory = False
            if str(args.get("teacher_disable_phrase_bias", "1")).lower() in ["1", "true", "t", "y", "yes"]:
                if hasattr(self.teacher_model, "use_phrase_logit_bias"):
                    self.teacher_model.use_phrase_logit_bias = False
            if str(args.get("teacher_disable_clause_memory", "1")).lower() in ["1", "true", "t", "y", "yes"]:
                if hasattr(self.teacher_model, "use_clause_memory"):
                    self.teacher_model.use_clause_memory = False
            if str(args.get("teacher_disable_explicit_cpmi", "1")).lower() in ["1", "true", "t", "y", "yes"]:
                if hasattr(self.teacher_model, "use_explicit_cpmi"):
                    self.teacher_model.use_explicit_cpmi = False
            if str(args.get("teacher_disable_counterfactual", "1")).lower() in ["1", "true", "t", "y", "yes"]:
                if hasattr(self.teacher_model, "use_counterfactual_intervention"):
                    self.teacher_model.use_counterfactual_intervention = False
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            print(
                f">>> Distillation enabled: weight={self.distill_loss_weight}, "
                f"T={self.distill_temperature}, "
                f"teacher_disable_memory_prompt={args.get('teacher_disable_memory_prompt', '1')}, "
                f"teacher_disable_phrase_bias={args.get('teacher_disable_phrase_bias', '1')}, "
                f"teacher_disable_clause_memory={args.get('teacher_disable_clause_memory', '1')}, "
                f"teacher_disable_explicit_cpmi={args.get('teacher_disable_explicit_cpmi', '1')}, "
                f"teacher_disable_counterfactual={args.get('teacher_disable_counterfactual', '1')}"
            )

    def _load_pseudo_labels(self):
        if not os.path.exists(self.pseudo_label_path):
            print(f">>> ⚠️ Pseudo label file not found: {self.pseudo_label_path}. Pathology loss disabled.")
            self.pathology_loss_weight = 0.0
            return
        with open(self.pseudo_label_path, "r") as f:
            data = json.load(f)
        self.pathology_names = data.get("pathologies", [])
        self.pseudo_labels = data.get("labels", {})
        print(
            f">>> 🧬 Pathology BCE enabled: weight={self.pathology_loss_weight}, "
            f"labels={len(self.pseudo_labels)}, classes={len(self.pathology_names)}"
        )

    def _pathology_targets(self, images_id):
        targets = []
        valid = []
        for image_id in images_id:
            item = self.pseudo_labels.get(str(image_id))
            if item is None:
                targets.append([0.0] * len(self.pathology_names))
                valid.append(0.0)
                continue
            pathologies = item.get("pathologies", {})
            targets.append([float(pathologies.get(name, 0)) for name in self.pathology_names])
            valid.append(1.0)
        return (
            torch.tensor(targets, dtype=torch.float32).cuda(),
            torch.tensor(valid, dtype=torch.float32).cuda(),
        )

    def _get_abnormal_weights(self, images_id):
        """Per-sample loss weights for abnormal reports.

        Returns a [B] tensor where abnormal samples get abnormal_sample_weight
        and normal/unknown samples get 1.0. A sample is abnormal when its pseudo
        label has at least one positive pathology other than no_finding.
        """
        if self.abnormal_sample_weight == 1.0 or not self.pseudo_labels:
            return torch.ones(len(images_id), device='cuda')
        weights = torch.ones(len(images_id), device='cuda')
        for i, image_id in enumerate(images_id):
            item = self.pseudo_labels.get(str(image_id))
            if item is None:
                continue
            pathologies = item.get("pathologies", {})
            for name, value in pathologies.items():
                if name == "no_finding":
                    continue
                if int(value) == 1:
                    weights[i] = self.abnormal_sample_weight
                    break
        return weights

    def _load_template_labels(self):
        if not os.path.exists(self.template_label_path):
            print(f">>> Template label file not found: {self.template_label_path}. Template loss disabled.")
            self.template_loss_weight = 0.0
            return
        with open(self.template_label_path, "r") as f:
            data = json.load(f)
        self.template_names = data.get("templates", [])
        self.template_labels = data.get("labels", {})
        print(
            f">>> Template BCE enabled: weight={self.template_loss_weight}, "
            f"labels={len(self.template_labels)}, classes={len(self.template_names)}"
        )

    def _template_targets(self, images_id):
        targets = []
        valid = []
        for image_id in images_id:
            item = self.template_labels.get(str(image_id))
            if item is None:
                targets.append([0.0] * len(self.template_names))
                valid.append(0.0)
                continue
            templates = item.get("templates", {})
            targets.append([float(templates.get(name, 0)) for name in self.template_names])
            valid.append(1.0)
        return (
            torch.tensor(targets, dtype=torch.float32).cuda(),
            torch.tensor(valid, dtype=torch.float32).cuda(),
        )

    def _eval_and_save(self, epoch, current_step_info=""):
        """将原本在 Epoch 末尾的验证逻辑抽离成独立函数，方便在子轮次中调用"""
        log = {}
        # 1. 跑验证集
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            val_pbar = tqdm(self.val_dataloader, desc=f"Val {current_step_info}", leave=False)
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(val_pbar):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
            tp, lp = count_p(p[1:])
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            if self.log_clinical_metrics_during_train:
                val_met.update(
                    compute_clinical_metrics(
                        {i: [gt] for i, gt in enumerate(val_gts)},
                        {i: [re] for i, re in enumerate(val_res)},
                    )
                )
            for k, v in val_met.items():
                self.monitor.logkv(key='val_' + k, val=v)
            val_met['p'] = lp
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        # 2. 跑测试集
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            test_pbar = tqdm(self.test_dataloader, desc=f"Test {current_step_info}", leave=False)
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_pbar):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])
            tp, lp = count_p(p[1:])
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            if self.log_clinical_metrics_during_train:
                test_met.update(
                    compute_clinical_metrics(
                        {i: [gt] for i, gt in enumerate(test_gts)},
                        {i: [re] for i, re in enumerate(test_res)},
                    )
                )

            for k, v in test_met.items():
                self.monitor.logkv(key='test_' + k, val=v)
            test_met['p'] = lp
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        # ================= 🌟 强行在终端显示分数 (防覆盖机制) =================
        metric_name = self.args["monitor_metric"]
        val_metric_key = 'val_' + metric_name
        test_metric_key = 'test_' + metric_name
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"📊 Epoch {epoch} {current_step_info} 验证成绩单:")
        if val_metric_key in log:
            tqdm.write(f"🏆 Val 选择指标 {val_metric_key}: {log[val_metric_key]:.6f}")
        if test_metric_key in log:
            tqdm.write(f"🏆 Test 保存指标 {test_metric_key}: {log[test_metric_key]:.6f}")
        val_line = (
            f"📌 VAL  | BLEU_4: {log.get('val_BLEU_4', 0):.4f} "
            f"| CIDEr: {log.get('val_CIDEr', 0):.4f} "
            f"| ROUGE_L: {log.get('val_ROUGE_L', 0):.4f}"
        )
        test_line = (
            f"📌 TEST | BLEU_4: {log.get('test_BLEU_4', 0):.4f} "
            f"| CIDEr: {log.get('test_CIDEr', 0):.4f} "
            f"| ROUGE_L: {log.get('test_ROUGE_L', 0):.4f}"
        )
        if self.log_clinical_metrics_during_train:
            val_line += f" | CE_F1: {log.get('val_CE_F1', 0):.4f}"
            test_line += f" | CE_F1: {log.get('test_CE_F1', 0):.4f}"
        tqdm.write(val_line)
        tqdm.write(test_line)
        tqdm.write(f"{'='*50}\n")
        sys.stdout.flush()  # 强制刷新缓冲区，确保在使用 nohup 时立刻写入日志文件
        # =====================================================================

        # 3. 判断是否破纪录：val 用于规范选权重，test 用于保存用户指定的 model_best.pth
        if self.mnt_mode != 'off':
            self._record_best(log)

            improved_val = False
            if self.mnt_metric in log:
                improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            if improved_val:
                tqdm.write(f"🌟 [Val Breakthrough] {self.mnt_metric} improved from {self.mnt_best:.6f} to {log[self.mnt_metric]:.6f}")
                self.mnt_best = log[self.mnt_metric]
                self.best_epoch = f"{epoch} {current_step_info}"
                self._save_checkpoint(epoch, save_best=True, best_filename='model_best_val.pth',
                                      monitor_best=self.mnt_best, best_epoch=self.best_epoch)

            improved_test = False
            if self.mnt_metric_test in log:
                improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best_test) or \
                                (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best_test)
            if improved_test:
                tqdm.write(f"🌟 [Test Breakthrough] {self.mnt_metric_test} improved from {self.mnt_best_test:.6f} to {log[self.mnt_metric_test]:.6f}")
                self.mnt_best_test = log[self.mnt_metric_test]
                self.best_epoch_test = f"{epoch} {current_step_info}"
                self._save_checkpoint(epoch, save_best=True, best_filename='model_best.pth',
                                      monitor_best=self.mnt_best_test, best_epoch=self.best_epoch_test)

            tqdm.write(
                f"✅ 当前 model_best_val.pth: {getattr(self, 'best_epoch', 'N/A')} | {self.mnt_metric} = {self.mnt_best:.6f}"
            )
            tqdm.write(
                f"✅ 当前 model_best.pth: {getattr(self, 'best_epoch_test', 'N/A')} | {self.mnt_metric_test} = {self.mnt_best_test:.6f}"
            )
        return log


    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.train()
        start_time = time.time()
        
        # === 🌟 读取外部传进来的验证间隔 ===
        val_intervals = self.args.get("val_intervals", 1)
        total_batches = len(self.train_dataloader)
        check_step_size = max(1, total_batches // val_intervals)

        train_pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", total=total_batches)

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(train_pbar):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode='train')
            sample_weights = self._get_abnormal_weights(images_id)
            if self.abnormal_sample_weight != 1.0 and sample_weights.max() > 1.0:
                log_probs = output[0]
                seq_len = log_probs.size(1)
                target = reports_ids[:, 1:seq_len + 1]
                mask = reports_masks[:, 1:seq_len + 1]
                token_nll = -log_probs.gather(2, target.unsqueeze(2).long()).squeeze(2) * mask
                per_sample_loss = token_nll.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                nll_loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1)
            else:
                nll_loss = self.criterion(output, reports_ids, reports_masks)
            loss = nll_loss
            pathology_loss_value = None
            template_loss_value = None
            distill_loss_value = None
            output_dict = output[2] if isinstance(output, list) and len(output) > 2 else {}
            if self.distill_loss_weight > 0 and self.teacher_model is not None:
                student_logits = output_dict.get("raw_logits")
                if student_logits is not None:
                    with torch.no_grad():
                        teacher_output = self.teacher_model(images, reports_ids, mode='train')
                    teacher_dict = teacher_output[2] if isinstance(teacher_output, list) and len(teacher_output) > 2 else {}
                    teacher_logits = teacher_dict.get("raw_logits")
                    if teacher_logits is not None:
                        distill_loss = compute_distill_loss(
                            student_logits,
                            teacher_logits,
                            reports_masks[:, 1:],
                            temperature=self.distill_temperature,
                        )
                        loss = loss + self.distill_loss_weight * distill_loss
                        distill_loss_value = distill_loss.item()
            if self.pathology_loss_weight > 0:
                pathology_logits = output_dict.get("pathology_logits")
                if pathology_logits is not None:
                    pathology_targets, pathology_valid = self._pathology_targets(images_id)
                    pathology_loss = compute_cls_loss(pathology_logits, pathology_targets, pathology_valid)
                    loss = loss + self.pathology_loss_weight * pathology_loss
                    pathology_loss_value = pathology_loss.item()
            if self.template_loss_weight > 0:
                template_logits = output_dict.get("template_logits")
                if template_logits is not None:
                    template_targets, template_valid = self._template_targets(images_id)
                    template_loss = compute_cls_loss(template_logits, template_targets, template_valid)
                    loss = loss + self.template_loss_weight * template_loss
                    template_loss_value = template_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            # 更新进度条信息
            current_mean_loss = train_loss / (batch_idx + 1)
            postfix = {"loss": f"{loss.item():.3f}", "mean loss": f"{current_mean_loss:.3f}"}
            if pathology_loss_value is not None:
                postfix["path loss"] = f"{pathology_loss_value:.3f}"
            if template_loss_value is not None:
                postfix["temp loss"] = f"{template_loss_value:.3f}"
            if distill_loss_value is not None:
                postfix["kd loss"] = f"{distill_loss_value:.3f}"
            train_pbar.set_postfix(postfix)

            if self.args["lr_scheduler"] != 'StepLR':
                self.lr_scheduler.step()
                
            # === 🌟 进度拦截器：子轮次验证逻辑 ===
            current_step = batch_idx + 1
            is_interval_step = current_step % check_step_size == 0
            is_last_interval = current_step + check_step_size >= total_batches
            if val_intervals > 1 and is_interval_step and not is_last_interval:
                progress_pct = int((current_step / total_batches) * 100)
                tqdm.write(f"\n>>> 🚀 进度达到 {progress_pct}%，执行子轮次验证 (Intra-Epoch Validation)...")
                # 调用独立出去的验证函数
                self._eval_and_save(epoch, current_step_info=f"({progress_pct}%)")
                # 验证完切回训练模式
                self.model.train()
                tqdm.write(f">>> 🔄 子轮次验证结束，继续训练 Epoch {epoch}...")

        if self.args["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        tqdm.write(f"\n\tEpoch {epoch}\tmean_loss: {log['train_loss']:.4f}\ttime: {time.time() - start_time:.4f}s")

        # Epoch 结束时的全量验证
        eval_log = self._eval_and_save(epoch, current_step_info="(100%)")
        log.update(eval_log)

        # 打印每一轮后的指标
        tqdm.write(f"\n--- Epoch {epoch} Metrics ---")
        for k, v in log.items():
            if isinstance(v, float):
                tqdm.write(f"  {k:15s}: {v:.4f}")
            else:
                tqdm.write(f"  {k:15s}: {v}")
        tqdm.write("-----------------------------\n")

        if self.args['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.monitor.dumpkv(epoch)
        return log

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p, img_ids = [], [], [], []
            hvs, hvs_id = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                # print(output[0])
                # print(reports[0])
                # marked_img = mark_local(images, part_inx)
                # vis_img(marked_img[0], title='high attn')
                # vis_img(torch.cat([images[0, 0], images[0, 1]], dim=-1), title='ori')

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            # test_res = torch.load("results/mimic_cxr/DMIRG/DMIRG/118_report_100.npy")
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            
            save_report(test_res, test_gts, img_ids, os.path.join(self.checkpoint_dir, 'report.csv'))
            print(test_met)
            # print(lp)

    def local_feature(self, idx):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p, img_ids = [], [], [], []
            hvs, hvs_id = [], []
            p = torch.zeros([1, self.args.max_seq_length]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if idx in images_id:
                    _, m = self.model(images, mode='extract')
                    attn = m['attn']

                    encoder_attn = encoder_heatmap(images, attn)
                    for i in range(encoder_attn.size(0)):
                        if idx == images_id[i]:
                            vis_heatmap(images[i], encoder_attn[i], title='encoder_attn')
                            print(images_id['i'])

    def keyword_feature(self, idx):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if idx in images_id:
                    key_id = self.model.tokenizer.token2idx['pleural']
                    target = torch.tensor([0, key_id, 0]).to(images).long()
                    target = target.unsqueeze(0).repeat(images.size(0), 1)
                    attn = self.model(images, target, mode='keyword')

                    encoder_attn = decoder_heatmap(images, attn)
                    for i in range(encoder_attn.size(0)):
                        if idx == images_id[i]:
                            vis_heatmap(images[i], encoder_attn[i], title='encoder_attn')
                            print(images_id[i])

    def extract(self):
        self.model.eval()
        with torch.no_grad():
            hvs, hvs_id = [], []
            p = torch.zeros([1, self.args.max_seq_length]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                h, _ = self.model(images, mode='extract')
                _hvs, _hvs_id = self.select_feature(h, images_id)
                hvs.extend(_hvs)
                hvs_id.extend(_hvs_id)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            # hvs_id = hvs_id.
            torch.save({'id': hvs_id, 'feature': hvs}, 'results/mimic_cxr/DMIRG/DMIRG/118_feature.npy')

    def select_feature(self, h, images_id):
        hv_list = []
        hv_ids = []
        h = h.detach().cpu().numpy()
        for i in range(h.shape[0]):
            if images_id[i] in self.for_tsne:
                hv_list.append(h[i])
                hv_ids.append(images_id[i])
        return hv_list, hv_ids


def count_p(p):
    t = torch.unique(p, dim=0)
    l = t.size(0)
    return t, l


def mark_local(img, inx):
    """
    img [B, [N], 3, H, W]
    output [B, 3, H, W]
    """
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        for i in range(B):
            mask[i, inx[i, :]] = 1

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)

    else:
        B, C, H, W = img.size()
        img = patchify(img, 16)
        mask = torch.zeros([B, 14 * 14]).to(img.device)
        for i in range(B):
            mask[i, inx[i, :]] = 1
        output = unpatchify(img, mask)

    return output


def encoder_heatmap(img, attn):
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)

    else:
        # B, C, H, W = img.size()
        mask = torch.zeros_like(img).to(img)
        mask = patchify(mask, 16)  # B, L, N

        length = len(attn)
        last_map = attn[0]
        for i in range(1, length):
            last_map = torch.matmul(attn[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        attn_map = last_map.mean(dim=1).unsqueeze(-1)
        mask += attn_map
        output = unpatchify(mask)

    return output


def decoder_heatmap(img, attn):
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)

    else:
        attn = attn[0] + attn[1] + attn[2]
        # B, C, H, W = img.size()
        mask = torch.zeros_like(img).to(img)
        mask = patchify(mask, 16)  # B, L, N

        last_map = attn[:, :, 1, 1:]
        attn_map = last_map.mean(dim=1).unsqueeze(-1)
        mask += attn_map
        output = unpatchify(mask)

    return output


def save_report(inference, reference, ids, output_dir):
    df = pd.DataFrame({'Report Impression': inference})
    df.to_csv(output_dir)
