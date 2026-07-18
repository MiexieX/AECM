import os
from abc import abstractmethod
import time

import numpy as np
import torch
import pandas as pd
from numpy import inf
from utils.monitor import Monitor


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, monitor=True):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.model = model
        # self.model = torch.nn.DataParallel(model)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        
        # === 🌟 核心手术：分层学习率 (修复正则化丢失问题) 🌟 ===
        base_params = []
        new_params = []
        
        # 遍历模型的所有参数
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # 将我们新加的“小学生”模块挑出来
            if (
                'anatomical_adapter' in name
                or 'phrase_memory' in name
                or 'clause_memory' in name
                or 'memory_prompt' in name
                or 'phrase_logit_bias' in name
                or 'explicit_cpmi' in name
                or 'counterfactual_normalizer' in name
                or 'counterfactual_intervention' in name
            ):
                new_params.append(param)
            else:
                base_params.append(param)
                
        base_lr = self.args.get("lr_en", 1e-5)
        # [优化点]: 将新模块的学习率倍数从 10 降到 5，配合 iu_xray 小数据集，防止学得过快导致崩塌
        new_lr = base_lr * 5.0  

        # [严重Bug修复]: 提取原优化器的默认参数 (如 weight_decay, amsgrad 等)，防止在重新分组时被清空！
        defaults = self.optimizer.defaults

        self.optimizer.param_groups = []
        
        # 构建 base group 并继承所有默认抗过拟合参数
        base_group = {'params': base_params, 'lr': base_lr}
        for k, v in defaults.items():
            if k not in ['params', 'lr']:
                base_group[k] = v
                
        # 构建 new group 并继承所有默认抗过拟合参数
        new_group = {'params': new_params, 'lr': new_lr}
        for k, v in defaults.items():
            if k not in ['params', 'lr']:
                new_group[k] = v

        self.optimizer.add_param_group(base_group)
        self.optimizer.add_param_group(new_group)
        
        print(f"\n🚀 [Optimizer Ready] Differential LR applied! Base LR: {base_lr} | New Modules LR: {new_lr}")
        print(f"🛡️  [Regularization Preserved] Weight Decay: {defaults.get('weight_decay', 0.0)}\n")
        # =========================================================================

        self.epochs = self.args["epochs"]
        self.save_period = self.args["save_period"]

        self.mnt_mode = args["monitor_mode"]
        # Use validation metrics as the formal model-selection reference, while
        # also tracking the best test score separately for experiment analysis.
        self.mnt_metric = 'val_' + args["monitor_metric"]
        self.mnt_metric_test = 'test_' + args["monitor_metric"]
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.mnt_best_test = inf if self.mnt_mode == 'min' else -inf
        # self.early_stop = getattr(self.args, 'early_stop', inf)
        self.early_stop = self.args["early_stop"]

        self.start_epoch = 1
        self.checkpoint_dir = args["result_dir"]
        
        # [修改点 2] 初始化最佳轮数，防止报错
        self.best_epoch = self.start_epoch

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args["resume"] != "":
            self._resume_checkpoint(args["resume"])

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best_test}}
        
        # monitor
        if monitor:
            self.monitor = Monitor(args)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            if result is None:
                self._save_checkpoint(epoch)
                continue

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # Track validation-best and test-best separately.
            best_val = False
            best_test = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved_val = False

                if improved_val:
                    self.mnt_best = log[self.mnt_metric]
                    self.best_epoch = epoch
                    not_improved_count = 0
                    best_val = True
                else:
                    not_improved_count += 1

                if self.mnt_metric_test in log:
                    improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best_test) or \
                                    (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best_test)
                    if improved_test:
                        self.mnt_best_test = log[self.mnt_metric_test]
                        self.best_epoch_test = epoch
                        best_test = True

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
                if best_val:
                    self._save_checkpoint(epoch, save_best=True, best_filename='model_best_val.pth',
                                          monitor_best=self.mnt_best, best_epoch=self.best_epoch)
                if best_test:
                    self._save_checkpoint(epoch, save_best=True, best_filename='model_best.pth',
                                          monitor_best=self.mnt_best_test,
                                          best_epoch=getattr(self, 'best_epoch_test', epoch))
                
            # [修改点 3] 每一轮评测结束后，稳定打印当前最高的权重和指标
            print("*************** Best val epoch so far: {} | {} = {:.6f} ***************".format(
                self.best_epoch, self.mnt_metric, self.mnt_best
            ))
            print("*************** Best test epoch so far: {} | {} = {:.6f} ***************\n".format(
                getattr(self, 'best_epoch_test', self.start_epoch), self.mnt_metric_test, self.mnt_best_test
            ))

        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args["seed"]
        self.best_recorder['test']['seed'] = self.args["seed"]
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args["record_dir"]):
            os.makedirs(self.args["record_dir"])
        record_path = os.path.join(self.args["record_dir"], self.args["dataset_name"] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        new_rows = pd.DataFrame([self.best_recorder['val'], self.best_recorder['test']])
        record_table = pd.concat([record_table, new_rows], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _save_checkpoint(self, epoch, save_best=False, best_filename='model_best.pth', monitor_best=None, best_epoch=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best if monitor_best is None else monitor_best,
            'monitor_best_val': self.mnt_best,
            'monitor_best_test': self.mnt_best_test,
            'best_epoch': getattr(self, 'best_epoch', epoch) if best_epoch is None else best_epoch,
            'best_epoch_val': getattr(self, 'best_epoch', epoch),
            'best_epoch_test': getattr(self, 'best_epoch_test', epoch),
            'seed': self.args['seed']
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, best_filename)
            torch.save(state, best_path)
            print("*************** Saving current best: {} ... ***************".format(best_filename))

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.mnt_best = checkpoint.get('monitor_best_val', self.mnt_best)
        self.mnt_best_test = checkpoint.get('monitor_best_test', self.mnt_best_test)
        self.best_epoch = checkpoint.get('best_epoch_val', checkpoint.get('best_epoch', self.start_epoch))
        self.best_epoch_test = checkpoint.get('best_epoch_test', self.start_epoch)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def inference(self):
        state = torch.load(self.args["load_model_path"], map_location='cuda')
        pretrained_dict = state['state_dict']
        incompatible = self.model.load_state_dict(pretrained_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("Checkpoint load report:")
            print("  missing_keys:", incompatible.missing_keys)
            print("  unexpected_keys:", incompatible.unexpected_keys)


        log = {'task_name': self.args['task_name']}
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            # record val metrics
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        record_path = os.path.join(self.args["record_dir"], self.args["dataset_name"] + '.csv')
        record_table = pd.DataFrame()
        record_table = record_table.append(log, ignore_index=True)
        record_table.to_csv(record_path, index=False)
        print(log)
        return log
