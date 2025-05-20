import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
from datetime import timedelta
import sys

from data_utils import *
from utils import *
from data_utils import * 
from config import *


import json
class Trainer():
    def __init__(self, args, model, dataset):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        
        self.dataloader = BertDataloader(args, dataset)
        self.log_save_id = create_log_id(self.args.save_dir)
        
        
    def train(self):
        if self.args.is_distributed:
            dist.init_process_group("gloo", timeout=timedelta(seconds=3000))
            rank = dist.get_rank()
            is_main_process = (rank == 0)
            self.model = DDP(self.model)        
        else:
            is_main_process = True

        self.train_loader = self.dataloader.get_train_loader()
        

        # Logging chỉ ở process chính
        if is_main_process:
            logging_config(folder=self.args.save_dir, name=f'log{self.log_save_id}', no_console=False)
            logging.info(self.args)
            logging.info(self.model)

        self.model.train()
        best_recall = 0
        best_epoch = 0
        recall_list = []
        min_valid_batch_size = 16
        should_stop = False

        for epoch in range(1, self.num_epochs + 1):
            if self.args.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            tqdm_dataloader = tqdm(self.train_loader) if is_main_process else self.train_loader

            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(tqdm_dataloader):
                try:
                    batch = [x.to(self.device) for x in batch]
                except AttributeError:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                loss = self.calculate_loss(batch)

                if torch.isnan(loss).any():
                    if batch[0].size(0) >= min_valid_batch_size:
                        if is_main_process:
                            logging.error(f'ERROR: Epoch {epoch}, batch {batch_idx + 1} Loss is nan.')
                        if self.args.is_distributed:
                            dist.destroy_process_group()
                        sys.exit()
                    else:
                        continue

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1


            # Tính average_loss: nếu phân tán thì cần reduce từ các GPU
            average_loss_tensor = torch.tensor([total_loss, num_batches], dtype=torch.float32, device=self.device)

            if self.args.is_distributed:
                dist.all_reduce(average_loss_tensor, op=dist.ReduceOp.SUM)

            total_loss = average_loss_tensor[0].item()
            num_batches = average_loss_tensor[1].item()
            average_loss = total_loss / max(num_batches, 1)

            if is_main_process:
                logging.info(f'Epoch {epoch:04d} | Average Loss: {average_loss:.4f}')

            dist.barrier()
                
            if (epoch % self.args.evaluate_every == 0 or epoch == self.args.num_epochs) and is_main_process:
                if self.args.is_distributed:
                    self.model.module.eval()
                else:
                    self.model.eval()

                all_metrics = {
                    "Recall@1": [],
                    "Recall@5": [],
                    "Recall@10": [],
                    "NDCG@1": [],
                    "NDCG@5": [],
                    "NDCG@10": []
                }

                with torch.no_grad():
                    self.val_loader = self.dataloader.get_val_loader()
                    tqdm_dataloader = tqdm(self.val_loader)
                    for batch_idx, batch in enumerate(tqdm_dataloader):
                        batch_size = batch[0].size(0)
                        batch = [x.to(self.device) for x in batch]

                        metrics = self.calculate_metrics(batch)
                        if batch_size >= min_valid_batch_size:
                            for key in all_metrics:
                                all_metrics[key].append(metrics[key])

                avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
                recall_list.append(avg_metrics[self.args.best_metric])

                logging.info('Val: Epoch {:04d} | Recall [{:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f} {:.4f}]'.format(
                            epoch, avg_metrics['Recall@1'],  avg_metrics['Recall@5'], avg_metrics['Recall@10'],  avg_metrics['NDCG@1'], avg_metrics['NDCG@5'], avg_metrics['NDCG@10']))

                best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)
                

                if recall_list[-1] == best_recall:
                    if self.args.is_distributed:
                        model_state_dict = self.model.module.state_dict()
                    else:
                        model_state_dict = self.model.state_dict()

                    model_save_path = os.path.join(self.args.save_dir, 'model', 'best_model.pth')
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save({'model_state_dict': model_state_dict, 'epoch': epoch},
                               model_save_path)
                    logging.info(f'Save model at epoch {epoch:04d}!')
                    best_epoch = epoch
            if should_stop:
                    break    
        dist.destroy_process_group()

    def test(self):
        logging_config(folder=self.args.save_dir, name=f'log{self.log_save_id}', no_console=False)
        print('Test best model with test set!')
        best_model = torch.load(os.path.join(self.args.save_dir, 'model', 'best_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)

        self.model.eval()
        all_metrics = {
            "Recall@1": [],
            "Recall@5": [],
            "Recall@10": [],
            "NDCG@1": [],
            "NDCG@5": [],
            "NDCG@10": []
        }
        
        min_test_batch_size = 16
        
        self.test_loader = self.dataloader.get_test_loader()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch_size = batch[0].size(0)
                batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)
                if batch_size >= min_test_batch_size:
                    for key in all_metrics:
                        all_metrics[key].append(metrics[key])

        avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
        logging.info('Test: Recall [{:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f} {:.4f}]'.format(avg_metrics['Recall@1'],  avg_metrics['Recall@5'], avg_metrics['Recall@10'],  avg_metrics['NDCG@1'], avg_metrics['NDCG@5'], avg_metrics['NDCG@10']))

    def predict(self, top_n=10):
        print('Predict with best model.')
        best_model = torch.load(os.path.join(self.args.save_dir, 'model', 'best_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        
        self.model.eval()
        
        self.predict_loader = self.dataloader.get_predict_loader()
        all_preds = []
        all_scores = []
        with torch.no_grad():
            for batch in self.predict_loader:
                seq = batch['seq'].to(self.device)          # (batch_size, seq_len)
                history = batch['history']
                logits = self.model(seq)     
                last_logits = logits[:, -1, :] 
                
                for i in range(last_logits.size(0)):
                    last_logits[i, history[i]] = float('-inf')
                    
                topk_scores, topk_items = torch.topk(last_logits, k=top_n, dim=-1)
                all_preds.extend(topk_items.cpu().tolist())
                all_scores.extend(topk_scores.cpu().tolist())

        return all_preds, all_scores
        
    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics