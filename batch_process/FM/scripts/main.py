import sys
import random
import logging
from time import time
import datetime
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data_utils import *
from model import FM
from config import args
from utils import *

def evaluate(args, model, databuilder, Ks):
    test_batch_size = databuilder.test_batch_size
    train_user_dict = databuilder.train_user_dict
    test_user_dict = databuilder.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_users = len(user_ids)
    n_items = databuilder.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    cf_users = []
    cf_items = []
    cf_scores = []

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user in user_ids_batches:
            feature_values = generate_test_batch(batch_user, databuilder.n_items, databuilder.user_matrix, databuilder.feat_matrix)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)            # (batch_size)

            cf_users.extend(np.repeat(batch_user, n_items).tolist())
            cf_items.extend(item_ids * len(batch_user))
            cf_scores.append(batch_scores.cpu())
            pbar.update(1)

    rows = [user_idx_map[u] for u in cf_users]
    cols = cf_items
    cf_scores = torch.cat(cf_scores)
    cf_score_matrix = torch.Tensor(sp.coo_matrix((cf_scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    metrics_dict = calc_metrics_at_k(cf_score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, Ks)

    cf_score_matrix = cf_score_matrix.numpy()
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_dict[k][m] = metrics_dict[k][m].mean()
    return cf_score_matrix, metrics_dict

def train(args):
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=7200))
    rank = dist.get_rank()

    # Seed đồng bộ trên tất cả process
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.manual_seed(args.seed)

    # Chỉ rank 0 log thông tin
    if rank == 0:
        log_save_id = create_log_id(args.save_dir)
        logging_config(folder=args.save_dir, name=f'log{log_save_id}', no_console=False)
        logging.info(args)

    # Load data + DistributedSampler
    data_builder = DataBuilderFM(args, logging)
    train_dataset = TrainDatasetFM(data_builder.train_user_dict)
    
    sampler = DistributedSampler(
        train_dataset,
        shuffle=True
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=data_builder.train_batch_size // dist.get_world_size(),
        sampler=sampler,
        num_workers=0,
        drop_last=True
    )

    model = FM(args, data_builder.n_users, data_builder.n_items, data_builder.n_entities, data_builder.n_user_attr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    initial_epoch = 1
    
    if args.preload == 1:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1

    model = DDP(model)

    if rank == 0:
        logging.info(model)

    if rank == 0:
        best_epoch = -1
        best_recall = 0
        Ks = eval(args.Ks)
        k_min = min(Ks)
        k_max = max(Ks)
        epoch_list = []
        metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

        if args.preload == 1:
            best_epoch = checkpoint['best_epoch']
            best_recall = checkpoint['best_recall']
            epoch_list = checkpoint['epoch_list']
            metrics_list = checkpoint['metrics_list']

    # Huấn luyện
    steps_per_epoch = data_builder.n_cf_train // data_builder.train_batch_size
    
    for epoch in range(initial_epoch, args.n_epoch + 1):
        model.train()
        dataloader.sampler.set_epoch(epoch)  

        total_loss = 0.0
        dataloader_iter = iter(dataloader)

        for step in range(steps_per_epoch):
            try:
                batch_user = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch_user = next(dataloader_iter)
    
            pos_feature_values, neg_feature_values = process_user_batch(
                batch_user=batch_user,
                user_dict=data_builder.train_user_dict,
                user_matrix=data_builder.user_matrix,
                feat_matrix=data_builder.feat_matrix
            )
            
            batch_loss = model(pos_feature_values, neg_feature_values, is_train=True)
            
            if torch.isnan(batch_loss).any():
                logging.error(f'ERROR: Epoch {epoch} Loss is nan.')
                sys.exit()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

        average_loss = total_loss / steps_per_epoch
        
        average_loss = torch.tensor(average_loss).to(args.device)
        dist.all_reduce(average_loss, op=dist.ReduceOp.SUM)
        average_loss = average_loss.item() / dist.get_world_size()

        if rank == 0:
            logging.info(f'Epoch {epoch:04d} | Average Loss: {average_loss:.4f}')

        dist.barrier()
        # Đánh giá (chỉ rank 0)
        if rank == 0 and (epoch % args.evaluate_every == 0 or epoch == args.n_epoch):
            _, metrics_dict = evaluate(args, model.module, data_builder, Ks)

            # Log và lưu metrics
            logging.info('CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])

            # Early stopping
            best_recall, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)
            if should_stop:
                break

            if metrics_list[Ks[-1]]['recall'][-1] == best_recall:
                save_model(model.module, args.save_dir, epoch, best_epoch)
                logging.info(f'Save model at epoch {epoch:04d}!')
                best_epoch = epoch
                
        if rank == 0 and (epoch % args.checkpoint_every == 0 or epoch == args.n_epoch):
            save_checkpoint(args.save_dir, model.module, optimizer, epoch, best_recall, best_epoch, metrics_list, epoch_list)
        dist.barrier()
    # Lưu kết quả cuối cùng (rank 0)
    if rank == 0:
        metrics_df = [epoch_list]
        metrics_cols = ['epoch_idx']
        for k in Ks:
            for m in ['precision', 'recall', 'ndcg']:
                metrics_df.append(metrics_list[k][m])
                metrics_cols.append('{}@{}'.format(m, k))
        metrics_df = pd.DataFrame(metrics_df).transpose()
        metrics_df.columns = metrics_cols
        metrics_df.to_csv(args.save_dir + '/metrics.csv', index=False)

        best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
        logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
            int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))

    dist.destroy_process_group()


if __name__ == "__main__":
    from pyspark.ml.torch.distributor import TorchDistributor
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("TorchDistributedTraining") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=False)
    distributor.run(train, args)