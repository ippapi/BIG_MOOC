import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SASREC
from data_utils import *
from evaluate_utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--sequence_size', default=10, type=int)
    parser.add_argument('--embedding_dims', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    dist.init_process_group(backend='gloo', init_method='env://')
    device = torch.device('cpu')

    dataset = data_retrieval()
    [train, validation, test, num_users, num_courses] = dataset

    sampler = DistributedSampler(train, num_users, num_courses, batch_size = args.batch_size, sequence_size = args.sequence_size, world_size = 2, rank = args.local_rank)
    model = SASREC(num_users, num_courses, args.device, embedding_dims=args.embedding_dims,
                   sequence_size=args.sequence_size, dropout_rate=args.dropout_rate,
                   num_blocks=args.num_blocks).to(device)

    model = DDP(model)

    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.module.position_emb.weight.data[0, :] = 0
    model.module.course_emb.weight.data[0, :] = 0

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        model.load_state_dict(torch.load(args.state_dict_path, map_location=map_location))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1

    if args.inference_only:
        model.eval()
        eval_result = evaluate(model.module, dataset, sequence_size=args.sequence_size, k=10)
        if dist.get_rank() == 0:
            print('test (NDCG@10: %.4f, Hit@10: %.4f, Recall@10: %4f)' % (
                eval_result["NDCG@k"], eval_result["Hit@k"], eval_result["Recall@k"]))
        return

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))

    num_batch = (len(train) - 1) // args.batch_size + 1

    best_val_ndcg, best_val_hr, best_val_recall = 0.0, 0.0, 0.0
    best_test_ndcg, best_test_hr, best_test_recall = 0.0, 0.0, 0.0
    total_time = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        sampler.set_epoch(epoch)
        if args.inference_only: break

        model.train()
        with tqdm(total=num_batch, desc=f"Epoch {epoch}", unit="batch", disable=(dist.get_rank() != 0)) as pbar:
            for step in range(num_batch):
                user, seq_course, pos_course, neg_course = sampler.next_batch()
                user, seq_course, pos_course, neg_course = np.array(user), np.array(seq_course), np.array(pos_course), np.array(neg_course)

                user = torch.LongTensor(user).to(device)
                seq_course = torch.LongTensor(seq_course).to(device)
                pos_course = torch.LongTensor(pos_course).to(device)
                neg_course = torch.LongTensor(neg_course).to(device)

                pos_logits, neg_logits = model(user, seq_course, pos_course, neg_course)
                pos_labels = torch.ones_like(pos_logits).to(device)
                neg_labels = torch.zeros_like(neg_logits).to(device)

                adam_optimizer.zero_grad()
                indices = (pos_course != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.module.course_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)

                loss.backward()
                adam_optimizer.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)

    if dist.get_rank() == 0:
        final_model_path = os.path.join("/content/drive/MyDrive/BIG_MOOC/train_dir", "SASRec.final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

    dist.destroy_process_group()
    if dist.get_rank() == 0:
        print("Training done.")

if __name__ == '__main__':
    main()
