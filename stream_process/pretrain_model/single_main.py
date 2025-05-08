import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

from utils.single_model import SASREC
from utils.single_data_utils import *
from utils.single_evaluate_utils import *

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
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()

    train_dir = "/content/drive/MyDrive/BIG_MOOC/train_dir"
    dataset = data_retrieval()
    [train, _, _, num_users, num_courses] = dataset
    num_batch = (len(train) - 1) // args.batch_size + 1

    f = open("/content/drive/MyDrive/BIG_MOOC/log.txt", 'w')
    f.write('epoch (val_ndcg, val_hit, val_recall) (test_ndcg, test_hit, test_recall)\n')

    sampler = Sampler(train, num_users, num_courses, batch_size=args.batch_size, sequence_size=args.sequence_size)
    model = SASREC(num_users, num_courses, args.device, embedding_dims = args.embedding_dims, sequence_size = args.sequence_size, dropout_rate = args.dropout_rate, num_blocks = args.num_blocks).to(args.device)
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    model.position_emb.weight.data[0, :] = 0
    model.course_emb.weight.data[0, :] = 0
    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1

    if args.inference_only:
        model.eval()
        eval_result = evaluate(model, dataset, sequence_size = 10, k = 10)
        print('test (NDCG@10: %.4f, Hit@10: %.4f, Recall@10: %4f)' % (eval_result["NDCG@k"], eval_result["Hit@k"], eval_result["Recall@k"]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    best_val_ndcg, best_val_hr, best_val_recall = 0.0, 0.0, 0.0
    best_test_ndcg, best_test_hr, best_test_recall = 0.0, 0.0, 0.0
    total_time = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: 
            break

        with tqdm(total=num_batch, desc=f"Epoch {epoch}/{args.num_epochs}", unit="batch") as pbar:
            for step in range(num_batch):
                user, seq_course, pos_course, neg_course = sampler.next_batch()
                user, seq_course, pos_course, neg_course = np.array(user), np.array(seq_course), np.array(pos_course), np.array(neg_course)

                pos_logits, neg_logits = model(user, seq_course, pos_course, neg_course)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

                adam_optimizer.zero_grad()
                indices = np.where(pos_course != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.course_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)

                loss.backward()
                adam_optimizer.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)

        if epoch % 25 == 0:
            model.eval()
            t1 = time.time() - t0
            total_time += t1
            print('Evaluating')
            for k in [10]:
                test_result = evaluate(model, dataset, sequence_size = 10, k = k)
                val_result = evaluate_validation(model, dataset, sequence_size = 10, k = k)
                print('epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, Hit@%d: %.4f, Recall@%d: %.4f), test (NDCG@%d: %.4f, Hit@%d: %.4f, Recall@%d: %.4f)' %
                    (epoch, total_time, k, val_result["NDCG@k"], k, val_result["Hit@k"], k, val_result["Recall@k"],
                    k, test_result["NDCG@k"], k, test_result["Hit@k"], k, test_result["Recall@k"]))


            if val_result["NDCG@k"] > best_val_ndcg or val_result["Hit@k"] > best_val_hr or val_result["Recall@k"] > best_val_recall or test_result["NDCG@k"] > best_test_ndcg or test_result["Hit@k"] > best_test_hr or test_result["Recall@k"] > best_test_recall:
                best_val_ndcg = max(val_result["NDCG@k"], best_val_ndcg)
                best_val_hr = max(val_result["Hit@k"], best_val_hr)
                best_val_recall = max(val_result["Recall@k"], best_val_recall)
                best_test_ndcg = max(test_result["NDCG@k"], best_test_ndcg)
                best_test_hr = max(test_result["Hit@k"], best_test_hr)
                best_test_recall = max(test_result["Recall@k"], best_test_recall)
                folder = train_dir
                fname = 'SASRec.epoch={}.learning_rate={}.layer={}.head={}.embedding_dims={}.sequence_size={}.pth'
                fname = fname.format(epoch, args.learning_rate, args.num_blocks, args.num_heads, args.embedding_dims, args.sequence_size)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(epoch) + ' ' + str(val_result) + ' ' + str(test_result) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

    f.close()
    print("Done")

if __name__ == '__main__':
    main()

    