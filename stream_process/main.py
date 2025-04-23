import os
import time
import torch
import argparse
import numpy as np

from model import SASRec
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
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)

    args = parser.parse_args()
    train_dir = "/content/drive/BIG_MOOC/train_dir"
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    dataset = data_retrieval()

    [train, validation, test, num_users, num_courses] = dataset
    num_batch = (len(train) - 1) // args.batch_size + 1

    f = open("/content/drive/BIG_MOOC/log.txt", 'w')
    f.write('epoch (val_ndcg, val_hit, val_recall) (test_ndcg, test_hit, test_recall)\n')

    sampler = sample_function(train, num_users, num_courses, batch_size=args.batch_size, sequence_size=args.sequence_size)
    model = SASRec(num_users, num_courses, args.device, hidden_units = 64, maxlen = 50, dropout_rate = 0.1, num_blocks = 2).to(args.device)

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
    adam_optimizer = torch.optim.Adam(model.parameters(), learning_rate=args.learning_rate, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    total_time = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: 
            break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            total_time += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_validation(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, total_time, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + '_' + train_dir
                fname = 'SASRec.epoch={}.learning_rate={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.learning_rate, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + train_dir
            fname = 'SASRec.epoch={}.learning_rate={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.learning_rate, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")


if __name__ == '__main__':
    main()

    