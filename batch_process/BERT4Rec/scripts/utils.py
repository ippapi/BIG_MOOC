import os
import logging
import torch
import pandas as pd
import ast
from collections import defaultdict
import ast


def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder

def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_checkpoint(model_dir, model, optimizer, current_epoch, best_recall, best_epoch, metrics_list, epoch_list):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_file = os.path.join(model_dir, 'checkpoint_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': current_epoch,
                'best_recall': best_recall,
                'best_epoch': best_epoch,
                'metrics_list': metrics_list,
                'epoch_list': epoch_list
               }, checkpoint_file)

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics

def create_dataset(args):
    num_users = 0
    num_courses = 0
    train = defaultdict(list)
    val = defaultdict(list)
    test = defaultdict(list)

    def load_train(path, storage):
        nonlocal num_users, num_courses
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            user = int(row['user'])
            courses = ast.literal_eval(row['feature'])
            courses = [course + 1 for course in courses]
            storage[user].extend(courses)
            num_users = max(num_users, user)
            if courses:
                num_courses = max(num_courses, max(courses))

    def load_single_label_file(path, label_column, storage):
        nonlocal num_users, num_courses
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            user = int(row['user'])
            course = int(row[label_column])
            storage[user].append(course + 1)
            num_users = max(num_users, user)
            num_courses = max(num_courses, course)

    data_dir = args.data_dir
    load_train(os.path.join(data_dir, 'train_df.csv'), train)
    load_single_label_file(os.path.join(data_dir, 'val_df.csv'), 'val_label', val)
    load_single_label_file(os.path.join(data_dir, 'test_df.csv'), 'test_label', test)
    args.num_items = num_courses + 1

    dataset = {'train': train,'val': val,'test': test,'user_count': num_users + 1,'item_count': num_courses + 1}
    return dataset