import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from utils.model import SASREC
from utils.distributed_data_utils import *
from utils.distributed_evaluate_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()

    dataset = data_retrieval(mode = "train")

    model = SASREC(99970, 2828, args.device, embedding_dims = 64, sequence_size=15, dropout_rate=0.2).to(args.device)
    state_dict = torch.load(args.state_dict_path, map_location=args.device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.eval()
    k = 10
    print("Evaluating")
    test_result = evaluate(model, dataset, sequence_size = 10, k = k)
    val_result = evaluate_validation(model, dataset, sequence_size = 10, k = k)
    print('valid (NDCG@%d: %.4f, Hit@%d: %.4f, Recall@%d: %.4f), test (NDCG@%d: %.4f, Hit@%d: %.4f, Recall@%d: %.4f)' %
        (k, val_result["NDCG@k"], k, val_result["Hit@k"], k, val_result["Recall@k"],
        k, test_result["NDCG@k"], k, test_result["Hit@k"], k, test_result["Recall@k"]))

    print("Done")

if __name__ == '__main__':
    main()

    