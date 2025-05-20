import logging
import torch

from model import FM
from main import evaluate
from data_utils import *
from utils import *
from config import args

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_builder = DataBuilderFM(args, logging)
     
    model = FM(args, data_builder.n_users, data_builder.n_items, data_builder.n_entities, data_builder.n_user_attr)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)
    
    cf_scores, metrics_dict = evaluate(args, model, data_builder, Ks)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
    

if __name__ == '__main__':
    args.mode = "test"
    args.pretrain_model_path = "path/to/pretrained/model"
    predict(args)