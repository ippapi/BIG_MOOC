import numpy as np
import pandas as pd
import os
import random
import scipy.sparse as sp
import ast
from torch.utils.data import Dataset

class DataBuilderFM(object):
    def __init__(self, args, logging):
        self.args = args
        self.data_dir = args.data_dir
        self.train_file = os.path.join(self.data_dir, 'train_df.csv')
        self.test_file = os.path.join(
            self.data_dir,
            'val_df.csv' if self.args.mode == 'train' else 'test_df.csv'
        )
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.user_file = os.path.join(self.data_dir, "user_list.txt")

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        users_info = self.load_user_info(self.user_file)
        self.cf_train_data, self.train_user_dict = self.load_train_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_test_cf(self.test_file)
        self.statistic_cf()

        self.construct_data(kg_data, users_info)
        self.print_info(logging)

    def load_train_cf(self, filename):
        user = []
        item = []
        user_dict = dict()
    
        df = pd.read_csv(filename)
    
        for _, row in df.iterrows():
            user_id = int(row['user'])
    
            item_ids = list(set(ast.literal_eval(row['feature'])))
    
            for item_id in item_ids:
                user.append(user_id)
                item.append(item_id)
            user_dict[user_id] = item_ids
    
        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def load_test_cf(self, filename):
        user = []
        item = []
        user_dict = dict()
    
        df = pd.read_csv(filename, header=0, names=['user', 'label', 'time'])
    
        for _, row in df.iterrows():
            user_id = int(row['user'])
            item_id = int(row['label'])

            user.append(user_id)
            item.append(item_id)
            if user_id not in user_dict:
                user_dict[user_id] = []
            user_dict[user_id].append(item_id)
    
        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def load_user_info(self, filename):
        user_data = pd.read_csv(filename, sep=' ')
        user_data = user_data.drop_duplicates()
        return user_data

    def construct_data(self, kg_data, users_info):
        # construct user matrix
        feat_rows = list(range(self.n_users))
        feat_cols = list(range(self.n_users))
        feat_data = [1] * self.n_users

        self.n_user_attr = self.n_users

        if users_info is not None:
            user_cols = [col for col in users_info.columns
                             if col not in ['id', 'remap_id']]
            
            for col in user_cols:
                feat_rows += list(range(self.n_users))
                feat_cols += (users_info[col] + self.n_user_attr).to_list()
                feat_data += [1] * users_info.shape[0]
                self.n_user_attr += max(users_info[col]) + 1

        self.user_matrix = sp.coo_matrix((feat_data, (feat_rows, feat_cols)), shape=(self.n_users, self.n_user_attr)).tocsr()

        # construct feature matrix
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1

        feat_rows = list(range(self.n_items))
        feat_cols = list(range(self.n_items))
        feat_data = [1] * self.n_items

        filtered_kg_data = kg_data[kg_data['h'] < self.n_items]
        feat_rows += filtered_kg_data['h'].tolist()
        feat_cols += filtered_kg_data['t'].tolist()
        feat_data += [1] * filtered_kg_data.shape[0]

        self.feat_matrix = sp.coo_matrix((feat_data, (feat_rows, feat_cols)), shape=(self.n_items, self.n_entities)).tocsr()

        self.n_users_entities = self.n_user_attr + self.n_entities

    def print_info(self, logging):
        logging.info('n_users:              %d' % self.n_users)
        logging.info('n_items:              %d' % self.n_items)
        logging.info('n_entities:           %d' % self.n_entities)
        logging.info('n_user_attr:           %d' % self.n_user_attr)
        logging.info('n_users_entities:     %d' % self.n_users_entities)

        logging.info('n_cf_train:           %d' % self.n_cf_train)
        logging.info('n_cf_test:            %d' % self.n_cf_test)

        logging.info('shape of user_matrix: {}'.format(self.user_matrix.shape))
        logging.info('shape of feat_matrix: {}'.format(self.feat_matrix.shape))


class TrainDatasetFM(Dataset):
    def __init__(self, user_dict):
        self.all_users = list(user_dict.keys())

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, idx):
        user = self.all_users[idx]
        return user


def process_user_batch(batch_user, user_dict, user_matrix, feat_matrix):
    def sample_pos_items_for_u(user_dict, u, num_samples=1):
        pos_items = user_dict[u]
        return random.sample(pos_items, num_samples)
    
    def sample_neg_items_for_u(user_dict, u, num_samples=1, all_item_ids=None):
        pos_items = set(user_dict[u])
        neg_items = []
        while len(neg_items) < num_samples:
            item = random.choice(all_item_ids)
            if item not in pos_items:
                neg_items.append(item)
        return neg_items

    batch_user = batch_user.tolist()  # Tensor to list if needed

    # To get all item IDs once
    all_item_ids = list(range(feat_matrix.shape[0]))

    pos_items, neg_items = [], []

    for u in batch_user:
        pos_items += sample_pos_items_for_u(user_dict, u, 1)
        neg_items += sample_neg_items_for_u(user_dict, u, 1, all_item_ids=all_item_ids)

    batch_user_np = np.array(batch_user)
    batch_pos_item_np = np.array(pos_items)
    batch_neg_item_np = np.array(neg_items)


    user_features = user_matrix[batch_user_np]             # shape: (B, user_feat_dim)
    pos_item_features = feat_matrix[batch_pos_item_np]     # shape: (B, item_feat_dim)
    neg_item_features = feat_matrix[batch_neg_item_np]     # shape: (B, item_feat_dim)

    pos_feature_values = sp.hstack([user_features, pos_item_features])
    neg_feature_values = sp.hstack([user_features, neg_item_features])

    return pos_feature_values, neg_feature_values



def generate_test_batch(batch_user, n_items, user_matrix, feat_matrix):
    rep_batch_user = np.repeat(batch_user, n_items)
    batch_user_sp = user_matrix[rep_batch_user]

    batch_item_sp = sp.vstack([feat_matrix] * len(batch_user))

    feature_values = sp.hstack([batch_user_sp, batch_item_sp])
    return  feature_values