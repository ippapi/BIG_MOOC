import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import scipy.sparse as sp

class FM(nn.Module):

    def __init__(self, args,
                 n_users, n_items, n_entities, n_user_attr):

        super(FM, self).__init__()
        self.preload = args.preload
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_attr = n_user_attr
        self.n_entities = n_entities
        self.n_features = n_user_attr + n_entities

        self.embed_dim = args.embed_dim
        self.l2loss_lambda = args.l2loss_lambda

        self.linear = nn.Linear(self.n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)

        self.feature_embed = nn.Parameter(torch.Tensor(self.n_features, self.embed_dim))
        nn.init.xavier_uniform_(self.feature_embed)

        self.h = nn.Linear(self.embed_dim, 1, bias=False)
        with torch.no_grad():
            self.h.weight.copy_(torch.ones([1, self.embed_dim]))
        for param in self.h.parameters():
            param.requires_grad = False

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()

    def calc_score(self, feature_values):
        """
        feature_values:  (batch_size, n_features), n_features = n_users + n_entities, torch.sparse.FloatTensor
        """
        # Bi-Interaction layer
        # Equation (4) / (3)
        feature_values = self.convert_coo2tensor(feature_values.tocoo())
        sum_square_embed = torch.mm(feature_values, self.feature_embed).pow(2)           # (batch_size, embed_dim)
        square_sum_embed = torch.mm(feature_values.pow(2), self.feature_embed.pow(2))    # (batch_size, embed_dim)
        z = 0.5 * (sum_square_embed - square_sum_embed)                                  # (batch_size, embed_dim)

        # Prediction layer
        # Equation (6)
        y = self.h(z)                                       # (batch_size, 1)
        # Equation (2) / (7) / (8)
        y = self.linear(feature_values) + y                 # (batch_size, 1)
        return y.squeeze()

    def calc_loss(self, pos_feature_values, neg_feature_values):
        """
        pos_feature_values:  (batch_size, n_features), torch.sparse.FloatTensor
        neg_feature_values:  (batch_size, n_features), torch.sparse.FloatTensor
        """
        pos_scores = self.calc_score(pos_feature_values)            # (batch_size)
        neg_scores = self.calc_score(neg_feature_values)            # (batch_size)

        loss = (-1.0) * torch.log(1e-10 + F.sigmoid(pos_scores - neg_scores))
        loss = torch.mean(loss)

        l2_loss = torch.norm(self.h.weight, 2).pow(2) / 2
        loss += self.l2loss_lambda * l2_loss
        return loss


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)