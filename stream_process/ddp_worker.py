import os
import time
import torch
import torch.nn as nn
import socket
import pickle
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pretrain_model.utils.model import SASREC
from pretrain_model.utils.distributed_data_utils import data_retrieval

class DPP_Worker:
    def __init__(self, local_rank, port):
        dist.init_process_group(
            backend='gloo',
            init_method='env://'
        )
        self.local_rank = local_rank
        self.device = "cpu"
        model = SASREC(99970, 2828, self.device, embedding_dims = 64, sequence_size=15, dropout_rate=0.2).to(self.device)
        model.load_state_dict(torch.load("/content/drive/MyDrive/BIG_MOOC/train_dir/SASRec.final.pth", map_location=self.device))
        model.train()
        self.model = DDP(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("localhost", port))
        self.server_socket.listen()

        print(f"Rank {local_rank}: ready for update at port {port}!")

    def process_sample(self, data):
        user, seq_course, pos_course, neg_course = np.array(data["user_id"]), np.array(data["seq"]), np.array(data["pos"]), np.array(data["neg"])

        self.optimizer.zero_grad()
        pos_logits, neg_logits = self.model(user, seq_course, pos_course, neg_course)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device = self.device), torch.zeros(neg_logits.shape, device= self.device)

        loss = self.bce_loss(pos_logits[indices], pos_labels[indices])
        loss += self.bce_loss(neg_logits[indices], neg_labels[indices])
        indices = np.where(pos_course != 0)
        loss.backward()
        self.optimizer.step()

        print(f"RANK {self.local_rank}: Updated user {user} - loss {loss.item():.4f}")

        return loss.item()
    
    
    def run(self):
        client_socket, addr = self.server_socket.accept()
        while client_socket:
            try:
                data = b""
                while True:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                data = pickle.loads(data)
            except Exception:
                print(f"[RANK {self.local_rank}] is idle ...")
                time.sleep(1)
                continue

            self.process_sample(data)

        dist.destroy_process_group()
