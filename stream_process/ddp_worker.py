import os
import torch
import torch.nn as nn
import socket
import pickle
import requests
import numpy as np
from threading import Thread
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from pretrain_model.utils.model import SASREC
from pretrain_model.utils.distributed_data_utils import data_retrieval

class DPP_Worker:
    def __init__(self, local_rank):
        dist.init_process_group(
            backend='gloo',
            init_method='env://'
        )
        self.local_rank = local_rank
        self.device = "cpu"
        model = SASREC(99970, 2828, self.device, embedding_dims = 64, sequence_size=15, dropout_rate=0.2).to(self.device)
        state_dict = torch.load("/content/drive/MyDrive/BIG_MOOC/train_dir/SASRec.final.pth", map_location=self.device)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("localhost", 1601 + local_rank))
        self.server_socket.listen()

        print(f"Rank {local_rank}: ready for update at port {1601 + local_rank}!")

    def process_sample(self, data):
        num_epochs = 25
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            user, seq_course, pos_course, neg_course = np.array(data["user_id"]), np.array(data["seq"]), np.array(data["pos"]), np.array(data["neg"])
            user = torch.LongTensor(user).to(self.device)
            seq_course = torch.LongTensor(seq_course).to(self.device)
            pos_course = torch.LongTensor(pos_course).to(self.device)
            neg_course = torch.LongTensor(neg_course).to(self.device)

            self.optimizer.zero_grad()
            pos_logits, neg_logits = self.model(user, seq_course, pos_course, neg_course)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device = self.device), torch.zeros(neg_logits.shape, device= self.device)

            indices = np.where(pos_course != 0)
            loss = self.bce_loss(pos_logits[indices], pos_labels[indices])
            loss += self.bce_loss(neg_logits[indices], neg_labels[indices])
            loss.backward()
            self.optimizer.step()

            print(f"RANK {self.local_rank}: Updated user {user} epoch {epoch} - loss {loss.item():.4f}")

        self.model.eval()

        with torch.no_grad():
            predict_courses = data["pred"]
            predictions = -self.model.predict(
                user,
                seq_course,
                predict_courses
            )[0]

            top5 = np.argsort(predictions)[:5]
            top5_course_ids = [predict_courses[i] for i in top5]

            print(f"RANK {self.local_rank}: Top-5 predicted courses: {top5_course_ids}")

        return loss.item(), top5_course_ids
    
    def handle_connection(self, client_socket):
        with client_socket:
            try:
                data = b""
                while True:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                data = pickle.loads(data)
                loss, top5_course_ids = self.process_sample(data)

                self.send_result_back_to_local({
                    "user_id": data["user_id"],
                    "top5_courses": top5_course_ids,
                    "loss": loss
                })

            except Exception as e:
                print(f"[RANK {self.local_rank}] Error while receiving: {e}")

    def send_result_back_to_local(self, result_data):
        try:
            ngrok_url = ngrok_static_domain + "/receive"
            headers = {'Content-Type': 'application/json'}
            response = requests.post(ngrok_url, json=result_data, headers=headers)
            print(f"[RANK {self.local_rank}] Response: {response}")
        except Exception as e:
            print(f"[RANK {self.local_rank}] Failed to send result back: {e}")
    
    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"[RANK {self.local_rank}] Accepted connection from {addr}")
            Thread(target=self.handle_connection, args=(client_socket,), daemon=True).start()

        dist.destroy_process_group()

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    worker = DPP_Worker(local_rank)
    worker.run()