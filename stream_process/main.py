import json
import torch
import numpy as np
from torch import nn
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from model import SASREC
from utils import get_all_course_ids_except, topk_indices

device = "cpu"
model = SASREC(99970, 2828, device, embedding_dims = 64,
                sequence_size=15, dropout_rate=0.2).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/BIG_MOOC/train_dir/SASRec.final.pth", map_location=device))
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
bce_loss = nn.BCEWithLogitsLoss()

# Get all possible course IDs
ALL_COURSE_IDS = list(range(args.num_courses))  # assume known

def online_update_and_recommend(record):
    try:
        data = json.loads(record)
        user_id = data['user_id']
        seq = data['seq_course']
        pos = data['pos_course']
        neg = data['neg_courses']

        # Convert to tensor
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
        pos_tensor = torch.tensor([[pos]], dtype=torch.long).to(device)
        neg_tensor = torch.tensor([neg], dtype=torch.long).to(device)

        # Online update
        pos_logit, neg_logit = model(user_tensor, seq_tensor, pos_tensor, neg_tensor)
        optimizer.zero_grad()
        loss = bce_loss(pos_logit, torch.ones_like(pos_logit)) + \
               bce_loss(neg_logit, torch.zeros_like(neg_logit))
        for param in model.course_emb.parameters():
            loss += args.l2_emb * torch.norm(param)
        loss.backward()
        optimizer.step()

        print(f"[UPDATE] User {user_id} updated with loss: {loss.item():.4f}")

        # Recommend top-k
        all_courses = torch.tensor(ALL_COURSE_IDS, dtype=torch.long).to(device)
        user_seq = seq_tensor  # 1 user only

        # Duplicate input for all candidate courses
        user_batch = user_tensor.repeat(len(all_courses))
        seq_batch = user_seq.repeat(len(all_courses), 1)
        course_batch = all_courses.unsqueeze(1)

        # Forward pass
        scores, _ = model(user_batch, seq_batch, course_batch, course_batch)  # use dummy neg for shape
        scores = scores.detach().cpu().numpy()

        # Mask already seen
        seen = set(seq + [pos] + neg)
        unseen_idx = [i for i, cid in enumerate(ALL_COURSE_IDS) if cid not in seen]
        filtered_scores = scores[unseen_idx]
        filtered_courses = [ALL_COURSE_IDS[i] for i in unseen_idx]

        topk = 5
        topk_idx = np.argsort(filtered_scores)[-topk:][::-1]
        recommended = [filtered_courses[i] for i in topk_idx]

        print(f"[RECOMMEND] Top-{topk} for user {user_id}: {recommended}")
    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    spark = SparkSession.builder.appName("OnlineCourseRec").getOrCreate()
    ssc = StreamingContext(spark.sparkContext, 1)

    # Stream from socket or Kafka etc.
    lines = ssc.socketTextStream("localhost", 9999)
    lines.foreachRDD(lambda rdd: rdd.foreach(online_update_and_recommend))

    ssc.start()
    ssc.awaitTermination()

if __name__ == "__main__":
    main()
