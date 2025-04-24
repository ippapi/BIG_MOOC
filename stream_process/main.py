import os
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType
import socket
import pickle
from threading import Thread
from pretrain_model.utils.distributed_data_utils import data_retrieval
import numpy as np

schema = StructType([
    StructField("result", StructType([
        StructField("user_id", StringType()),
        StructField("course_id", StringType())
    ]))
])

def run_distributor(num_workers):
    try:
        distributor = TorchDistributor(
            num_processes=num_workers,
            local_mode=True,
            use_gpu=False
        )
        
        distributor.run("/content/BIG_MOOC/stream_process/ddp_worker.py")
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()

def sample_negative_item_for_user(user_id, users_interacts, num_courses, sequence_size=10):
    def random_neq(l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    if user_id not in users_interacts or len(users_interacts[user_id]) <= 1:
        return None

    seq_course = np.zeros([sequence_size], dtype=np.int32)
    pos_course = np.zeros([sequence_size], dtype=np.int32)
    neg_course = np.zeros([sequence_size], dtype=np.int32)
    next_course = users_interacts[user_id][-1]
    next_id = sequence_size - 1

    course_set = set(users_interacts[user_id])
    for index in reversed(users_interacts[user_id][:-1]):
        seq_course[next_id] = index
        pos_course[next_id] = next_course
        if next_course != 0:
            neg_course[next_id] = random_neq(0, num_courses, course_set)
        next_course = index
        next_id -= 1
        if next_id == -1:
            break

    predict_courses = list(set(range(num_courses)) - course_set)

    return {
        "user_id": user_id,
        "seq": [seq_course.tolist()],
        "pos": [pos_course.tolist()],
        "neg": [neg_course.tolist()],
        "pred": [predict_courses]
    }

def send_to_ddp_worker(partition, num_workers = 2):
    for record in partition:
        user_id = record["user_id"]
        seq = record["seq"]
        pos = record["pos"]
        neg = record["neg"]
        pred = record["pred"]

        sample = {
            "user_id": user_id,          
            "seq": seq,                  
            "pos": pos,
            "neg": neg,
            "pred": pred,
        }

        worker_rank = user_id % num_workers
        worker_port = 1601 + worker_rank

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("localhost", worker_port))
                s.sendall(pickle.dumps(sample))
        except Exception as e:
            print(f"[ERROR] Failed to send to worker {worker_rank} (port {worker_port}): {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=2, type=int)
    args = parser.parse_args()

    dataset = data_retrieval()
    [interact_history, _, _, num_users, num_courses] = dataset

    spark = SparkSession.builder \
        .appName("Streaming REC") \
        .getOrCreate()

    thread = Thread(target=run_distributor, args=(args.num_workers,))
    thread.start()

    print(f"[Main] Launched {args.num_workers} workers")

    def parse_and_send(df, batch_id):
        print("received")
        if df.rdd.isEmpty():
            return
        
        print(df)

        parsed_df = df.select(from_json(col("value"), schema).alias("data"))

        parsed_df = parsed_df.select("data.result.user_id", "data.result.course_id")

        records = parsed_df.rdd.collect()

        print(records)

        training_data = []
        for record in records:
            user_id = int(record["user_id"])
            course_id = int(record["course_id"])

            if user_id not in interact_history:
                interact_history[user_id] = []

            interact_history[user_id].append(course_id)

            training_row = sample_negative_item_for_user(user_id, interact_history, num_courses, sequence_size=15)

            if training_row:
                training_data.append(training_row)

            print(training_row)

        send_to_ddp_worker(training_data, num_workers=args.num_workers)

    spark = SparkSession.builder.appName("StreamingRec").getOrCreate()

    df = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()
    
    df.writeStream \
        .format("console") \
        .outputMode("append") \
        .start()


    query = df.writeStream \
        .foreachBatch(parse_and_send) \
        .outputMode("append") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
