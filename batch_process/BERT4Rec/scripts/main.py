from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql import SparkSession

from model import BERT4Rec
from utils import *
from data_utils import * 
from config import *
from trainer import Trainer

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("TorchDistributedTraining") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=False)
    
    dataset = create_dataset(args)
    # course_pretrained_embedding = torch.load(args.pretrained_course_embedding_path)
    model = BERT4Rec(args)
    trainer = Trainer(args, model, dataset)
    
    distributor.run(trainer.train)

    trainer.test()

    trainer.predict()
