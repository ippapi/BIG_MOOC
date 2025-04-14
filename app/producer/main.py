import socket
from fastapi import FastAPI
from pydantic import BaseModel
from confluent_kafka import Producer

app = FastAPI()
conf = {'bootstrap.servers': "kafka:9092",
        'client.id': socket.gethostname()}
producer = Producer(conf)

class Interaction(BaseModel):
    user_id: int
    course_id: int

@app.post("/produce")
async def produce(interaction: Interaction):
    data = interaction.model_dump()

    user_id_bytes = str(data['user_id']).encode('utf-8')
    course_id_bytes = str(data['course_id']).encode('utf-8')

    producer.produce('user_course_interact', key=user_id_bytes, value=course_id_bytes)
    producer.flush()
    return {"message": "Data sent to Kafka", "interaction": data}