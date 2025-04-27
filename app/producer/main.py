import socket
from fastapi import FastAPI, Request
from pydantic import BaseModel
from confluent_kafka import Producer

app = FastAPI()
conf = {'bootstrap.servers': "kafka:9092",
        'client.id': socket.gethostname()}
producer = Producer(conf)

user_recommendations = {}
class Interaction(BaseModel):
    user_id: str
    course_id: str

@app.post("/produce")
async def produce(interaction: Interaction):
    data = interaction.model_dump()

    user_id_bytes = str(data['user_id']).encode('utf-8')
    course_id_bytes = str(data['course_id']).encode('utf-8')

    producer.produce('user_course_interact', key=user_id_bytes, value=course_id_bytes)
    producer.flush()
    return {"message": "Data sent to Kafka", "interaction": data}

@app.post("/receive")
async def receive_data(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    top5_courses = data.get("top5_courses")

    if user_id and top5_courses:
        user_recommendations[user_id] = top5_courses
    print("Received data via ngrok:", data)
    return {"status": "received"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    user_data = user_recommendations.get(user_id)
    if user_data:
        return {"recommendedCourses": user_data}
    else:
        return {"recommendedCourses": []}