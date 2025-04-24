import requests
from confluent_kafka import Consumer, KafkaException, KafkaError
import json
import sys
import time

conf = {'bootstrap.servers': "kafka:9092",
        'group.id': "foo",
        'auto.offset.reset': 'earliest'}

consumer = Consumer(conf)
running = True
colab_url = "https://70d3-34-138-240-141.ngrok-free.app/receive"  # Change over time, broke hahaha

def basic_consume_loop(consumer, topics):
    print("Consumer is running and trying to subscribe to topics:", topics)
    
    while running:
        try:
            consumer.subscribe(topics)
            sys.stdout.flush()
            while running:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                         (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    try:
                        key = msg.key().decode('utf-8')
                        value = msg.value().decode('utf-8')

                        if key and value:
                            print(f"Saved: user_id = {key}, course_id = {value}")
                            requests.post(colab_url, json={"result": {"user_id": key, "course_id": value}})
                            sys.stdout.flush()
                        else:
                            print("Công chúa nhập sai định dạng rồi: ", key, value)
                            sys.stdout.flush()

                    except Exception as e:
                        print("Error parsing message", e)
                        sys.stdout.flush()
        except KafkaException as e:
            print(f"Kafka error occurred: {e}, retrying in 5 seconds...")
            sys.stdout.flush()
            time.sleep(5)
    
    sys.stdout.flush()
    consumer.close()

if __name__ == "__main__":
    basic_consume_loop(consumer, ['user_course_interact'])
