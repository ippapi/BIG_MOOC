import csv
from cassandra.query import BatchStatement
from cassandra.cluster import Cluster
import os

cassandra_host = os.getenv('CASSANDRA_HOST', 'cassandra')  # Default to 'cassandra' if not set
cassandra_port = int(os.getenv('CASSANDRA_PORT', 9042))    # Default to 9042 if not set

# Connect to Cassandra
cluster = Cluster([cassandra_host], port=cassandra_port)
session = cluster.connect()

try:
    session.set_keyspace('mooc')
    print("Connected to 'mooc' keyspace.")
except Exception as e:
    print(f"Failed to connect to the 'mooc' keyspace: {e}")

def init_schema():
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS mooc
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
    """)
    session.set_keyspace('mooc')

    session.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            course_id TEXT PRIMARY KEY,
            name TEXT,
            about TEXT
        );
    """)
    session.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            year_of_birth TEXT
        );
    """)
    session.execute("""
        CREATE TABLE IF NOT EXISTS user_course (
            user_id TEXT,
            course_id TEXT,
            enroll_time TIMESTAMP,
            PRIMARY KEY ((user_id), course_id)
        );
    """)
    print("Keyspace and tables are ready.")


def load_users():
    batch_size = 50
    batch = BatchStatement()
    count = 0

    with open('./data/user.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.add("""
                INSERT INTO users (user_id, name, gender, year_of_birth)
                VALUES (%s, %s, %s, %s)
            """, (row['id'], row['name'], row['gender'], row['year_of_birth']))

            count += 1

            if count % batch_size == 0:
                session.execute(batch)
                batch = BatchStatement()

    if len(batch) > 0:
        session.execute(batch)

def load_courses():
    batch_size = 50
    batch = BatchStatement()
    count = 0

    with open('./data/course.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.add("""
                INSERT INTO courses (course_id, name, about)
                VALUES (%s, %s, %s)
            """, (row['id'], row['name'], row['about']))
            count += 1

            if count % batch_size == 0:
                session.execute(batch)
                batch = BatchStatement()
                
    if len(batch) > 0:
        session.execute(batch)

# Function to batch insert enrollments
def load_enrollments():
    batch_size = 50
    batch = BatchStatement()
    count = 0
    with open('./data/user_course.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.add("""
                INSERT INTO user_course (user_id, course_id, enroll_time)
                VALUES (%s, %s, %s)
            """, (row['user'], row['course'], row['enroll_time']))
            count += 1

            if count % batch_size == 0:
                session.execute(batch)
                batch = BatchStatement()
                
    if len(batch) > 0:
        session.execute(batch)

init_schema()
load_users()
load_courses()
load_enrollments()
