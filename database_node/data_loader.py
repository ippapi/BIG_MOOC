import csv
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from cassandra.query import SimpleStatement

cluster = Cluster(["BIG_MOOC"])
session = cluster.connect("MOOC")

def load_users():
    batch = BatchStatement()
    with open('./data/user.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.add("""
                INSERT INTO users (user_id, name, gender, year_of_birth)
                VALUES (%s, %s, %s, %s)
            """, (row['user_id'], row['name'], row['gender'], int(row['year_of_birth'])))
    session.execute(batch)

def load_courses():
    batch = BatchStatement()
    with open('./data/course.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.add("""
                INSERT INTO courses (course_id, name, about)
                VALUES (%s, %s, %s)
            """, (row['course_id'], row['name'], row['about']))
    session.execute(batch)

# Function to batch insert enrollments
def load_enrollments():
    batch = BatchStatement()
    with open('./data/user_course.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.add("""
                INSERT INTO user_course_enrollments (user_id, course_id, enroll_time)
                VALUES (%s, %s, %s)
            """, (row['user_id'], row['course_id'], row['enroll_time']))
    session.execute(batch)

load_users()
load_courses()
load_enrollments()
cluster.shutdown()
