CREATE KEYSPACE IF NOT EXISTS mooc
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE mooc;

CREATE TABLE IF NOT EXISTS courses (
  course_id TEXT PRIMARY KEY,
  name TEXT,
  about TEXT
);

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  name TEXT,
  gender TEXT,
  year_of_birth TEXT
);

CREATE TABLE IF NOT EXISTS user_course (
  user_id TEXT,
  course_id TEXT,
  enroll_time TIMESTAMP,
  PRIMARY KEY ((user_id), course_id)
);
