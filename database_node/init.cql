-- Create Keyspace
CREATE KEYSPACE IF NOT EXISTS MOOC
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE MOOC;

-- Table 1: Course Info
CREATE TABLE IF NOT EXISTS COURSE (
  course_id TEXT PRIMARY KEY,
  name TEXT,
  about TEXT
);

-- Table 2: User Info
CREATE TABLE IF NOT EXISTS USER (
  user_id TEXT PRIMARY KEY,
  name TEXT,
  gender TEXT,
  year_of_birth INT
);

-- Table 3: User-Course Interactions
CREATE TABLE IF NOT EXISTS USER_COURSE (
  user_id TEXT,
  course_id TEXT,
  enroll_time TIMESTAMP,
  PRIMARY KEY ((user_id), course_id)
);
