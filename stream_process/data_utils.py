import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict

def data_retrieval():
    num_users = 0
    num_courses = 0
    train = defaultdict(list)
    validation = defaultdict(list)
    test = defaultdict(list)

    def read_file(path, storage):
        nonlocal num_users, num_courses
        with open(path, 'r') as f:
            for line in f:
                user, course = map(int, line.strip().split())
                storage[user].append(course)
                num_users = max(num_users, user)
                num_courses = max(num_courses, course)

    read_file('/content/BIG_MOOC/dataset/train.txt', train)
    read_file('/content/BIG_MOOC/dataset/val.txt', validation)
    read_file('/content/BIG_MOOC/dataset/test.txt', test)

    return [train, validation, test, num_users + 1, num_courses + 1]

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(users_interacts, num_users=99970, num_courses=2827, batch_size=64, sequence_size=10):
    def sample(user_id):
        while len(users_interacts[user_id]) <= 1:
            user_id = np.random.randint(1, num_users + 1)

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

        return (user_id, seq_course, pos_course, neg_course)

    np.random.seed(1601)
    user_ids = np.arange(0, num_users, dtype=np.int32)
    np.random.shuffle(user_ids)
    one_batch = []

    for i in range(batch_size):
        one_batch.append(sample(user_ids[i]))

    return list(zip(*one_batch))