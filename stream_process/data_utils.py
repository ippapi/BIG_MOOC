import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict

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

def evaluate(model, dataset, sequence_size = 10, k = 1):
    [train, validation, test, num_users, num_courses] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(0, num_users)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_course = np.zeros([sequence_size], dtype=np.int32)
        next_index = sequence_size - 1
        seq_course[next_index] = validation[user][0] if len(validation[user]) > 0 else 0
        next_index -= 1
        for i in reversed(train[user]):
            seq_course[next_index] = i
            next_index -= 1
            if next_index == -1:
                break

        interacted_courses = set(train[user])
        interacted_courses.add(0)
        predict_courses = [test[user][0]]
        for _ in range(100):
            course = np.random.randint(0, num_courses)
            while course in interacted_courses:
                course = np.random.randint(0, num_courses)
            predict_courses.append(course)

        predictions = -model.predict(*[np.array(l) for l in [[user], [seq_course], predict_courses]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return {
        "NDCG@k": NDCG / valid_user,
        "Hit@k": HIT / valid_user,
        "Recall@k": RECALL / valid_user
    }

def evaluate_validation(model, dataset, sequence_size = 10, k = 1):
    [train, validation, test, num_users, num_courses] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(0, num_users)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_course = np.zeros([sequence_size], dtype=np.int32)
        next_index = sequence_size - 1
        for i in reversed(train[user]):
            seq_course[next_index] = i
            next_index -= 1
            if next_index == -1:
                break

        interacted_courses = set(train[user])
        interacted_courses.add(0)
        predict_courses = [validation[user][0]]
        for _ in range(100):
            predict_course = np.random.randint(0, num_courses)
            while course in interacted_courses:
                course = np.random.randint(0, num_courses)
            predict_courses.append(course)

        predictions = -model.predict(*[np.array(l) for l in [[user], [seq_course], predict_courses]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return {
        "NDCG@k": NDCG / valid_user,
        "Hit@k": HIT / valid_user,
        "Recall@k": RECALL / valid_user
    }