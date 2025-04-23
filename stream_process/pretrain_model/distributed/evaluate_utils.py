import sys
import copy
import numpy as np

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