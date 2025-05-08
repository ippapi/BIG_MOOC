import numpy as np
import pandas as pd
import ast
from collections import defaultdict

import pandas as pd
import ast
from collections import defaultdict

def data_retrieval():
    num_users = 0
    num_courses = 0
    train = defaultdict(list)
    validation = defaultdict(list)
    test = defaultdict(list)

    def load_train(path, storage):
        nonlocal num_users, num_courses
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            user = int(row['user'])
            courses = ast.literal_eval(row['feature'])
            storage[user].extend(courses)
            num_users = max(num_users, user)
            if courses:
                num_courses = max(num_courses, max(courses))

    def load_single_label_file(path, label_column, storage):
        nonlocal num_users, num_courses
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            user = int(row['user'])
            course = int(row[label_column]) + 1
            storage[user].append(course)
            num_users = max(num_users, user)
            num_courses = max(num_courses, course)

    load_train('/content/drive/MyDrive/BIG_MOOC/dataset/train_df.csv', train)
    load_single_label_file('/content/drive/MyDrive/BIG_MOOC/dataset/val_df.csv', 'val_label', validation)
    load_single_label_file('/content/drive/MyDrive/BIG_MOOC/dataset/test_df.csv', 'test_label', test)

    return [train, validation, test, num_users + 1, num_courses + 1]


class Sampler:
    def __init__(self, users_interacts, num_users=99970, num_courses=2827, batch_size=64, sequence_size=10):
        self.users_interacts = users_interacts
        self.num_users = num_users
        self.num_courses = num_courses
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.user_ids = np.arange(0, self.num_users, dtype=np.int32)
        np.random.seed(1601)
        np.random.shuffle(self.user_ids)
        self.index = 0

    def random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def sample(self, user_id):
        while len(self.users_interacts[user_id]) <= 1:
            user_id = np.random.randint(1, self.num_users + 1)

        seq_course = np.zeros([self.sequence_size], dtype=np.int32)
        pos_course = np.zeros([self.sequence_size], dtype=np.int32)
        neg_course = np.zeros([self.sequence_size], dtype=np.int32)
        next_course = self.users_interacts[user_id][-1]
        next_id = self.sequence_size - 1

        course_set = set(self.users_interacts[user_id])
        for index in reversed(self.users_interacts[user_id][:-1]):
            seq_course[next_id] = index
            pos_course[next_id] = next_course
            if next_course != 0:
                neg_course[next_id] = self.random_neq(0, self.num_courses, course_set)
            next_course = index
            next_id -= 1
            if next_id == -1:
                break

        return user_id, seq_course, pos_course, neg_course

    def next_batch(self):
        if self.index + self.batch_size > len(self.user_ids):
            np.random.shuffle(self.user_ids)
            self.index = 0

        batch_user_ids = self.user_ids[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        batch = [self.sample(uid) for uid in batch_user_ids]
        return list(zip(*batch)) 