import sys
import copy
import torch
import torch.distributed as dist

def evaluate(model, dataset, sequence_size=10, k=1, device='cpu'):
    [train, validation, test, num_users, num_courses] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    # users = range(0, num_users)
    users = range(0, 1)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_course = torch.zeros(sequence_size, dtype=torch.long)
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

        while len(predict_courses) < 100:
            course = torch.randint(1, num_courses + 1, (1,)).item()
            if course not in interacted_courses:
                predict_courses.append(course)
                print(predict_courses)

        user_tensor = torch.tensor([user], dtype=torch.long, device=device)
        seq_tensor = seq_course.unsqueeze(0).to(device)
        items_tensor = torch.tensor(predict_courses, dtype=torch.long, device=device)

        with torch.no_grad():
            real_model = model.module if hasattr(model, 'module') else model
            predictions = -real_model.predict(user_tensor, seq_tensor, items_tensor)
            predictions = predictions[0].cpu()

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / torch.log2(torch.tensor(rank + 2.0)).item()
            HIT += 1
            RECALL += 1

        if valid_user % 10000 == 0:
            print('.', end="")
            sys.stdout.flush()

    if dist.is_initialized():
        metrics = torch.tensor([NDCG, HIT, RECALL, valid_user], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        NDCG, HIT, RECALL, valid_user = metrics.tolist()

    if valid_user != 0:
        return {
            "NDCG@k": NDCG / valid_user,
            "Hit@k": HIT / valid_user,
            "Recall@k": RECALL / valid_user
        }
    else:
        return {
            "NDCG@k": 0.0,
            "Hit@k": 0.0,
            "Recall@k": 0.0
        }


def evaluate_validation(model, dataset, sequence_size=10, k=1, device='cpu'):
    [train, validation, test, num_users, num_courses] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(0, num_users)

    for user in users:
        if len(train[user]) < 1 or len(validation[user]) < 1:
            continue

        seq_course = torch.zeros(sequence_size, dtype=torch.long)
        next_index = sequence_size - 1
        for i in reversed(train[user]):
            seq_course[next_index] = i
            next_index -= 1
            if next_index == -1:
                break

        interacted_courses = set(train[user])
        interacted_courses.add(0)
        predict_courses = [validation[user][0]]

        while len(predict_courses) < 100:
            course = torch.randint(1, num_courses + 1, (1,)).item()
            if course not in interacted_courses:
                predict_courses.append(course)
                print(predict_courses)

        user_tensor = torch.tensor([user], dtype=torch.long, device=device)
        seq_tensor = seq_course.unsqueeze(0).to(device)
        items_tensor = torch.tensor(predict_courses, dtype=torch.long, device=device)

        with torch.no_grad():
            real_model = model.module if hasattr(model, 'module') else model
            predictions = -real_model.predict(user_tensor, seq_tensor, items_tensor)
            predictions = predictions[0].cpu()

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        print(valid_user)

        if rank < k:
            NDCG += 1 / torch.log2(torch.tensor(rank + 2.0)).item()
            HIT += 1
            RECALL += 1

        if valid_user % 10000 == 0:
            print('.', end="")
            sys.stdout.flush()

    if dist.is_initialized():
        metrics = torch.tensor([NDCG, HIT, RECALL, valid_user], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        NDCG, HIT, RECALL, valid_user = metrics.tolist()

    if valid_user != 0:
        return {
            "NDCG@k": NDCG / valid_user,
            "Hit@k": HIT / valid_user,
            "Recall@k": RECALL / valid_user
        }
    else:
        return {
            "NDCG@k": 0.0,
            "Hit@k": 0.0,
            "Recall@k": 0.0
        }
