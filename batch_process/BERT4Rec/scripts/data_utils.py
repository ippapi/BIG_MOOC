import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import trange
import pickle
import numpy as np
import os

class BertTrainDataset(Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

class PredictDataset(Dataset):
    def __init__(self, u2seq, max_len, mask_token):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        history_items = list(set(seq))

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return {
            'seq': torch.LongTensor(seq),
            'history': torch.LongTensor(history_items)  
        }
    

class RandomNegativeSampler():
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, save_dir):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, 'negative_sample-sample_size{}-seed{}.pkl'.format(self.sample_size, self.seed))
        
    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def get_negative_samples(self):
        """Lấy negative samples từ file nếu đã có, nếu không thì tạo mới và lưu vào file."""
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(self.save_path):
            print("Negatives samples exist. Loading.")
            with open(self.save_path, "rb") as f:
                negative_samples = pickle.load(f)
        else:
            print("Negative samples don't exist. Generating.")
            negative_samples = self.generate_negative_samples()
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            with open(self.save_path, "wb") as f:
                pickle.dump(negative_samples, f)
            print(f"Saved negative samples to {self.save_path}")

        return negative_samples

class BertDataloader():
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.user_count = dataset['user_count']
        self.item_count = dataset['item_count']
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        train_negative_sampler = RandomNegativeSampler(self.train, self.val, self.test,
                                                       self.user_count, self.item_count,
                                                       args.train_negative_sample_size,
                                                       args.train_negative_sampling_seed,
                                                       args.data_dir
                                                      )
        test_negative_sampler = RandomNegativeSampler(self.train, self.val, self.test,
                                                      self.user_count, self.item_count,
                                                      args.test_negative_sample_size,
                                                      args.test_negative_sampling_seed,
                                                      args.data_dir
                                                      )

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

        self.predict = {}
        for key in dataset['train'].keys():
            self.predict[key] = (
                dataset['train'].get(key, []) + 
                dataset['val'].get(key, []) +
                dataset['test'].get(key, [])
            )

    def get_pytorch_dataloaders(self):
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        test_loader = self.get_test_loader()
        if self.args.is_distributed:
            return train_loader, val_loader, test_loader

        return train_loader, val_loader, test_loader

    def get_train_loader(self):
        dataset = self._get_train_dataset()
        if self.args.is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, sampler=train_sampler,
                                           shuffle=(train_sampler is None), pin_memory=True)
        return dataloader
    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset

    def get_predict_loader(self):
        dataset = PredictDataset(self.predict,  self.max_len, self.CLOZE_MASK_TOKEN)
        dataloader = DataLoader(dataset, batch_size=self.args.test_batch_size,
                                           shuffle=False, pin_memory=True, collate_fn=lambda x: {
                                               'seq': torch.stack([item['seq'] for item in x]),
                                               'history': [item['history'] for item in x]
                                           })
        return dataloader