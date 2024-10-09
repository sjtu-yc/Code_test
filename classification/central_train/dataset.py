import json
import torch
import random
from loguru import logger
from torch.utils.data import Dataset, DataLoader


def cls_batch_preprocess(prompts, tokenizer, prompt_max_len=150):
    batch_input_ids = []
    context_lens = []
    
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, max_length=prompt_max_len, truncation=True)
        input_ids = prompt_ids
        batch_input_ids.append(input_ids)
        context_lens.append(len(prompt_ids))
    longest_len = max([len(input_ids) for input_ids in batch_input_ids])
    for i in range(len(batch_input_ids)):
        batch_input_ids[i] = torch.LongTensor([tokenizer.pad_token_id] * (longest_len - len(batch_input_ids[i])) + batch_input_ids[i])

    return torch.stack(batch_input_ids)


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
    

class SentimentDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.data = json.load(open(path, "r"))
        print(f'Loaded {len(self.data)} samples from {path}')
    
    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def __getitem__(self, idx):
        try:
            ann = self.data[idx]
        except Exception as e:
            logger.info(f'Missing index {idx}')
            return self.__getitem__(random.randint(0, len(self.data)-1)) # load another random sample
        prompt = ann['review'].replace('"', '')
        label = int(ann['label'].replace('"', ''))
        label = int(label/4)
        return prompt, label


class SentimentDataset_method(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.data = json.load(open(path, "r"))
        print(f'Loaded {len(self.data)} samples from {path}')
    
    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def __getitem__(self, idx):
        try:
            ann = self.data[idx]
        except Exception as e:
            logger.info(f'Missing index {idx}')
            return self.__getitem__(random.randint(0, len(self.data)-1)) # load another random sample
        prompt = ann['review'].replace('"', '')
        label = int(ann['label'].replace('"', ''))
        label = int(label/4)
        client_id = ann['user']
        return prompt, label, client_id
    

class AmazonDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.data = json.load(open(path, "r"))
        print(f'Loaded {len(self.data)} samples from {path}')
    
    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def __getitem__(self, idx):
        try:
            ann = self.data[idx]
        except Exception as e:
            logger.info(f'Missing index {idx}')
            return self.__getitem__(random.randint(0, len(self.data)-1)) # load another random sample
        prompt = ann['review']
        label = int(ann['label'])
        if label <= 3:
            label = 0
        elif label == 4:
            label = 1
        else:
            label = 2
        return prompt, label


class AmazonDataset_method(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.data = json.load(open(path, "r"))
        print(f'Loaded {len(self.data)} samples from {path}')
    
    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def __getitem__(self, idx):
        try:
            ann = self.data[idx]
        except Exception as e:
            logger.info(f'Missing index {idx}')
            return self.__getitem__(random.randint(0, len(self.data)-1)) # load another random sample
        prompt = ann['review']
        client = ann['user']
        label = int(ann['label'])
        if label <= 3:
            label = 0
        elif label == 4:
            label = 1
        else:
            label = 2
        return prompt, label, client
    

class AmazonDatasetPer(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.data = json.load(open(path, "r"))
        print(f'Loaded {len(self.data)} samples from {path}')
    
    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def __getitem__(self, idx):
        try:
            ann = self.data[idx]
        except Exception as e:
            logger.info(f'Missing index {idx}')
            return self.__getitem__(random.randint(0, len(self.data)-1)) # load another random sample
        prompt = ann['review']
        label = int(ann['label'])
        user = ann['user']
        prompt = f'User {user}|{prompt}'
        if label <= 3:
            label = 0
        elif label == 4:
            label = 1
        else:
            label = 2
        return prompt, label
    
