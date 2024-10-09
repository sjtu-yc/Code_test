import json
import torch
import random
from math import ceil
from loguru import logger
from torch.utils.data import Dataset


def simple_collate(batch):
    # print(batch)
    # print([x for x in zip(*batch)])
    return [x for x in zip(*batch)]   


def word_batch_preprocess(prompts, tokenizer, prompt_max_len=150):
    batch_input_ids = []
    context_lens = []
    batch_labels = []
    # print(prompts)
    for prompt in prompts:
        # print(prompt)
        prompt_ids = tokenizer.encode(prompt, max_length=prompt_max_len, truncation=True)
        prompt_ids = [id for id in prompt_ids if id != tokenizer.encode('|')[0]]
        # print(prompt, prompt_ids)
        input_ids = prompt_ids + [tokenizer.eos_token_id]
        batch_input_ids.append(input_ids)
        context_lens.append(len(prompt_ids))
    longest_len = max([len(input_ids) for input_ids in batch_input_ids])
    for i in range(len(batch_input_ids)):
        # print(tokenizer.pad_token_id)
        labels = [-100]*(longest_len - len(batch_input_ids[i])) +  batch_input_ids[i]
        batch_input_ids[i] = torch.LongTensor([tokenizer.pad_token_id] * (longest_len - len(batch_input_ids[i])) + batch_input_ids[i])
        batch_labels.append(torch.LongTensor(labels))
    return torch.stack(batch_input_ids), torch.stack(batch_labels)


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


def random_split(path, ratio, args, include=True):
    data = json.load(open(path, "r"))
    random.shuffle(data)
    split_point = ceil(ratio * len(data))
    train_data, valid_data = data[:split_point], data[split_point:]
    if include:
        logger.info(f'Loaded samples from {path} and split to {len(train_data) + len(valid_data)}/{len(valid_data)}')
        return SentimentDataset(data), SentimentDataset(valid_data)
    else:
        logger.info(f'Loaded samples from {path} and split to {len(train_data)}/{len(valid_data)}')
        return SentimentDataset(train_data), SentimentDataset(valid_data)


def random_split_amazon(path, ratio, args, include=True):
    data = json.load(open(path, "r"))
    random.shuffle(data)
    split_point = ceil(ratio * len(data))
    train_data, valid_data = data[:split_point], data[split_point:]
    if include:
        logger.info(f'Loaded samples from {path} and split to {len(train_data) + len(valid_data)}/{len(valid_data)}')
        return AmazonDataset(data), AmazonDataset(valid_data)
    else:
        logger.info(f'Loaded samples from {path} and split to {len(train_data)}/{len(valid_data)}')
        return AmazonDataset(train_data), AmazonDataset(valid_data)


def load_train_test_set(train_path, test_path):
    train_data = json.load(open(train_path, "r"))
    test_data = json.load(open(test_path, "r"))
    return SentimentDataset(train_data), SentimentDataset(test_data)


def load_train_test_set_amazon(train_path, test_path):
    train_data = json.load(open(train_path, "r"))
    test_data = json.load(open(test_path, "r"))
    return AmazonDataset(train_data), AmazonDataset(test_data)


class SentimentDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
        # print(f'Loaded {len(self.data)} samples')
    
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


class AmazonDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
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
