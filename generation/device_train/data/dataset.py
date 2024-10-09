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

def batch_preprocess(prompts, targets, tokenizer, prompt_max_len=150, target_max_len=50):
    batch_input_ids = []
    context_lens = []
    batch_labels = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer.encode(prompt, max_length=prompt_max_len, truncation=True)
        target_ids = tokenizer.encode(target, max_length=target_max_len, truncation=True, add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        batch_input_ids.append(input_ids)
        context_lens.append(len(prompt_ids))
    longest_len = max([len(input_ids) for input_ids in batch_input_ids])
    for i in range(len(batch_input_ids)):
        labels = [-100]*(context_lens[i]) + batch_input_ids[i][(context_lens[i]):] + [-100]*(longest_len - len(batch_input_ids[i])) 
        batch_input_ids[i] = torch.LongTensor(batch_input_ids[i] + [tokenizer.pad_token_id] * (longest_len - len(batch_input_ids[i])))
        batch_labels.append(torch.LongTensor(labels))
    return torch.stack(batch_input_ids), torch.stack(batch_labels)


def batch_prefix_preprocess(prefix_type, prompts, targets, tokenizer, prompt_max_len=150, target_max_len=50, src_label=None):
    batch_input_ids = []
    context_lens = []
    batch_labels = []
    
    for prompt, target in zip(prompts, targets):
        prompt = f'{prefix_type}|{prompt}'
        prompt_ids = tokenizer.encode(prompt, max_length=prompt_max_len, truncation=True)
        target_ids = tokenizer.encode(target, max_length=target_max_len, truncation=True, add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        batch_input_ids.append(input_ids)
        context_lens.append(len(prompt_ids))
    longest_len = max([len(input_ids) for input_ids in batch_input_ids])
    for i in range(len(batch_input_ids)):
        if src_label:
            labels = [-100]*(longest_len - len(batch_input_ids[i])) + batch_input_ids[i]
        else:
            labels = [-100]*(longest_len - len(batch_input_ids[i])) +  [-100]*(context_lens[i]) + batch_input_ids[i][(context_lens[i]):]
        batch_input_ids[i] = torch.LongTensor([tokenizer.pad_token_id] * (longest_len - len(batch_input_ids[i])) + batch_input_ids[i])
        batch_labels.append(torch.LongTensor(labels))
    return torch.stack(batch_input_ids), torch.stack(batch_labels)    

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
        return RedditDataset(data), RedditDataset(valid_data)
    else:
        logger.info(f'Loaded samples from {path} and split to {len(train_data)}/{len(valid_data)}')
        return RedditDataset(train_data), RedditDataset(valid_data)


def load_train_test_set(train_path, test_path):
    train_data = json.load(open(train_path, "r"))
    test_data = json.load(open(test_path, "r"))
    return RedditDataset(train_data), RedditDataset(test_data)


class RedditDataset(Dataset):
    def __init__(self, data) -> None:
        self.prompt = '|'
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
        input = ann['input']
        target = ann['top1']
        prompt = self.prompt + input
        #logger.info(prompt)
        return prompt, target, input