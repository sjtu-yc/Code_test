import json
import torch
import random
from loguru import logger
from torch.utils.data import Dataset, DataLoader


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
    

class RedditDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.prompt = '|'
        with open(path, 'r') as f:
            self.data = f.readlines()
        # self.data = json.load(open(path, "r"))
        logger.info(f"Loaded {len(self.data)} samples from {path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            ann = json.loads(self.data[idx])
        except Exception as e:
            logger.info(f'Missing index {idx}')
            return self.__getitem__(random.randint(0, len(self.data)-1)) # load another random sample
        input = ann['input']
        target = ann['top1']
        prompt = self.prompt + input
        return prompt, target, input
