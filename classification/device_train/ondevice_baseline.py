import os
os.environ['PYTHONIOENCODING'] = 'utf8'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from loguru import logger
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import sys
from nodes.client import Client_self_train
from copy import deepcopy
from data.dataset import load_train_test_set, load_train_test_set_amazon
from utils import seed_everything
from transformers import AutoTokenizer, AutoModel


def ondevice_train(model_c, tokenizer_c, client_num, gpu_card, args):
    logger.add(os.path.join(args.output_dir, 'log.log'))
    try:   
        if args.dataset == 'sentiment':
            if not (os.path.exists(os.path.join(args.data_folder, f'user_train/user_{client_num}.json')) and os.path.exists(os.path.join(args.data_folder, f'user_test/user_{client_num}.json'))):
                logger.info(f'Client {client_num}: Data not exist!')
                return
            else:
                train_data, valid_data = load_train_test_set(os.path.join(args.data_folder, f'user_train/user_{client_num}.json'), os.path.join(args.data_folder, f'user_test/user_{client_num}.json'))
        else:
            if not (os.path.exists(os.path.join(args.data_folder, f'user_train/user{client_num}.json')) and os.path.exists(os.path.join(args.data_folder, f'user_test/user{client_num}.json'))):
                logger.info(f'Client {client_num}: Data not exist!')
                return
            else:
                train_data, valid_data = load_train_test_set_amazon(os.path.join(args.data_folder, f'user_train/user{client_num}.json'), os.path.join(args.data_folder, f'user_test/user{client_num}.json'))
        client = Client_self_train(deepcopy(model_c), deepcopy(tokenizer_c), train_data, valid_data, gpu_card, client_num=client_num, args=args)
        client.train_test_acc()
    except Exception as e:
        print(e)
        logger.info('Error happens')
        sys.exit(1)
    return


class clsModel(nn.Module):
    def __init__(self, transformer, num_labels, enc_dec=False, pad_token_id=0):
        super(clsModel, self).__init__()
        self.backbone = transformer
        # Assuming that the avg_pool is meant to be an average pooling layer over the sequence dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.enc_dec = enc_dec
        self.pad_token_id = pad_token_id

    def forward(self, input_ids):
        if self.enc_dec:
            decoder_input_ids = torch.full((input_ids.size(0), 1), self.pad_token_id, dtype=torch.long).to(input_ids.device)
            outputs = self.backbone(input_ids=input_ids, decoder_input_ids=decoder_input_ids).last_hidden_state
        else:
            outputs = self.backbone(input_ids=input_ids).last_hidden_state
        pooled_output = outputs[:,-1,:]
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Load the model's state_dict
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_clients', type=int, default=5)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--process_per_gpu', type=int, default=1)

    parser.add_argument('--transformer', default='./model/gpt2')
    parser.add_argument('--model', default='')
    parser.add_argument('--enc_dec', action='store_true')
    parser.add_argument('--data_folder', default='./sent_cls/data')
    parser.add_argument('--output_dir', default='local_output/')
    parser.add_argument('--init_path', default=None)
    
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--grad_norm', type=float, default=5.0)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--n_tokens', type=int, default=1)
    parser.add_argument('--client_epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='sentiment', choices=['sentiment', 'amazon']) 

    args = parser.parse_args()
    
    seed_everything(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'clients/'), exist_ok=True)
    logger.add(os.path.join(args.output_dir, 'log.log'))
    logger.info('arguments: {}'.format(args.__repr__()))

    tokenizer = AutoTokenizer.from_pretrained(args.transformer, trust_remote_code=True)
    if 't5' in args.transformer.lower():
        init_transformer = AutoModel.from_pretrained(args.transformer, trust_remote_code=True)
    else:
        init_transformer = AutoModel.from_pretrained(args.transformer, trust_remote_code=True, torch_dtype=torch.float32)

    if args.dataset == 'sentiment':
        init_model = clsModel(init_transformer, num_labels=2, enc_dec=args.enc_dec, pad_token_id=tokenizer.pad_token_id)
    else:
        init_model = clsModel(init_transformer, num_labels=3, enc_dec=args.enc_dec, pad_token_id=tokenizer.pad_token_id)
    init_model.load(args.model)

    logger.info("Left padding for decoder-only models")
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------- Select Clients and Finetuning on Device -------------------------------------
    # sample clients
    data_processes = []
    process_per_round = args.process_per_gpu * args.world_size
    assert args.total_clients % process_per_round == 0
    for gpu_round in range(args.total_clients // process_per_round):
        clients_of_this_round = range(1 + gpu_round * process_per_round, 1 + gpu_round * process_per_round + process_per_round)
        for gpu_card in range(args.world_size):
            for process_num in range(args.process_per_gpu):
                client_num = clients_of_this_round[gpu_card*args.process_per_gpu + process_num]
                process = mp.Process(target=ondevice_train, args=(init_model, tokenizer, client_num, gpu_card, args))
                process.start()
                data_processes.append(process)
        for process in data_processes:
            process.join(900)
        for process in data_processes:
            if process.is_alive():
                logger.info('Kill process!')
                process.terminate()
