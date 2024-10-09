import os
os.environ['PYTHONIOENCODING'] = 'utf8'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from loguru import logger
import argparse
import torch.multiprocessing as mp
import sys
from nodes import Client_self_train
from copy import deepcopy
from data.dataset import load_train_test_set
from utils import seed_everything
from transformers import AutoTokenizer, GPT2LMHeadModel


def split_cluster(model_c, tokenizer_c, client_num, gpu_card, args):
    logger.add(os.path.join(args.output_dir, 'log.log'))
    try:
        train_data, valid_data = load_train_test_set(os.path.join(args.data_folder, f'train/user_{client_num}.json'), os.path.join(args.data_folder, f'test/user_{client_num}.json'))
        # print(valid_data[1])
        if len(train_data) == 0 or len(valid_data) == 0:
            logger.info(f'Client {client_num}: Data not exist!')
            return
        client = Client_self_train(deepcopy(model_c), deepcopy(tokenizer_c), train_data, valid_data, gpu_card, client_num=client_num, args=args)
        # client.pure_train()
        client.train_eval_acc()
    except Exception as e:
        print(e)
        logger.info('Error happens')
        sys.exit(1)
    return


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_clients', type=int, default=5)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--process_per_gpu', type=int, default=1)

    parser.add_argument('--model', default='../model/gpt2_reddit_1407_user_baseline/ckpt')
    parser.add_argument('--data_folder', default='../data/1407_user/user_data')
    parser.add_argument('--output_dir', default='local_output/')
    parser.add_argument('--init_path', default=None)
    
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_eval', type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--grad_norm', type=float, default=5.0)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--split_pinyin', action='store_true')
    parser.add_argument('--simple_prompt', action='store_true')
    parser.add_argument('--default_prefix', default='null')

    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--client_epochs', type=int, default=20)

    args = parser.parse_args()
    
    seed_everything(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'clients/'), exist_ok=True)
    logger.add(os.path.join(args.output_dir, 'log.log'))
    logger.info('arguments: {}'.format(args.__repr__()))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    init_model = GPT2LMHeadModel.from_pretrained(args.model)

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
                process = mp.Process(target=split_cluster, args=(init_model, tokenizer, client_num, gpu_card, args))
                # split_cluster_debug(init_model, tokenizer, client_num, gpu_card, args)
                process.start()
                data_processes.append(process)
        for process in data_processes:
            process.join(900)
        for process in data_processes:
            if process.is_alive():
                logger.info('Kill process!')
                process.terminate()
