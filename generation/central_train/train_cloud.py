import os 
os.environ['PYTHONIOENCODING'] = 'utf8'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import gc
import math
import torch
import argparse
from tqdm import tqdm
from loguru import logger
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from dataset import batch_preprocess, create_sampler, create_loader, RedditDataset
import utils


def train(model, epoch, train_loader, optimizer, lr_scheduler, args, device):
    model.train()
    total_loss = 0.0
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        prompts, targets, _ = batch
        input_ids, labels = batch_preprocess(prompts, targets, tokenizer)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        loss = model(input_ids=input_ids, labels=labels).loss
        if torch.isnan(loss): # skip the batch if loss is nan
            logger.info('loss is nan')
            continue
        total_loss += loss.detach().float().item()
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (step+1) % 100 == 0 and utils.is_main_process():
            logger.info(f'Epoch: {epoch}, Step: {step}, Loss: {total_loss / (step + 1)}')

        del loss
        del prompts
        del targets
        del input_ids
        del labels
        del batch
        if (step + 1) % 1000 == 0:
            gc.collect()

    train_epoch_loss = total_loss/len(train_loader)
    train_ppl = math.exp(train_epoch_loss)
    logger.info(f'Epoch: {epoch}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')


def evaluate(model, test_loader, tokenizer, device, args):
    model.eval()
    eval_preds = []
    eval_labels = []
    eval_topn, eval_scores = [],[]
    eval_inputs, eval_contexts = [],[]
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        prompts, targets, inputs = batch
        input_ids = tokenizer(prompts, return_tensors='pt', padding='longest', max_length=50, truncation=True).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                num_beams=args.num_beams, return_dict_in_generate=True, output_scores=True, num_return_sequences=args.num_beams)
        if args.enc_dec:
            batch_preds = tokenizer.batch_decode(outputs.sequences.detach().cpu().numpy(), skip_special_tokens=True)
        else:
            batch_preds = tokenizer.batch_decode(outputs.sequences[:,len(input_ids[0]):].detach().cpu().numpy(), skip_special_tokens=True)
        batch_preds = [pred.strip().replace(' ', '') for pred in batch_preds]
        if args.num_beams > 1:
            batch_scores = outputs.sequences_scores.cpu().numpy().tolist()
        else:
            batch_scores = [None for _ in batch_preds]
        for i in range(len(input_ids)):
            eval_topn.append(batch_preds[i*args.num_beams:(i+1)*args.num_beams])
            eval_scores.append(batch_scores[i*args.num_beams:(i+1)*args.num_beams])

        eval_preds.extend(batch_preds[::args.num_beams])
        eval_labels.extend(targets)
        eval_inputs.extend(inputs)
        eval_contexts.extend(prompts)
    result = [{'label': label, 'pred': pred.strip().replace(' ',''), 'input': input, 'context': context,  'top_n': topn, 'scores': scores}
              for label, pred, topn, scores, input, context in zip(eval_labels, eval_preds, eval_topn, eval_scores, eval_inputs, eval_contexts)]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--model_name_or_path", type=str, default='../model/gpt2')
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='./output/tmp/')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--grad_norm', type=float, default=5.0)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--num_beams', type=int, default=10)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    parser.add_argument('--grad_ckpt', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--lr_scheduler', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--default_prefix', type=str, default='null')

    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, 'log.log'))
    logger.info('arguments: {}'.format(args.__repr__()))

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    utils.seed_everything(args.seed+utils.get_rank())


    scaler = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if 'gpt2' in args.model_name_or_path.lower():
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True)
    if args.fp16:
        model = model.to(torch.bfloat16)
    if args.grad_ckpt:
        model.supports_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    
    logger.info("Left padding for decoder-only models")
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.to(device)
    model_without_ddp = model
    if args.distributed and (not args.evaluate):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    if args.train_data_dir is None:
        args.train_data_dir = args.data_dir

    train_dataset  = RedditDataset(os.path.join(args.data_dir, 'Reddit_client_train_data_new.json'))
    test_dataset = RedditDataset(os.path.join(args.data_dir, 'Reddit_client_test_data_new_gpt2sorted.json'))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, test_dataset], [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]
    train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers, batch_size=[args.batch_size, args.batch_size],
        num_workers=[args.num_workers, args.num_workers], is_trains=[True, False], collate_fns=[None, None])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    tot_steps = len(train_loader) * args.epochs//args.accumulation_steps
    logger.info("total training steps={}".format(tot_steps))
    if args.lr_scheduler == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=tot_steps
        )
    else:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=tot_steps
        )
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train(model, epoch=epoch, train_loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler, args=args, device=device)

        results = evaluate(model_without_ddp, test_loader, tokenizer, device, args)
        result_file = os.path.join(args.output_dir, 'pred_epoch{}_rank{}.json'.format(epoch, utils.get_rank()))
        json.dump(results, open(result_file, 'w', encoding='utf8'), ensure_ascii=False)
        if dist.is_initialized():
            dist.barrier()
        if utils.is_main_process():
            results = []
            for rank in range(utils.get_world_size()):
                result_file = os.path.join(args.output_dir, 'pred_epoch{}_rank{}.json'.format(epoch, rank))
                with open(result_file, 'r', encoding='utf8') as fr:
                    result = json.load(fr)
                results += result
            with open(os.path.join(args.output_dir, f'predictions_epoch{epoch}.json'), 'w', encoding='utf8') as fw:
                for pred in results:
                    fw.write(json.dumps(pred, ensure_ascii=False) + '\n')
            correct, total = 0, 0
            correctNum = dict()
            for res in results:
                total += 1 
                pred  = res['pred']
                target = res['label']
                if target.strip() == pred.strip():
                    correct += 1
                for j in range(1,args.num_beams+1):
                    if target in res['top_n'][:j]:
                        correctNum[j] = correctNum.get(j,0)+1
            eval_acc = correct / total
            logger.info(f'Epoch: {epoch}, Eval Accuracy: {correct / total}') 
            for j in range(1,args.num_beams+1):
                logger.info(f'Top {j} Acc : {correctNum.get(j,0) / total}')
            if eval_acc >= best_acc and (not args.evaluate):
                best_acc = eval_acc
                logger.info(f'Best Accuracy Boosted To: {best_acc}')
            
            ckpt_path = os.path.join(args.output_dir, f'ckpt_epoch{epoch}')
            logger.info(f'''save model to: {ckpt_path}''')
            model_without_ddp.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

        if args.evaluate:
            break
        if dist.is_initialized():
            dist.barrier()
