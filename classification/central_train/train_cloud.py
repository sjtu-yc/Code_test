import os 
os.environ['PYTHONIOENCODING'] = 'utf8'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from loguru import logger
import torch.distributed as dist
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from dataset import cls_batch_preprocess, SentimentDataset, create_sampler, create_loader, AmazonDataset
import utils


def train(model, epoch, loss_func, train_loader, optimizer, lr_scheduler, args, device):
    model.train()
    total_loss = 0.0
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        prompts, labels = batch
        input_ids = cls_batch_preprocess(prompts, tokenizer)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids)
        # print(outputs.shape)
        loss = loss_func(outputs, labels)
        if torch.isnan(loss): # skip the batch if loss is nan
            logger.info('loss is nan')
            continue
        total_loss += loss.detach().float()
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            if args.use_scheduler:
                lr_scheduler.step()
            optimizer.zero_grad()
        if (step+1) % 100 == 0 and utils.is_main_process():
            logger.info(f'Epoch: {epoch}, Step: {step+1}, Loss: {total_loss / (step + 1)}')
    train_epoch_loss = total_loss/len(train_loader)
    train_ppl = torch.exp(train_epoch_loss)
    logger.info(f'Epoch: {epoch}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')


def evaluate(model, test_loader, tokenizer, device, args):
    model.eval()
    correct = 0.0
    ttl = 0.0
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        prompts, labels = batch
        input_ids = cls_batch_preprocess(prompts, tokenizer)
        input_ids = input_ids.to(device)
        # next token prediction acc: let the model perform one forward pass, and directly if the next toekn is correct        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            predicted_token_ids = torch.argmax(outputs, dim=-1)
        next_token_ids = labels.to(device)
        correct_predictions = (predicted_token_ids == next_token_ids).sum().item()
        correct += correct_predictions
        ttl += next_token_ids.numel()
    result = {
        'correct': correct,
        'ttl': ttl
    }
    return result


class clsModel(nn.Module):
    def __init__(self, transformer, num_labels, enc_dec=False):
        super(clsModel, self).__init__()
        self.backbone = transformer
        # Assuming that the avg_pool is meant to be an average pooling layer over the sequence dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.enc_dec = enc_dec

    def forward(self, input_ids):
        if self.enc_dec:
            decoder_input_ids = torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long).to(input_ids.device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--model_name_or_path", type=str, default='./model/gpt2')
    parser.add_argument("--ckpt_path", type=str, default=None, help='the path to the full model checkpoint (pytorch_model.bin)')
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='./data/sentiment_data/')
    parser.add_argument("--output_dir", type=str, default='./output/tmp/')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--accumulation_steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--grad_norm', type=float, default=5.0)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)

    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    parser.add_argument('--simple_prompt', type=bool, default=False)
    parser.add_argument('--grad_ckpt', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--src_label', action='store_true')
    parser.add_argument('--enc_dec', type=bool, default=False)
    parser.add_argument('--lr_scheduler', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--default_prefix', type=str, default='null')
    parser.add_argument('--dataset', type=str, default='sentiment', choices=['sentiment', 'amazon']) 
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, 'log.log'))
    logger.info('arguments: {}'.format(args.__repr__()))

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    utils.seed_everything(args.seed+utils.get_rank())


    scaler = None
    enc_dec_backbone = False

    if 't5' in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if 'gpt2' in args.model_name_or_path.lower():
        transformer = AutoModel.from_pretrained(args.model_name_or_path)
    elif 't5' in args.model_name_or_path.lower():
        transformer = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        enc_dec_backbone = True
    elif 'bart' in args.model_name_or_path.lower():
        transformer = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        args.enc_dec = True
    else:
        transformer = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True)
    if args.fp16:
        transformer = transformer.half()
    if args.grad_ckpt:
        transformer.supports_gradient_checkpointing = True
        transformer.gradient_checkpointing_enable()
    transformer.enable_input_require_grads()

    if args.enc_dec == False:
        logger.info("Left padding for decoder-only models")
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info('add tokenizer')

    if args.dataset == 'sentiment':
        model = clsModel(transformer, num_labels=2, enc_dec=enc_dec_backbone)
    else:
        model = clsModel(transformer, num_labels=3, enc_dec=enc_dec_backbone)

    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
        logger.info("Loaded weights from {}".format(args.ckpt_path))

    model = model.to(device)
    model_without_ddp = model
    if args.distributed and (not args.evaluate):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    if args.train_data_dir is None:
        args.train_data_dir = args.data_dir

    if args.dataset == 'sentiment':
        train_dataset  = SentimentDataset(os.path.join('../data/sentiment_data', 'filtered100_central_train.json'))
        test_dataset = SentimentDataset(os.path.join('../data/sentiment_data', 'filtered100_central_test.json'))
    else:
        train_dataset  = AmazonDataset(os.path.join('../data/amazon_data', 'client_train.json'))
        test_dataset = AmazonDataset(os.path.join('../data/amazon_data', 'client_test.json'))     

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, test_dataset], [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]
    train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers, batch_size=[args.batch_size, args.batch_size],
        num_workers=[args.num_workers, args.num_workers], is_trains=[True, False], collate_fns=[None, None])

    loss_func = nn.CrossEntropyLoss()
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
            train(model, epoch=epoch, loss_func=loss_func, train_loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler, args=args, device=device)
        try:
            results = evaluate(model_without_ddp, test_loader, tokenizer, device, args)
            result_file = os.path.join(args.output_dir, 'pred_epoch{}_rank{}.json'.format(epoch, utils.get_rank()))
            json.dump(results, open(result_file, 'w'), ensure_ascii=False)
            if dist.is_initialized():
                dist.barrier()
            if utils.is_main_process():
                all_results = []
                for rank in range(utils.get_world_size()):
                    result_file = os.path.join(args.output_dir, 'pred_epoch{}_rank{}.json'.format(epoch, rank))
                    with open(result_file, 'r') as fr:
                        result = json.load(fr)
                    all_results.append(result)

                correct, total = 0, 0
                for res in all_results:
                    total += res['ttl']
                    correct += res['correct']
                eval_acc = correct / total
                logger.info(f'Epoch: {epoch}, Eval Accuracy: {correct / total}') 
                save_name = f'model_weight_epoch{epoch}.pth'
                ckpt_path = os.path.join(args.output_dir, save_name)
                model_without_ddp.save_pretrained(ckpt_path)
                logger.info(f'''save model to: {ckpt_path}''')
        except:
            logger.exception('error in evaluation')
            ckpt_path = os.path.join(args.output_dir, 'gpt2_weight.pth')
            logger.info(f'''save latest model to: {ckpt_path}''')
            model_without_ddp.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

        if args.evaluate:
            break
        if dist.is_initialized():
            dist.barrier()