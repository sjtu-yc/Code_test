import torch
import os
import pickle
from loguru import logger
from copy import deepcopy
from torch.utils.data import DataLoader
from data.dataset import simple_collate, batch_preprocess
import json
from utils import get_nb_trainable_parameters
from soft_embedding import SoftSingleEmbedding, SoftSingleEmbedding_beta
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class ClientDist:
    def __init__(self, model, tokenizer, train_set, valid_set, rank, client_num, args) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        if rank >= 0:
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = torch.device('cpu')
        self.client_num = client_num
        self.train_set, self.valid_set = train_set, valid_set
        self.args = args
        self.prompt = SoftSingleEmbedding(model.get_input_embeddings(), n_tokens=self.args.n_tokens) 
        logger.add(os.path.join(args.output_dir, 'log.log'))
    
    def save_prompt(self):
        # prompt_dic = {'avg':self.prompt.avg.data, 'var':self.prompt.var.data}
        prompt_dic = {'alpha':self.prompt.alpha.data, 'beta':self.prompt.beta.data}
        pickle.dump(prompt_dic, open(os.path.join(self.args.output_dir, f'clients/client{self.client_num}.pickle'), 'wb'))

    def evaluate(self, model, test_loader, save_path=None):
        model.to(self.device)
        model.eval()
        eval_preds = []
        eval_labels = []
        eval_inputs, eval_contexts = [],[]
        # for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        for _, batch in enumerate(test_loader):
            prompts, targets, inputs = batch
            input_ids = self.tokenizer(prompts, return_tensors='pt', padding='longest', max_length=50, truncation=True).input_ids
            input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1).to(self.device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_tokens=self.args.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                                            return_dict_in_generate=True, output_scores=True, num_return_sequences=1, use_cache=False)
            batch_preds = self.tokenizer.batch_decode(outputs.sequences[:,len(input_ids[0]):].detach().cpu().numpy(), skip_special_tokens=True)
            batch_preds = [pred.strip().replace(' ', '') for pred in batch_preds]
            eval_preds.extend(batch_preds[::1])
            eval_labels.extend(targets)
            eval_inputs.extend(inputs)
            eval_contexts.extend(prompts)
        result = [{'label': label, 'pred': pred.strip().replace(' ',''), 'input': input_pinyin, 'context': context}
                    for label, pred, input_pinyin, context in zip(eval_labels, eval_preds, eval_inputs, eval_contexts)]
        if save_path:
            with open(save_path, 'w', encoding='utf8') as fw:
                for pred in result:
                    fw.write(json.dumps(pred, ensure_ascii=False) + '\n')
        correct, total = 0.0, 0.0
        for res in result:
            total += 1 
            pred  = res['pred']
            target = res['label']
            if target.strip() == pred.strip():
                correct += 1
        if total == 0.0:
            return 0
        eval_acc = correct / total
        # logger.info(f'{eval_acc}')
        model.cpu()
        return eval_acc
    
    def evaluate_loss(self, model, test_loader):
        model.to(self.device)
        model.eval()
        total_loss = 0.
        for _, batch in enumerate(test_loader):
            prompts, targets, _ = batch
            input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
            input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1)
            labels = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), -100), labels], dim=1)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            loss = model(input_ids=input_ids, labels=labels).loss
            total_loss += loss.detach().float()
        eval_epoch_loss = total_loss/len(test_loader)
        model.cpu()
        return eval_epoch_loss.cpu().item()

    def train(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        model = deepcopy(self.model)
        model.set_input_embeddings(self.prompt)
        para, all_param = get_nb_trainable_parameters(model)
        logger.info(f"trainable params: {para:,d}, trainable: {100 * para / all_param:.4f}%")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
        best_acc = 0.0
        test_loader = DataLoader(dataset=self.valid_set, batch_size=self.args.batch_size_eval, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        tmp_acc = self.evaluate(model, test_loader)
        logger.info(f'Start point: Client {self.client_num}: valid acc: {tmp_acc}')
        # for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        for epoch in range(self.args.client_epochs):
            model.train()
            model.to(self.device)
            total_loss = 0.
            for step, batch in enumerate(train_loader):
                prompts, targets, _ = batch
                input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
                input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1)
                labels = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), -100), labels], dim=1)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss = model(input_ids=input_ids, labels=labels).loss
                # regulation_loss = self.prompt.apply_regulation()
                # print(f'loss: {loss}, regulation loss: {regulation_loss}')
                # exit()
                # loss = loss + regulation_loss
                if torch.isnan(loss): # skip the batch if loss is nan
                    logger.info('loss is nan')
                    continue
                total_loss += loss.detach().float()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            train_epoch_loss = total_loss/len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            logger.info(f'Client {self.client_num}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')
            test_loader = DataLoader(dataset=self.valid_set, batch_size=self.args.batch_size_eval, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
            tmp_acc = self.evaluate(model, test_loader)
            logger.info(f'Epoch {epoch}: Client {self.client_num}: valid acc: {tmp_acc}')
            self.prompt = model.get_input_embeddings()
            if tmp_acc > best_acc:
                best_acc = tmp_acc
                self.save_prompt()
                logger.info(f'Client {self.client_num}: best acc update to {best_acc}')
        torch.cuda.empty_cache()
        model.cpu()
        # self.save_prompt()
        return train_epoch_loss.cpu().item()
    
class ClientDist_beta:
    def __init__(self, model, tokenizer, train_set, valid_set, rank, client_num, init_beta_path, args) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        if rank >= 0:
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = torch.device('cpu')
        self.client_num = client_num
        self.train_set, self.valid_set = train_set, valid_set
        self.args = args
        if init_beta_path is None:
            self.prompt = SoftSingleEmbedding_beta(model.get_input_embeddings(), n_tokens=self.args.n_tokens)
        else:
            self.prompt = SoftSingleEmbedding_beta(model.get_input_embeddings(), n_tokens=self.args.n_tokens, init_path=init_beta_path)
        logger.add(os.path.join(args.output_dir, 'log.log'))
    
    def save_prompt(self):
        # prompt_dic = {'avg':self.prompt.avg.data, 'var':self.prompt.var.data}
        prompt_dic = {'alpha':self.prompt.alpha.data, 'beta':self.prompt.beta.data}
        pickle.dump(prompt_dic, open(os.path.join(self.args.output_dir, f'clients/client{self.client_num}.pickle'), 'wb'))

    def evaluate(self, model, test_loader, save_path=None):
        model.to(self.device)
        model.eval()
        eval_preds = []
        eval_labels = []
        eval_inputs, eval_contexts = [],[]
        # for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        for _, batch in enumerate(test_loader):
            prompts, targets, inputs = batch
            input_ids = self.tokenizer(prompts, return_tensors='pt', padding='longest', max_length=50, truncation=True).input_ids
            input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1).to(self.device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_tokens=self.args.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                                            return_dict_in_generate=True, output_scores=True, num_return_sequences=1, use_cache=False)
            batch_preds = self.tokenizer.batch_decode(outputs.sequences[:,len(input_ids[0]):].detach().cpu().numpy(), skip_special_tokens=True)
            batch_preds = [pred.strip().replace(' ', '') for pred in batch_preds]
            eval_preds.extend(batch_preds[::1])
            eval_labels.extend(targets)
            eval_inputs.extend(inputs)
            eval_contexts.extend(prompts)
        result = [{'label': label, 'pred': pred.strip().replace(' ',''), 'input': input_pinyin, 'context': context}
                    for label, pred, input_pinyin, context in zip(eval_labels, eval_preds, eval_inputs, eval_contexts)]
        if save_path:
            with open(save_path, 'w', encoding='utf8') as fw:
                for pred in result:
                    fw.write(json.dumps(pred, ensure_ascii=False) + '\n')
        correct, total = 0.0, 0.0
        for res in result:
            total += 1 
            pred  = res['pred']
            target = res['label']
            if target.strip() == pred.strip():
                correct += 1
        if total == 0.0:
            return 0
        eval_acc = correct / total
        # logger.info(f'{eval_acc}')
        model.cpu()
        return eval_acc
    
    def evaluate_loss(self, model, test_loader):
        model.to(self.device)
        model.eval()
        total_loss = 0.
        for _, batch in enumerate(test_loader):
            prompts, targets, _ = batch
            input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
            input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1)
            labels = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), -100), labels], dim=1)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            loss = model(input_ids=input_ids, labels=labels).loss
            total_loss += loss.detach().float()
        eval_epoch_loss = total_loss/len(test_loader)
        model.cpu()
        return eval_epoch_loss.cpu().item()

    def train(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        model = deepcopy(self.model)
        model.set_input_embeddings(self.prompt)
        para, all_param = get_nb_trainable_parameters(model)
        logger.info(f"trainable params: {para:,d}, trainable: {100 * para / all_param:.4f}%")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
        best_acc = 0.0
        # for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        for epoch in range(self.args.client_epochs):
            model.train()
            model.to(self.device)
            total_loss = 0.
            # for step, batch in enumerate(train_loader):
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                prompts, targets, _ = batch
                input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
                input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1)
                labels = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), -100), labels], dim=1)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss = model(input_ids=input_ids, labels=labels).loss
                if torch.isnan(loss): # skip the batch if loss is nan
                    logger.info('loss is nan')
                    continue
                total_loss += loss.detach().float()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            train_epoch_loss = total_loss/len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            logger.info(f'Client {self.client_num}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')
            test_loader = DataLoader(dataset=self.valid_set, batch_size=self.args.batch_size_eval, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
            tmp_acc = self.evaluate(model, test_loader)
            logger.info(f'Epoch {epoch}: Client {self.client_num}: valid acc: {tmp_acc}')
            self.prompt = model.get_input_embeddings()
            if tmp_acc > best_acc:
                best_acc = tmp_acc
                self.save_prompt()
                logger.info(f'Client {self.client_num}: best acc update to {best_acc}')
        torch.cuda.empty_cache()
        model.cpu()
        # self.save_prompt()
        return train_epoch_loss.cpu().item()
    
    def pure_train(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        model = deepcopy(self.model)
        model.set_input_embeddings(self.prompt)
        para, all_param = get_nb_trainable_parameters(model)
        logger.info(f"trainable params: {para:,d}, trainable: {100 * para / all_param:.4f}%")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
        best_loss = 999.0
        # for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        for epoch in range(self.args.client_epochs):
            model.train()
            model.to(self.device)
            total_loss = 0.
            # for step, batch in enumerate(train_loader):
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                prompts, targets, _ = batch
                input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
                input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1)
                labels = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), -100), labels], dim=1)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss = model(input_ids=input_ids, labels=labels).loss
                if torch.isnan(loss): # skip the batch if loss is nan
                    logger.info('loss is nan')
                    continue
                total_loss += loss.detach().float()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            train_epoch_loss = total_loss/len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            logger.info(f'Client {self.client_num}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')
            model.cpu()
            self.prompt = model.get_input_embeddings()
            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
                self.save_prompt()
                logger.info(f'Client {self.client_num}: best train epoch loss reduce to {best_loss}')
        torch.cuda.empty_cache()
        model.cpu()
        return train_epoch_loss.cpu().item()
    
    def train_eval_loss(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        model = deepcopy(self.model)
        model.set_input_embeddings(self.prompt)
        para, all_param = get_nb_trainable_parameters(model)
        logger.info(f"trainable params: {para:,d}, trainable: {100 * para / all_param:.4f}%")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=0.0*self.args.weight_decay, eps=self.args.adam_epsilon)
        tot_steps = len(train_loader) * self.args.client_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=tot_steps
        )
        best_loss = 999.0
        # for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        for epoch in range(self.args.client_epochs):
            model.train()
            model.to(self.device)
            total_loss = 0.
            # for step, batch in enumerate(train_loader):
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                prompts, targets, _ = batch
                input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
                input_ids = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), 1), input_ids], dim=1)
                labels = torch.cat([torch.full((input_ids.shape[0], self.args.n_tokens), -100), labels], dim=1)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss = model(input_ids=input_ids, labels=labels).loss
                if torch.isnan(loss): # skip the batch if loss is nan
                    logger.info('loss is nan')
                    continue
                total_loss += loss.detach().float()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_epoch_loss = total_loss/len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            logger.info(f'Client {self.client_num}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')

            test_loader = DataLoader(dataset=self.valid_set, batch_size=self.args.batch_size_eval, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
            tmp_loss = self.evaluate_loss(model, test_loader)
            logger.info(f'Epoch {epoch}: Client {self.client_num}: valid loss: {tmp_loss}')
            self.prompt = model.get_input_embeddings()
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                self.save_prompt()
                logger.info(f'Client {self.client_num}: best valid loss reduce to {best_loss}')
        torch.cuda.empty_cache()
        model.cpu()
        return train_epoch_loss.cpu().item()


class Client_self_train:
    def __init__(self, model, tokenizer, train_set, valid_set, rank, client_num, args) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        if rank >= 0:
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = torch.device('cpu')
        self.client_num = client_num
        self.train_set, self.valid_set = train_set, valid_set
        self.args = args

    def evaluate(self, model, test_loader, save_path=None):
        model.to(self.device)
        model.eval()
        eval_preds = []
        eval_labels = []
        eval_inputs, eval_contexts = [],[]
        # for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        for _, batch in enumerate(test_loader):
            prompts, targets, inputs = batch
            input_ids = self.tokenizer(prompts, return_tensors='pt', padding='longest', max_length=50, truncation=True).input_ids
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_tokens=self.args.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                                            return_dict_in_generate=True, output_scores=True, num_return_sequences=1)
            batch_preds = self.tokenizer.batch_decode(outputs.sequences[:,len(input_ids[0]):].detach().cpu().numpy(), skip_special_tokens=True)
            batch_preds = [pred.strip().replace(' ', '') for pred in batch_preds]
            eval_preds.extend(batch_preds[::1])
            eval_labels.extend(targets)
            eval_inputs.extend(inputs)
            eval_contexts.extend(prompts)
        result = [{'label': label, 'pred': pred.strip().replace(' ',''), 'input': input_pinyin, 'context': context}
                    for label, pred, input_pinyin, context in zip(eval_labels, eval_preds, eval_inputs, eval_contexts)]
        if save_path:
            with open(save_path, 'w', encoding='utf8') as fw:
                for pred in result:
                    fw.write(json.dumps(pred, ensure_ascii=False) + '\n')
        correct, total = 0.0, 0.0
        for res in result:
            total += 1 
            pred  = res['pred']
            target = res['label']
            if target.strip() == pred.strip():
                correct += 1
        if total == 0.0:
            return 0
        eval_acc = correct / total
        # logger.info(f'{eval_acc}')
        model.cpu()
        return eval_acc
    
    def train_eval_acc(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        model = deepcopy(self.model)

        model.eval()
        model.to(self.device)
        test_loader = DataLoader(dataset=self.valid_set, batch_size=self.args.batch_size_eval, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
        eval_acc = self.evaluate(model, test_loader)
        logger.info(f'Init Model, Client: {self.client_num}, Testset Size: {len(self.valid_set)}, valid acc: {eval_acc}')

        para, all_param = get_nb_trainable_parameters(model)
        logger.info(f"trainable params: {para:,d}, trainable: {100 * para / all_param:.4f}%")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
        tot_steps = len(train_loader) * self.args.client_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=tot_steps
        )
        # for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        for epoch in range(self.args.client_epochs):
            model.train()
            model.to(self.device)
            total_loss = 0.
            # for step, batch in enumerate(train_loader):
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                prompts, targets, _ = batch
                input_ids, labels = batch_preprocess(prompts, targets, self.tokenizer)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss = model(input_ids=input_ids, labels=labels).loss
                if torch.isnan(loss): # skip the batch if loss is nan
                    logger.info('loss is nan')
                    continue
                total_loss += loss.detach().float()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            train_epoch_loss = total_loss/len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            logger.info(f'Client {self.client_num}, Train Loss: {train_epoch_loss}, Train PPL: {train_ppl}')
            test_loader = DataLoader(dataset=self.valid_set, batch_size=self.args.batch_size_eval, num_workers=self.args.num_workers, shuffle=True, collate_fn=simple_collate)
            eval_acc = self.evaluate(model, test_loader)
            logger.info(f'Epoch: {epoch}, Client: {self.client_num}, Testset Size: {len(self.valid_set)}, valid acc: {eval_acc}')
        torch.cuda.empty_cache()
        model.cpu()
        return train_epoch_loss.cpu().item()
