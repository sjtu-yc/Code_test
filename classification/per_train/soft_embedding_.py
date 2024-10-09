import torch
import torch.nn as nn
import pickle
from torch.distributions.beta import Beta
    

class SoftClientEmbedding(nn.Module):
    '''
        beta distribution embedding of multiple clients
    '''
    def __init__(self, 
                num_clients: int,
                wte: nn.Embedding,
                n_tokens: int = 5,
                fp16 = False,
                init_path: str = None):

        super(SoftClientEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        # self.alpha = [nn.parameter.Parameter(torch.FloatTensor(n_tokens, wte.weight.size(1)), requires_grad=False) for _ in range(num_clients)]
        # self.beta = [nn.parameter.Parameter(torch.FloatTensor(n_tokens, wte.weight.size(1)), requires_grad=False) for _ in range(num_clients)]
        self.alpha = [torch.FloatTensor(n_tokens, wte.weight.size(1)) for _ in range(num_clients)]
        self.beta = [torch.FloatTensor(n_tokens, wte.weight.size(1)) for _ in range(num_clients)]
        self.new_user = torch.zeros(num_clients)

        if init_path is not None:
            for i in range(num_clients):
                f_name = f'{init_path}/client{i+1}.pickle'
                try:
                    with open(f_name, 'rb') as f:
                        client_prefix = pickle.load(f)
                    self.alpha[i].data = client_prefix['alpha'].cpu()
                    self.beta[i].data = client_prefix['beta'].cpu()
                except:
                    print(f'Error loading {f_name}')
                    self.alpha[i].data = self.alpha[1].data.clone()
                    self.beta[i].data = self.beta[1].data.clone()

        self.alphas = torch.stack(self.alpha)
        self.betas = torch.stack(self.beta)
        self.fp16 = fp16

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        client_idx = tokens[:, 0]
        beta_dists = Beta(self.alphas, self.betas)
        # sample_prefix = beta_dists.rsample().to(input_embedding.device)
        sample_prefix = beta_dists.sample().to(input_embedding.device)
        # new user sample_prefix to zero
        sample_prefix[self.new_user==1] = 0.0
        prefix = sample_prefix[client_idx-1]
        if self.fp16:
            prefix = prefix.to(torch.bfloat16)
        # return torch.cat([prefix, input_embedding], 1)
        return torch.cat([input_embedding, prefix], 1)


class SoftClientEmbedding_gaussian(nn.Module):
    '''
        gaussian distribution embedding of multiple clients
    '''
    def __init__(self, 
                num_clients: int,
                wte: nn.Embedding,
                n_tokens: int = 5,
                fp16 = False,
                small_std_ratio = 1,
                init_path: str = None):

        super(SoftClientEmbedding_gaussian, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.avg = [torch.FloatTensor(n_tokens, wte.weight.size(1)) for _ in range(num_clients)]
        self.var = [torch.ones(n_tokens, wte.weight.size(1)) for _ in range(num_clients)]
        self.new_user = torch.zeros(num_clients)

        if init_path is not None:
            for i in range(num_clients):
                f_name = f'{init_path}/client{i+1}.pickle'
                try:
                    with open(f_name, 'rb') as f:
                        client_prefix = pickle.load(f)
                    self.avg[i].data = client_prefix['avg'].cpu()
                    # self.var[i].data = client_prefix['beta'].cpu()
                except:
                    print(f'Error loading {f_name}')
                    self.avg[i].data = self.avg[1].data.clone()
                    # self.var[i].data = self.var[1].data.clone()
        self.avgs = torch.stack(self.avg)
        self.vars = torch.stack(self.var)
        for i in range(int(small_std_ratio*num_clients)):
            self.vars[i] = 0.2 * self.vars[i]
        print(self.avgs)
        self.fp16 = fp16

    def forward(self, tokens):
        # print(tokens.shape)
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        client_idx = tokens[:, 0]
        # print(client_idx)
        # sample_prefix = beta_dists.rsample().to(input_embedding.device)
        sample_prefix = torch.normal(mean=self.avgs, std=self.vars).to(input_embedding.device)
        prefix = sample_prefix[client_idx-1]
        if self.fp16:
            prefix = prefix.to(torch.bfloat16)
        return torch.cat([input_embedding, prefix], 1)
