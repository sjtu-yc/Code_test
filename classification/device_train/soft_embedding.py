import torch
import torch.nn as nn
import pickle
from torch.distributions.beta import Beta


class SoftSingleEmbedding(nn.Module):
    '''
        Gaussian prefix for a single client
    '''
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 5, 
                init_path = None,
                random_range: float = 0.01):
        super(SoftSingleEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.initialize_embedding(wte, n_tokens, random_range, init_path)
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 5, 
                             random_range: float = 0.43, 
                             init_path= None):
        self.avg = nn.parameter.Parameter(torch.full((n_tokens, wte.weight.size(1)), 0.1)) # initialize with 0.1 for t5
        self.var = torch.ones(n_tokens, wte.weight.size(1)) * 0.2
        if init_path is not None:
            with open(init_path, 'rb') as f:
                client_prefix = pickle.load(f)
            self.avg.data = client_prefix['avg']       
            
    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        sample = torch.normal(0, 1, size=(tokens.shape[0], self.n_tokens, self.wte.weight.size(1))).to(self.avg.device)
        prefix = sample * self.var.unsqueeze(0).to(self.avg.device) + self.avg.unsqueeze(0)
        return torch.cat([input_embedding, prefix], 1)

    def apply_regulation(self):
        # calculate the diff between (avg, var) and (0, 1)
        regulation_loss = torch.log(1/(abs(self.var) + 1e-6) + 1e-6) + (self.avg * self.avg) / 2 + self.var * self.var / 2
        return regulation_loss.mean()


class SoftSingleEmbedding_beta(nn.Module):
    '''
        Beta prefix for a single client
    '''
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 5,
                init_path=None):
        super(SoftSingleEmbedding_beta, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.initialize_embedding(wte, n_tokens, init_path)
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 5,
                             init_path=None):
        self.alpha = nn.parameter.Parameter(torch.full((n_tokens, wte.weight.size(1)), 2.0))
        self.beta = nn.parameter.Parameter(torch.full((n_tokens, wte.weight.size(1)), 5.0))

        if init_path is not None:
            with open(init_path, 'rb') as f:
                client_prefix = pickle.load(f)
            self.alpha.data = client_prefix['alpha']
            self.beta.data = client_prefix['beta']
            
    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])

        beta_dists = Beta(self.alpha, self.beta)
        prefix = beta_dists.rsample((tokens.shape[0],))
        prefix = prefix.to(input_embedding.device)
        return torch.cat([input_embedding, prefix], 1)
