import torch
import torch.nn as nn
import pickle
from torch.distributions.beta import Beta


class SoftSingleEmbedding(nn.Module):
    '''
        gaussian
    '''
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 5, 
                random_range: float = 0.01):
        super(SoftSingleEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.initialize_embedding(wte, n_tokens, random_range)
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 5, 
                             random_range: float = 0.05):
        self.avg = nn.parameter.Parameter(torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range))
        self.var = nn.parameter.Parameter(torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(1.25, 1.5))
            
    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        sample = torch.normal(0, 1, size=(tokens.shape[0], self.n_tokens, self.wte.weight.size(1))).to(self.var.device)
        prefix = sample * self.var.unsqueeze(0) + self.avg.unsqueeze(0)
        return torch.cat([prefix, input_embedding], 1)

    def apply_regulation(self):
        regulation_loss = torch.log(1/(abs(self.var) + 1e-6) + 1e-6) + (self.avg * self.avg) / 2 + self.var * self.var / 2
        return regulation_loss.mean()


class SoftSingleEmbedding_beta(nn.Module):
    '''
        beta
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
        self.alpha = nn.parameter.Parameter(torch.full((n_tokens, wte.weight.size(1)), 5.0))
        self.beta = nn.parameter.Parameter(torch.full((n_tokens, wte.weight.size(1)), 6.0))

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
        return torch.cat([prefix, input_embedding], 1)
