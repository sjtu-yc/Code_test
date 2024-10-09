import torch
import torch.nn as nn
import pickle
from torch.distributions.beta import Beta
    

class SoftClientEmbedding(nn.Module):
    def __init__(self, 
                num_clients: int,
                wte: nn.Embedding,
                n_tokens: int = 5, 
                init_path: str = None):

        super(SoftClientEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.alpha = [torch.FloatTensor(n_tokens, wte.weight.size(1)) for _ in range(num_clients)]
        self.beta = [torch.FloatTensor(n_tokens, wte.weight.size(1)) for _ in range(num_clients)]
        self.sample_flag = True
        self.sample_prefix = None

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
                    self.alpha[i].data = torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(1, 2.5)
                    self.beta[i].data = torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(4, 5.5)

        self.alphas = torch.stack(self.alpha)
        self.betas = torch.stack(self.beta)

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        client_idx = tokens[:, 0]
        beta_dists = Beta(self.alphas, self.betas)
        if self.sample_flag:
            sample_prefix = beta_dists.sample().to(input_embedding.device)
            self.sample_prefix = sample_prefix
            self.sample_flag = False
        else:
            sample_prefix = self.sample_prefix
        prefix = sample_prefix[client_idx-1]
        return torch.cat([prefix, input_embedding], 1)
