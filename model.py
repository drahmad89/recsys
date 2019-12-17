import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class NMF(nn.Module):
    def __init__(self, user_len, item_len, latent_dim, processor='cpu'):
        super(NMF, self).__init__()
        device = torch.device(processor)
        self.user_w = Variable(torch.rand(user_len, latent_dim,device=device,dtype=torch.float)
, requires_grad = True)
        self.item_w = Variable(torch.rand(item_len, latent_dim,device=device,dtype=torch.float)
, requires_grad = True)



        return torch.dot(self.user_w[user_idx,:], self.item_w[item_idx,:])


latent_dim = 10
NMD_model = NMF(table.shape[0], table.shape[1], latent_dim)
optimzer=optim.adam(params=[NMD_model.user_w, NMD_model.item_w , lr=0.01, weight_decay=0.001)
total_loss = 0

i=0
report_every = 20

for idx in train_loader:

    user_idx = idx/users_len
    item_idx = idx - user_idx
    scores = table[user_idx, item_idx]
    pred = NMD_model(user_idx, item_idx)
    loss = 0.5*(pred - scores)^2
    loss.backward()
    optimize.step()
    total_loss += loss.detach()
    i += 1
    if i  % report_every ==0 :
        print('Pekabu')





