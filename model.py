import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from dataloading import data_loading, Dataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter





class NMF(nn.Module):
    def __init__(self, user_len, item_len, latent_dim=10, processor='cuda'):
        super(NMF, self).__init__()
        device = torch.device(processor)
        self.user_w = Variable(torch.rand(user_len, latent_dim,device=device,dtype=torch.float)
        , requires_grad = True)
        self.item_w = Variable(torch.rand(item_len, latent_dim,device=device,dtype=torch.float)
        , requires_grad = True)
        return

    def forward(self, user_idx, item_idx ):
        return  torch.einsum('bs,bs->b',self.user_w[user_idx,:] ,self.item_w[item_idx,:]).type(torch.DoubleTensor).cuda() 
