import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configs as cfg
from torch.utils.data import Dataset
from dataloading import data_loading, Dataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2)


class MF(nn.Module):
    def __init__(self, user_len, item_len, latent_dim=50, processor='cuda'):
        super(MF, self).__init__()
        device = torch.device(processor)
        self.user_w = Variable(torch.rand(user_len, latent_dim,device=device,dtype=torch.float)
        , requires_grad = True)
        self.item_w = Variable(torch.rand(item_len, latent_dim,device=device,dtype=torch.float)
        , requires_grad = True)
        return

    def forward(self, user_idx, item_idx ):
        return  torch.einsum('bs,bs->b',self.user_w[user_idx,:] ,self.item_w[item_idx,:]).type(torch.DoubleTensor).cuda() 

class NMF(nn.Module):
    def __init__(self, user_len, item_len, loss, embedding_layers=[20, 20, 15],
                 fusion_layer=[30 , 15 , 10], latent_dim=20,
                 processor='cuda', dropout = 0):
        super(NMF, self).__init__()
        self.loss = loss

        if loss =='mse_loss':
            self.logits_size = 1
        elif loss =='cross_entropy':
            self.logits_size = cfg.SCORE_SIZE
        device = torch.device(processor)
        self.user_w = Variable(torch.rand(user_len,
                                          latent_dim,device=device,dtype=torch.float32)
        , requires_grad = True)
        self.item_w = Variable(torch.rand(item_len,
                                          latent_dim,device=device,dtype=torch.float32)
        , requires_grad = True)

        user_embeddings = []
        for i, emb in  enumerate(embedding_layers):

            if not i :
                user_embeddings.append(torch.nn.Linear(latent_dim, emb))
            else:
                user_embeddings.append(torch.nn.Linear(embedding_layers[i-1], emb))

            user_embeddings.append(torch.nn.ReLU())
            user_embeddings.append(torch.nn.BatchNorm1d(emb))
            user_embeddings.append(torch.nn.Dropout(dropout))
        item_embeddings = user_embeddings
        self.user_embeddings = torch.nn.Sequential(*user_embeddings)
        self.item_embeddings = torch.nn.Sequential(*item_embeddings)

        fusion_net = []
        for i, lay in enumerate(fusion_layer):
            if not i:
                fusion_net.append(torch.nn.Linear(2 * emb, lay))
            else:
                fusion_net.append(torch.nn.Linear(fusion_layer[i-1], lay))
            fusion_net.append(torch.nn.BatchNorm1d(lay))
            fusion_net.append(torch.nn.Dropout(dropout))
        fusion_net.append(torch.nn.Linear(lay, self.logits_size))
        fusion_net.append(torch.nn.Sigmoid())
        self.fusion_net = torch.nn.Sequential(*fusion_net)


    def forward(self, user_idx, item_idx):
        user_x = self.user_w[user_idx]
        item_x = self.item_w[item_idx]
        user_y = self.user_embeddings(user_x)
        item_y = self.item_embeddings(item_x)
        return self.fusion_net(torch.cat((item_y, user_y), 1)) * \
    (1 if self.loss == 'cross_entropy' else cfg.SCORE_SCALE)
