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

class MLP(nn.Module):
    def __init__(self, user_len, item_len, loss= cfg.LOSS, embedding_layers=[20, 20, 15],
                 fusion_layer=cfg.FUSION_LAYERS, mlp_latent_dim=cfg.MLP_LATENT_DIM,
                 processor='cuda', dropout = cfg.DROPOUT):
        super(MLP, self).__init__()
        self.loss = loss
        device = torch.device(processor)

        self.user_embedding_mlp = torch.nn.Embedding(num_embeddings=user_len,embedding_dim=mlp_latent_dim)
        self.item_embedding_mlp = torch.nn.Embedding(num_embeddings=item_len,embedding_dim=mlp_latent_dim)

        if loss =='mse_loss':
            self.logits_size = 1
        elif loss =='cross_entropy':
            self.logits_size = cfg.SCORE_SIZE

        user_feat_reps = []
        for i, emb in  enumerate(embedding_layers):

            if not i :
                user_feat_reps.append(torch.nn.Linear(mlp_latent_dim, emb))
            else:
                user_feat_reps.append(torch.nn.Linear(embedding_layers[i-1], emb))

            user_feat_reps.append(torch.nn.ReLU())
            user_feat_reps.append(torch.nn.BatchNorm1d(emb))
            user_feat_reps.append(torch.nn.Dropout(dropout))
        item_feat_reps = user_feat_reps
        self.user_feat_reps = torch.nn.Sequential(*user_feat_reps)
        self.item_feat_reps = torch.nn.Sequential(*item_feat_reps)

        fusion_net = []
        for i, lay in enumerate(fusion_layer):
            if not i:
                fusion_net.append(torch.nn.Linear(2 * mlp_latent_dim, lay))
            else:
                fusion_net.append(torch.nn.Linear(fusion_layer[i-1], lay))
            fusion_net.append(torch.nn.BatchNorm1d(lay))
            fusion_net.append(torch.nn.Dropout(dropout))
        self.fusion_net = torch.nn.Sequential(*fusion_net)


    def forward(self, user_idx, item_idx):
        x_fusion = torch.cat((self.user_embedding_mlp(user_idx),
                                 self.item_embedding_mlp(item_idx)), 1)
        return  self.fusion_net(x_fusion)

        #return self.fusion_net(torch.cat((item_y, user_y), 1)) * \
    #(1 if self.loss == 'cross_entropy' else cfg.SCORE_SCALE)


class GMF(nn.Module):
    def __init__(self, user_len, item_len, gmf_latent_dim=cfg.GMF_LATENT_DIM, processor='cuda'):
        super(GMF, self).__init__()

        self.user_embedding_gmf = torch.nn.Embedding(num_embeddings=user_len, embedding_dim=gmf_latent_dim)
        self.item_embedding_gmf = torch.nn.Embedding(num_embeddings=item_len, embedding_dim=gmf_latent_dim)

    def forward(self, user_idx, item_idx):
        return torch.mul(self.user_embedding_gmf(user_idx), self.item_embedding_gmf(item_idx))
class NMF(nn.Module):
    def __init__(self, mlp_module, gmf_module, output_size=cfg.LOGITS, loss = cfg.LOSS,
                 gmf_flag= cfg.GMF_FLAG):
        super(NMF, self).__init__()
        self.mlp_module = mlp_module
        self.gmf_module = gmf_module
        self.output_size = output_size
        self.loss = loss
        self.gmf_flag = gmf_flag
        self.prediction_lay = torch.nn.Linear(2*cfg.GMF_LATENT_DIM, cfg.LOGITS)
    def forward(self, user_idx, item_idx):
        mlp_output = self.mlp_module(user_idx, item_idx)
        if self.gmf_flag:
            gmf_output = self.gmf_module(user_idx, item_idx)
            last_layer = torch.cat((mlp_output, gmf_output), 1)
            logits = self.prediction_lay(last_layer)
            return torch.nn.Sigmoid()(logits) * \
                    (5 if self.loss=='cross_entropy' else 1)





