import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from dataloading import data_loading, Dataset
from torch.autograd import Variable

class NMF(nn.Module):
    def __init__(self, user_len, item_len, latent_dim, processor='cpu'):
        super(NMF, self).__init__()
        device = torch.device(processor)
        self.user_w = Variable(torch.rand(user_len, latent_dim,device=device,dtype=torch.float)
        , requires_grad = True)
        self.item_w = Variable(torch.rand(item_len, latent_dim,device=device,dtype=torch.float)
        , requires_grad = True)
        return

    def forward(self, user_idx, item_idx ):
        return  torch.einsum('bs,bs->b',self.user_w[user_idx,:] ,self.item_w[item_idx,:]).type(torch.DoubleTensor).cuda() 

def train(table,train_loader, loss_fct = 'mse_loss'):
    latent_dim = 10
    NMF_model = NMF(table.shape[0], table.shape[1], latent_dim)
    optimzer=optim.Adam(params=[NMF_model.user_w, NMF_model.item_w] , lr=0.0001, weight_decay=0.001)
    best_loss = np.inf
    total_loss = 0
    i=0
    report_every = 20

    EPOCHS=10
    for epoch in range(EPOCHS):
        running_loss = 0.0
        time_step_count = 0

        for idx in train_loader:

            pred = NMF_model(idx[:,0].type(torch.LongTensor),
                                              idx[:,1].type(torch.LongTensor))
            loss = getattr(F,loss_fct)(pred, idx[:,2])
            #loss = 0.5*(pred - idx[:,2].type(torch.DoubleTensor).cuda())**2
            loss.backward()
            optimzer.step()
            #total_loss += loss.detach()
            total_samples = idx.numel()
            running_loss += loss.detach().item() * total_samples
            time_step_count += total_samples
            i += 1
            if i  % report_every ==0 :
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / time_step_count))
                print("REPORTING")
                with torch.no_grad():
                    print(loss.mean().detach())
                    total_loss = 0.
                    count = 0
                    for idx in test_loader:
                        pred = NMF_model(idx[:,0].type(torch.LongTensor),
                                         idx[:,1].type(torch.LongTensor))
                        loss = getattr(F,loss_fct)(pred, idx[:,2])
                        total_loss += loss.item() 
                        count += 1
                        valid_loss = total_loss / count
                        if valid_loss < best_loss:
                            print("Best valid loss:", valid_loss)
                            with open('nmd_model.pt', 'wb') as f:
                                torch.save(NMF_model, f)
                            best_loss = valid_loss
                        else:
                            print("Valid loss:", valid_loss)


    return

table_pivot, train_loader, test_loader = data_loading()
train(table_pivot, train_loader)




