import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from dataloading import data_loading, Dataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from model import NMF





def train(table,
          train_loader,
          valid_loader,
          loss_fct = 'mse_loss',
          epoch_nb = 10,
          Lr=0.0001,
          Weight_decay = 0.00001
            ):
    model = NMF(table.shape[0], table.shape[1])
    optimzer=optim.Adam(params=[model.user_w, model.item_w] , lr=Lr,
                        weight_decay=Weight_decay)
    best_loss = np.inf
    total_loss = 0
    i=0
    report_every = 20

    writer = SummaryWriter("logs/MF/")

    for epoch in range(epoch_nb):
        running_loss = 0.0
        time_step_count = 1

        for idx in train_loader:

            pred = model(idx[:,0].type(torch.LongTensor),
                                              idx[:,1].type(torch.LongTensor))
            loss = getattr(F,loss_fct)(pred, idx[:,2])
            loss.backward()
            optimzer.step()
            total_samples = idx.numel()
            running_loss += loss.detach().item() * total_samples
            time_step_count += total_samples


            i += 1
            if i  % report_every ==0 :

                writer.add_scalar('training_loss',
                                  running_loss/time_step_count,
                                  epoch * len(train_loader) + i)
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / time_step_count))
                running_loss = 0.0
                time_step_count = 0
            if i % report_every * 10 ==0:
                print("REPORTING!!")
                model.eval()
                with torch.no_grad():
                    total_loss = 0.
                    count = 0
                    for idx in valid_loader:
                        pred = model(idx[:,0].type(torch.LongTensor),
                                         idx[:,1].type(torch.LongTensor))
                        loss = getattr(F,loss_fct)(pred, idx[:,2])
                        total_loss += loss.item() 
                        count += 1
                    valid_loss = total_loss / count
                    writer.add_scalar('valid_loss',
                                     valid_loss,
                                     epoch * len(valid_loader) +i)
                    if valid_loss < best_loss:
                        print("Best valid loss:", valid_loss)
                        with open('nmd_model.pt', 'wb') as f:
                            torch.save(model, f)
                        best_loss = valid_loss
                    else:
                        print("Valid loss:", valid_loss)
                model.train()

    return





