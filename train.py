
" Training module"



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from dataloading import data_loading, Dataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from model import NMF, MLP, GMF
from checkpoints import CheckPoints
import configs as cfg


class train():
    def __init__(self,direc, model, train_loader, valid_loader, loss_fct='mse_loss',
                 epoch_nb=cfg.EPOCHS, lr=0.0001, weight_decay=0.00001):
        self.direc = direc
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fct = loss_fct
        self.epoch_nb = epoch_nb
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model
        self.writer = SummaryWriter("logs/NMF/")
        self.file_name = direc.split("/")[-1].split(".")[0]


    def training(self):
        optimzer=optim.Adam(self.model.parameters(), lr=self.lr,
                            weight_decay=self.weight_decay)
        best_loss = np.inf
        total_loss = 0
        i=0
        report_every = 20
        ckpt_train = CheckPoints (self.file_name, cfg.MODEL_NAME,
                                   self.loss_fct, self.lr, 'Train')
        ckpt_train.create_file_writer()

        ckpt_valid = CheckPoints (self.file_name, cfg.MODEL_NAME,
                                   self.loss_fct, self.lr, 'Valid')
        ckpt_valid.create_file_writer()


        for epoch in range(self.epoch_nb):
            running_loss = 0.0
            running_acc = 0.0
            time_step_count = 1

            for idx in self.train_loader:

                pred = self.model(idx[:, 0].long().cuda(),
                             idx[:,1].long().cuda())
                if self.loss_fct =='cross_entropy':
                    loss = getattr(F,self.loss_fct)(pred, idx[:,2].long())
                elif self.loss_fct == 'mse_loss':
                    loss = getattr(F,self.loss_fct)(pred, idx[:,2].unsqueeze(1).float())
                loss.backward()
                optimzer.step()
                total_samples = idx.numel()
                max_vals, max_indices = torch.max(pred.round(),1)
                #if self.loss_fct ==('cross_entropy'):
                running_acc += (max_indices == \
                                    idx[:,2]).sum().data.cpu().numpy()/max_indices.size()[0]
                running_loss += loss.detach().item() * total_samples
                time_step_count += total_samples


                i += 1
                if i  % report_every ==0 :

                    self.writer.add_scalar('training_loss',
                                      running_loss/time_step_count,
                                      epoch * len(self.train_loader) + i)
                    print('[%d, %5d] loss: %.3f'%
                          (epoch, i, running_loss / time_step_count,
                           ))
                    print('[%d, %5d] acc:%.3f' % ( epoch, i,
                                                  running_acc/(time_step_count/
                                                           total_samples)))
                    self.writer.add_scalar('training_acc',
                                      running_acc/(time_step_count/total_samples),
                                      epoch * len(self.train_loader) + i)
                    ckpt_train.write_line( epoch,epoch * len(self.train_loader) + i,
                                    running_loss/time_step_count,
                                    running_acc/(time_step_count/total_samples))
 
                    running_loss = 0.0
                    running_acc = 0.0
                    time_step_count = 0
                if i % report_every * 100 == 0:
                    ########
                    continue
                    ########
                    valid_acc, valid_loss = self.validate(i, epoch)
                    if valid_loss < best_loss:
                        parser = "Best"
                        best_loss = valid_loss
                        with open('nmd_model.pt', 'wb') as f:
                                torch.save(self.model, f)
                    print("{} valid loss:{}".format(parser, valid_loss))
                    ckpt_valid.write_line(epoch, epoch*len(self.train_loader)
                                          +i, valid_loss, valid_acc)
                    self.writer.add_scalar('valid_loss',
                                      valid_loss,
                                      epoch * len(self.train_loader) + i)
                    self.writer.add_scalar('valid_acc', valid_acc,
                                        epoch * len(self.train_loader) + i)
                    print("{} valid acc:{}".format(parser, valid_acc))
                    parser = ""
        ckpt_train.file_writer.close()
        ckpt_valid.file_writer.close()
    def validate( self, i, epoch):
        running_acc = 0
        self.model.eval()
        with torch.no_grad():
            total_acc = 0
            total_loss = 0
            count = 0
        for i, idx in enumerate(self.valid_loader):
            pred = self.model(idx[:,0].long().cuda(),
                         idx[:,1].long().cuda())
            if self.loss_fct =='cross_entropy':
                loss = getattr(F,self.loss_fct)(pred, idx[:,2].long())
            elif self.loss_fct == 'mse_loss':
                loss = getattr(F,self.loss_fct)(pred, idx[:,2].unsqueeze(1).float())
            max_vals, max_indices = torch.max(pred.round(),1)
            running_acc += (max_indices == \
                    idx[:,2]).sum().data.cpu().numpy()/max_indices.size()[0]
            total_loss += loss.item()
            count +=1
            if i > cfg.MAX_VALID:
                break
        valid_loss = total_loss / count
        valid_acc = running_acc/ count
        self.model.train()
        return valid_acc, valid_loss

