import numpy as np
import torch 
import warnings
import argparse
import configparser
from dataloading import data_loading
from train import train
from model import NMF, MLP, GMF
import configs as cfg


def main(direc):
    table, train_loader, valid_loader=data_loading(direc)
    mlp_model = MLP(table.shape[0], table.shape[1]).cuda().float()
    gmf_model = GMF(table.shape[0], table.shape[1]).cuda().float()
    model = NMF(mlp_model, gmf_model).cuda().float()
    train(direc, model, train_loader, valid_loader, loss_fct
          =cfg.LOSS).training()
    return


if   __name__ == "__main__":
    warnings.simplefilter(action='ignore',category=FutureWarning)
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("data_file", help='path to file name', type=str)
    args = parser.parse_args()
    main(args.data_file)
