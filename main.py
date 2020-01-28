import numpy as np
import torch 
from dataloading import data_loading
from train import train
from model import NMF


def main():
    table_pivot, train_loader, valid_loader = data_loading()
    train(table_pivot, train_loader, valid_loader)
    return


if __name__ == "__main__":
    main()
