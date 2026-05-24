import torch
import numpy as np
import argparse
from trainer import training
from data_loader import dataLoader
import os
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    same_seeds(4321)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TaFeng')
    parser.add_argument('--h1', type=int, default=5)
    parser.add_argument('--h2', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=1e-2)
    parser.add_argument('--decay', type=float, default=0.6)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--evalEpoch', type=int, default=1)
    parser.add_argument('--testOrder', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.9)

    config = parser.parse_args()

    print(config)

    dataset = dataLoader(config)
    if config.isTrain:
        config.padIdx = dataset.numItemsTrain
    else:
        config.padIdx = dataset.numItemsTest

    print('start training')
    training(dataset, config, device, writer)
