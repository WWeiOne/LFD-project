"""
Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

from utils import *
from model import *
from evaluate import *

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="../data/pm25_interpolate.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=24, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(args.dataroot, debug=False)

    # Initialize model
    print("==> Initialize DA-RNN model ...")
    model = DA_RNN(
        X,
        y,
        args.ntimestep,
        args.nhidden_encoder,
        args.nhidden_decoder,
        args.batchsize,
        args.lr,
        args.epochs
    )

    # Train
    print("==> Start training ...")
    model.train()

    # Prediction
    y_true = model.y[model.train_timesteps:]
    y_pred = model.test()

    evaluate_metrics(y_true, y_pred)
    plot_loss(y_true, y_pred, model.iter_losses, model.epoch_losses, 'pm25_')

if __name__ == '__main__':
    main()
