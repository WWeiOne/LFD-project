import argparse
import math
import time

import torch
import torch.nn as nn
from models import LSTNet
import numpy as np;
import importlib

from utils import *;
import Optim
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_Data(person_id, window_size, batch_size, shuffle):
    input_signal = True # To use a linear signal with noise, set False here
    if input_signal:
        person_id = 300
        window_size = 24
            
        save_name = './data/dataset_elec.npy' # location of npy file dataset
        data_my = np.load(save_name)
        
        # person_id = 300, data[-3000:-1000, person_id - 1]
        data_point_start = 3000
        training_point_end = 1000
        data_splice = data_my[-data_point_start:-training_point_end, person_id-1] # 2000
    else:
        np.random.seed(42) 
        data_splice = np.array([i for i in range(0, 2000)]) + np.random.rand(2000)*50

    
    num = len(data_splice) - window_size +1
    
    x_set = np.zeros([num, window_size, 1]).astype(np.float32)
    y_set = np.zeros([num, 1])
        
    for i in range(0, num):
        data_i = data_splice[ i : i + window_size + 1]
        
        x_set[i,:, :] = data_i[0:window_size].reshape(-1, 1)
        y_set[i,:] = data_i[-1]
        
    dataset = torch.utils.data.TensorDataset(torch.tensor(x_set), torch.tensor(y_set))
    loader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         shuffle = shuffle)
    return loader

def get_Data_test(person_id, window_size, batch_size, shuffle):
    input_signal = True # To use a linear signal with noise, set False here
    if input_signal:
        person_id = 300
        window_size = 24
            
        save_name = './data/dataset_elec.npy' # location of npy file dataset
        data_my = np.load(save_name)
        
        # person_id = 300, data[-3000:-1000, person_id - 1]
        data_point_start = 1000+window_size
        training_point_end = 881
    else:
        np.random.seed(42) 
        data_splice = np.array([i for i in range(2000-window_size+1,2000+2000)]) + np.random.rand(2167)*50
    
    num = len(data_splice) - window_size +1
    
    x_set = np.zeros([num, window_size, 1]).astype(np.float32)
    y_set = np.zeros([num, 1])
        
    for i in range(0, num):
        data_i = data_splice[ i : i + window_size + 1]
        
        x_set[i,:, :] = data_i[0:window_size].reshape(-1, 1)
        y_set[i,:] = data_i[-1]
        
    dataset = torch.utils.data.TensorDataset(torch.tensor(x_set), torch.tensor(y_set))
    loader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         shuffle = shuffle)
    return loader

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, fig_i):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
    
    loader = get_Data(300, 168, 6,True)
    for i, (X, Y) in enumerate(loader):  
        X = X.cuda()
        Y = Y.cuda()
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        scale = 1
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m);
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();


    # HHC
    loader = get_Data_test(300, 7*24, 1,False)
    pred = []
    truth = []
    for i, (X, Y) in enumerate(loader):  
        truth.append(Y)
        pred.append(np.squeeze(model(X.cuda()).cpu().detach().numpy()))

    np.save("./outputs./truth"+str(fig_i), np.array(truth))
    np.save("./outputs./pred"+str(fig_i), np.array(pred))
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    plt.plot(truth, label='List A', color='blue')  # Plotting list a with blue color
    plt.plot(pred, label='List B', color='red')   # Plotting list b with red color
    plt.legend()  # Show legend with labels
    plt.xlabel('X-axis')  # Label for the x-axis
    plt.ylabel('Y-axis')  # Label for the y-axis
    plt.title('Plot of Lists A and B')  # Title of the plot
    plt.grid(True)  # Show grid
    # plt.show()  # Display the plot
    plt.savefig("./outputs./"+str(fig_i)+".png")
    fig_i = fig_i+1
    return rse, rae, correlation;

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    loader = get_Data(300, 7*24, 6,True)
    for i, (X, Y) in enumerate(loader):  
        X = X.cuda()
        Y = Y.cuda()
        model.zero_grad();
        output = model(X);
        scale = 1
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.item();
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples


def evaluateORI(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m);
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation;

def trainORI(data, X, Y, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad();
        output = model(X);
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.item();
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples
    
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default="data/electricity.txt",
                    help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize);
print(Data.rse);

Data.m=1

model = eval(args.model).Model(args, Data);

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();
    
    
best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training');
    for epoch in range(1, args.epochs+1):
        
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size, epoch);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size, epoch);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
#test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size);
#print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
