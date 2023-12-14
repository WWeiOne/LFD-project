import numpy as np
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred): 
    a=(y_true - y_pred)
    b=y_true
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return np.mean(np.abs(c)) * 100

import matplotlib.pyplot as plt

def plot_data(x, y, label, plot_type, title, file_name, y_label='', x_label='', legend_loc='upper left', x2=None, y2=None, label2=None):
    plt.figure()
    if plot_type == 'semilogy':
        plt.semilogy(x, y, label=label)
    elif plot_type == 'plot':
        plt.plot(x, y, label=label)
        if x2 is not None and y2 is not None and label2 is not None:
            plt.plot(x2, y2, label=label2)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)
    plt.savefig(file_name)
    plt.close()



def evaluate_metrics(y_true, y_pred):
    MSE=np.mean((y_pred- y_true)**2)
    MAE=np.mean(np.abs((y_pred- y_true)))
    MAPE=mean_absolute_percentage_error(y_true,y_pred)
    WAPE=np.sum(np.abs(y_pred- y_true))/np.sum(np.abs(y_true))

    print('MAPE: ',MAPE)
    print('WAPE: ',WAPE)
    print('MAE: ',MAE)
    print('RMSE: ',MSE**0.5)

def plot_loss(y_true, y_pred, iter_losses=[], epoch_losses=[], name=''):
    if len(iter_losses) and len(epoch_losses):
        plot_data(range(len(iter_losses)), iter_losses, 'Iteration Losses', 'semilogy', 'Iteration Losses Over Time', name+'iter_loss.png')
        plot_data(range(len(epoch_losses)), epoch_losses, 'Epoch Losses', 'semilogy', 'Epoch Losses Over Time', name+'epoch_loss.png')
    plot_data(range(len(y_pred)), y_pred, 'Predicted', 'plot', 'Prediction vs True', name+'pred.png', x2=range(len(y_true)), y2=y_true, label2='True')

