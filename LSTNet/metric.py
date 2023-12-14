import numpy as np
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred): 
    a=(y_true - y_pred)
    b=y_true
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return np.mean(np.abs(c)) * 100

def plot_and_save(x, y, plot_type, label, title, file_name, legend_loc='upper left'):
    with plt.figure() as fig:
        if plot_type == 'semilogy':
            plt.semilogy(x, y, label=label)
        elif plot_type == 'plot':
            plt.plot(x, y, label=label)
        plt.title(title)
        plt.legend(loc=legend_loc)
        plt.savefig(file_name)


def evaluate_metrics(y_true, y_pred):
    MSE=np.mean((y_pred- y_true)**2)
    MAE=np.mean(np.abs((y_pred- y_true)))
    MAPE=mean_absolute_percentage_error(y_true,y_pred)
    WAPE=np.sum(np.abs(y_pred- y_true))/np.sum(np.abs(y_true))

    print('MAPE: ',MAPE)
    print('WAPE: ',WAPE)
    print('MAE: ',MAE)
    print('RMSE: ',MSE**0.5)

def plot_loss(y_true, y_pred, iter_losses=None, epoch_losses=None):
    if len(iter_losses) and len(epoch_losses):
        plot_and_save(range(len(iter_losses)), iter_losses, 'semilogy', 'Iteration Losses', 'Iteration Losses Over Time', '1.png')
        plot_and_save(range(len(epoch_losses)), epoch_losses, 'semilogy', 'Epoch Losses', 'Epoch Losses Over Time', '2.png')
    plot_and_save(range(len(y_pred)), y_pred, 'plot', 'Predicted', 'Prediction vs True', '3.png', legend_loc='upper left')
    plot_and_save(range(len(y_true)), y_true, 'plot', 'True', 'Prediction vs True', '3.png', legend_loc='upper left')