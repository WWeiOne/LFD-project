# DA-RNN

Refenrence: "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" paper, and an implementation using PyTorch: https://github.com/Zhenye-Na/DA-RNN

## Modifications
* rewrite utils.py to add the data loading and feature engineering for electricity and pm2.5 dataset.
* Write evaluation code to plot loss, and y_pred vs y_true.
* Change some parameters in main.py and model.py to unify the evaluation setting.

## Datasets

../data/dataset_elec.npy, ../data/pm25_interpolate.csv, ../nasdaq100_padding.csv

## How to run it

```
python main.py --lr 0.001 --epochs 500
```

may change hyperparameters
