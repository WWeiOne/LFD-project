# ARIMA
## Dependencies
1. Download [electric dataset](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), and put it in ARIMA.
2. `python load_electricity.py` generate data npy.
## RUN ARIMA
1. `python arima_run.py` to run ARIMA model.

- It will take about 3 hours as the parameters searching is time consuming, but the result is same as paper.
- You can change the Test_Only to True in `arima_run.py` to experience in less than 10 minutes
    - It will use small dataset and small search range, so the result is inaccurate.

```py
Test_Only = False # Cost 2~3 hours to reproduce the same results.
# Test_Only = True  # For quick test, will finish within 10 minutes, but inaccurate.
```

## Functions
1. Arima_Model class in `arima_model`
    1. init: compile combination of parameters.
    2. fit: test all parameters to choose the best combination and train the model.
    3. pred: predict the future value and plot.
2. `arima_utils`
    1. print_and_save_AIC: save the AIC value during training.
    2. plot_AIC: plot the AIC img after find the best parameter combination.
3. `load_electricity.py` written by Haochuan to generate the npy data.
4. `evaluate.py` written by Yibo to collect the metrics.

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


# LSTNet
Refenrence: "Modeling long-and short-term temporal patterns with deep neural networks." 

https://github.com/laiguokun/LSTNet

## Modifications
* write load_electricity.py to pre-process raw dataset
* write ./main.py/get_data() function to split data into splices
* write ./main.py/get_data_test() function to generate a linear signal with noise for comparison
* rewrite ./main.py/evaluation function to plot truth-pred comparison every 5 epoch
* write ./outputs/Plot/imshow_all.py to plot training, testing, prediction in one figure

## Usage
1. Download electricity dataset from 

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

2. Unzip and place under ./data/
  
4. Pre-process

```
python load_electricity.py
```

3. Train & test

```
python main.py --gpu 0 --save save/elec.pt --output_fun Linear
```

3. Train & test with generated linear signal with noise 

```
python main.py --gpu 0 --save save/elec.pt --output_fun Linear --high_window 0 --egsignal True
```

4. Results would be saved under ./outputs/ and can be plotted by ./outputs/imshow_all.py



