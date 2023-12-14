import random
import numpy as np
import pandas as pd
from arima_model import Arima_Model
from arima_utils import plot_AIC
from evaluate import evaluate_metrics
from evaluate import plot_loss

Test_Only = False # Cost 2~3 hours to reproduce the same results.
# Test_Only = True  # For quick test, will finish within 10 minutes, but inaccurate.

# Set random seed, to make all results reproducible.
np.random.seed(0)
random.seed(0)

# Load dataset. (run load_data to generate the dataset)
save_name = "dataset_elec.npy"
data_my = np.load(save_name)

# Split training length 2000 and test dataset length 2200, data after 2000 is unseen for model.
person_id = 300
data_point_start = 3000
training_point_end = 1000
if Test_Only: 
    data_point_start = 3000
    training_point_end = 2800
training_data = data_my[-data_point_start:-training_point_end, person_id-1]
test_data = data_my[-data_point_start:-training_point_end+200, person_id-1]

# parameter range for "Grid search" of seasonal ARIMA model.
p_range = range(5)
d_range = range(5)
q_range = range(5)

# the seasonal periodicy is 24 hours, 24*60/15 = 96 points per season.
seasonal_m = int(24 * 60 / 15)

# Train
arima = Arima_Model(p_range, d_range, q_range, seasonal_m, test_only=Test_Only)
arima.fit(training_data)
#arima.load("arima.pickle")

# Plot the AIC grid search result. 
plot_AIC()

# Predict the future value from 2000 to 2120
plot_start = 1800
pred_start = 2000
pred_steps = 120
if Test_Only:
    pred_start = 200
    plot_start = 0
y_true, y_pred = arima.pred(test_data, plot_start, pred_start, pred_steps)
print(len(test_data))
# Evaluate Metrics
evaluate_metrics(y_true, y_pred)

# Plot in same range to compare with different models
plot_loss(y_true, y_pred)