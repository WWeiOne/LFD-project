## Run ARIMA
### Dependencies
1. Download [electric dataset](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), and put it in ARIMA.
2. `python load_electricity.py` generate data npy.
### RUN ARIMA
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