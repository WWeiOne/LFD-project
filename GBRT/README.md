# GBRT

Refenrence: "Do We Really Need Deep Learning Models for 
Time Series Forecasting?" paper and its code https://github.com/Daniela-Shereen/GBRT-for-TSF/tree/main

## Modifications
* For electricity code, debugging and modifying the code with different vectorization since the input dimension is different.
* Write nasdaq and pm2.5 code based on the electricity code, rewrite the data loading and feature engineering part.
* Rewrite the evluation part for unifyed setting.

## Datasets

../data/dataset_elec.npy, ../data/pm25_interpolate.csv, ../nasdaq100_padding.csv

## How to run it

```
python xgboostWB_electricity.py
python xgboostWB_nasdaq.py
python xgboostWB_pm25.py
```

