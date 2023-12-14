"""
Helper functions.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def read_data(input_path, debug=True):
    """
    Read nasdaq stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    if 'nasdaq100' in input_path:
        df = pd.read_csv(input_path, nrows=250 if debug else None)
        print(df.shape)
        df = df.iloc[-3000: -880, :]
        X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].to_numpy()
        y = np.array(df.NDX)
    elif 'elec' in input_path:
        data_loaded = np.load(input_path)
        person_id = 300
        data_point_start = 6000
        training_point_end = 1760
        data_loaded = data_loaded[-data_point_start:-training_point_end, person_id-1]
        Data=[]
        start_date=datetime(2012, 1, 1,00,00,00) # define start date
        for i in range(0,len(data_loaded)):
            record=[]
            record.append(data_loaded[i])#adding the electricity value
            record.append(start_date.month)
            record.append(start_date.day)
            record.append(start_date.hour)
            record.append(start_date.weekday())
            record.append(start_date.timetuple().tm_yday)
            record.append(start_date.isocalendar()[1])
            start_date=start_date+ timedelta(hours=0.25)
            Data.append(record)
        headers=['electricity','month','day','hour','day_of_week','day_of_year','week_of_year']
        df = pd.DataFrame(Data, columns=headers)
        X = df.loc[:, [x for x in df.columns.tolist() if x != 'electricity']].to_numpy()
        y = np.array(df.electricity)
    elif 'pm25' in input_path:
        df = pd.read_csv(input_path)
        df.drop(['No', 'Is', 'Ir'], axis=1, inplace=True)
        cbwd_mapping = {label: idx for idx, label in enumerate(df['cbwd'].unique())}
        df['cbwd'] = df['cbwd'].map(cbwd_mapping)
        df = df[-3000:-880]
        X = df.loc[:, [x for x in df.columns.tolist() if x != 'pm2.5']].to_numpy()
        y = np.array(df['pm2.5'])


    return X, y
