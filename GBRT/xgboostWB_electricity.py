# -*- coding: utf-8 -*-
"""XGBoostWB_Electricity
"""
import sys
#Import Libraries
#import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
#import matplotlib
#import matplotlib.pyplot as plt
import random
# %matplotlib inline
import shutil
from random import shuffle
import itertools

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from evaluate import *

num_periods_output = 24 #to predict
num_periods_input=24 #input

ALL_Test_Data=[]
ALL_Test_Prediction=[]


"""## preprocessing"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from datetime import datetime, timedelta
from sklearn import preprocessing

def New_preprocessing(TimeSeries):
#    print(TimeSeries.shape)
#    print(TimeSeries[0].shape)
   Data=[]
   start_date=datetime(2012, 1, 1,00,00,00) # define start date
   for i in range(0,len(TimeSeries)):
      record=[]
    #   print(i)
    #   print(len(TimeSeries[i]))
      record.append(TimeSeries[i])#adding the electricity value
      record.append(start_date.month)
      record.append(start_date.day)
      record.append(start_date.hour)
      record.append(start_date.weekday())
      record.append(start_date.timetuple().tm_yday)
      record.append(start_date.isocalendar()[1])
      #print(start_date.month,' ',start_date.day,' ',start_date.hour,' ',start_date.weekday(),' ',start_date.timetuple().tm_yday,' ',start_date.isocalendar()[1])
      start_date=start_date+ timedelta(hours=0.25)
      #print('year',start_date.year,'Month:',start_date.month,' day:',start_date.day,' hour:',start_date.hour)
      Data.append(record)
   ########## change list of lists to df ################
   headers=['electricity','month','day','hour','day_of_week','day_of_year','week_of_year']
   Data_df = pd.DataFrame(Data, columns=headers)
   #print(Data_df)
   sub=Data_df.iloc[:,1:]
   #Normalize features to be from -0.5 to 0.5 as mentioned in the paper
   New_sub= preprocessing.minmax_scale(sub, feature_range=(-0.5, 0.5))
   Normalized_Data_df=pd.DataFrame(np.column_stack([Data_df.iloc[:,0],New_sub]), columns=headers)
   #################################################################################################
   # cut training and testing training is 25968
   print(Normalized_Data_df.shape)
   Train=Normalized_Data_df.iloc[0:1992,:]
   print(Train.shape)
   Train=Train.values
   Train = Train.astype('float32')
   
#    print('Traing length :',len(Train))
   Test=Normalized_Data_df.iloc[(1992-num_periods_input):,:]
   Test=Test.values
   Test = Test.astype('float32')
   print(Test.shape)
#    print('Test length :',len(Test))
   Number_Of_Features=7
   ############################################ Windowing ##################################
   end=len(Train)
   start=0
   next=0
   x_batches=[]
   y_batches=[]  
   count=0
   while next+(num_periods_input)<end:
        next=start+num_periods_input
        x_batches.append(Train[start:next,:])
        # print(Train[next:next+num_periods_output])
        y_batches.append(Train[next:next+num_periods_output,0])
        start=start+1
   y_batches=np.asarray(y_batches)
   y_batches = y_batches.reshape(-1, num_periods_output, 1) 
#    print('Length of y batches :',len(y_batches),' ',num_periods_input,' ',num_periods_output)
   x_batches=np.asarray(x_batches) 
   x_batches = x_batches.reshape(-1, num_periods_input, Number_Of_Features)   
#    print('len x_batches ',len(x_batches))
   ############################################ Windowing ##################################
   end_test=len(Test)
   start_test=0
   next_test=0
   x_testbatches=[]
   y_testbatches=[]
   while next_test+(num_periods_input)<end_test:
        next_test=start_test+num_periods_input
        x_testbatches.append(Test[start_test:next_test,:])
        y_testbatches.append(Test[next_test:next_test+num_periods_output,0])
        start_test=start_test+num_periods_input
   y_testbatches=np.asarray(y_testbatches)
   y_testbatches = y_testbatches.reshape(-1, num_periods_output, 1)   
   x_testbatches=np.asarray(x_testbatches)
   x_testbatches = x_testbatches.reshape(-1, num_periods_input, Number_Of_Features) 
   #print('len Test',len(Test))
   #print('len xTestbatches',len(x_testbatches))
   return x_batches, y_batches, x_testbatches, y_testbatches

data_path=r'../dataset/dataset_elec.npy'
data=np.load(data_path)
print("data:")
# data= data[0:70,:]
person_id = 300
data_point_start = 3000
training_point_end = 888
data = data[-data_point_start:-training_point_end, person_id-1]
print(data.shape)
###################################################
x_batches_Full=[]
y_batches_Full=[]
X_Test_Full=[]
Y_Test_Full=[]
###################################################
print(len(data))
for i in range(1):
    x_batches=[]
    y_batches=[]
    X_Test=[]
    Y_Test=[]
    TimeSeries=data
    x_batches, y_batches,X_Test,Y_Test=New_preprocessing(TimeSeries)
    # print("x_batches shape: " + str(x_batches.shape))   
    # print("y_batches shape: " + str(y_batches.shape)) 
    for element1 in (x_batches):
        x_batches_Full.append(element1)
               
    for element2 in (y_batches):
        y_batches_Full.append(element2)        
                    
    for element5 in (X_Test):
        X_Test_Full.append(element5)
        
    for element6 in (Y_Test):
        Y_Test_Full.append(element6)
# print("x_batches_Full shape: " + str(len(x_batches_Full))) 
# print("y_batches_Full shape: " + str(len(y_batches_Full))) 
#---------------------shuffle windows  X and target Y together-------------------------------------
# print(len(All_Testing_Instances[0]))

combined = list(zip(x_batches_Full, y_batches_Full))
random.shuffle(combined)
shuffled_batch_features, shuffled_batch_y = zip(*combined)

#xgboost part
All_Training_Instances=[]
#=============== flatten each training window into Instance =================================
for i in range(0,len(shuffled_batch_features)):
    hold=[]
    temp=[]
    for j in range(0,len(shuffled_batch_features[i])):
    #**************** to run without features -->comment if else condition (just keep else statement) **************************
      if j==(len(shuffled_batch_features[i])-1):
          hold=np.concatenate((hold, shuffled_batch_features[i][j][:]), axis=None)
          
      else:
          hold=np.concatenate((hold, shuffled_batch_features[i][j][0]), axis=None)
          
    All_Training_Instances.append(hold)
    
print(len(All_Training_Instances))
print(len(All_Training_Instances[0]))

#=============== flatten each testing window into Instance =================================
All_Testing_Instances=[]
for i in range(0,len(X_Test_Full)):
  hold=[]
  temp=[]
  for j in range(0,len(X_Test_Full[i])):
  #**************** to run without features -->comment if else condition (just keep else statement) **************************
      if j==(len(X_Test_Full[i])-1):
          hold=np.concatenate((hold, X_Test_Full[i][j][:]), axis=None)
      else:
          hold=np.concatenate((hold, X_Test_Full[i][j][0]), axis=None)
   
  All_Testing_Instances.append(hold)

#=========================== final shape check =========================
All_Testing_Instances=np.reshape(All_Testing_Instances, (len(All_Testing_Instances),len(All_Testing_Instances[0])))
Y_Test_Full=np.reshape(Y_Test_Full, (len(Y_Test_Full),num_periods_output))

All_Training_Instances=np.reshape(All_Training_Instances, (len(All_Training_Instances),len(All_Training_Instances[0])))
shuffled_batch_y=np.reshape(shuffled_batch_y, (len(shuffled_batch_y),num_periods_output))
# print(shuffled_batch_y[10])

#=========================== CALLING XGBOOST ===========================
model=xgb.XGBRegressor(learning_rate =0.2,
 n_estimators=150,
 max_depth=8,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 scale_pos_weight=1,
 seed=42)

multioutput=MultiOutputRegressor(model).fit(All_Training_Instances,shuffled_batch_y)

print('Fitting Done!')

#============================== PREDICTION ===============================
y_pred=multioutput.predict(All_Testing_Instances)
rows, cols = y_pred.shape
y_pred = y_pred.reshape(rows*cols, 1)
rows, cols = Y_Test_Full.shape
y_true = Y_Test_Full.reshape(rows*cols, 1)

print(y_true.shape)
evaluate_metrics(y_true, y_pred)
plot_loss(y_true, y_pred, name='elec_')

