# -*- coding: utf-8 -*-
import matplotlib
from datetime import date, datetime, time, timedelta
from matplotlib import pyplot as plt
import pandas as pd

import math
from pylab import rcParams
import sys
import numpy as np
import seaborn as sns
import time


#### Input params ##################
try:
    stk_path = sys.argv[1]
except IndexError:
    #stk_path = ".\data\AMZN.csv"   #Anaconda env
	stk_path = "./data/AMZN.csv" #Linux env
    
test_size = 0.2                 # proportion of dataset to be used as test set
cv_size = 0.2                   # proportion of dataset to be used as cross-validation set
Nmax = 2                       # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
                                # Nmax is the maximum N we are going to test
fontsize = 14
ticklabelsize = 14
####################################

################DATA READ###############################
df = pd.read_csv(stk_path, sep = ",")

# Convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Get month of each sample
df['month'] = df['date'].dt.month

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

# Get sizes of each of the datasets
num_cv = int(cv_size*len(df))
num_test = int(test_size*len(df))
num_train = len(df) - num_cv - num_test

# Split into train, cv, and test
train = df[:num_train]
cv = df[num_train:num_train+num_cv]
train_cv = df[:num_train+num_cv]
test = df[num_train+num_cv:]

# Read all dataframes for the different methods
test_mov_avg = pd.read_csv("./out/test_mov_avg.csv", index_col=0)
test_mov_avg.loc[:, 'date'] = pd.to_datetime(test_mov_avg['date'],format='%Y-%m-%d')

test_lin_reg = pd.read_csv("./out/test_lin_reg.csv", index_col=0)
test_lin_reg.loc[:, 'date'] = pd.to_datetime(test_lin_reg['date'],format='%Y-%m-%d')

test_knn = pd.read_csv("./out/test_knn.csv", index_col=0)
test_knn.loc[:, 'date'] = pd.to_datetime(test_knn['date'],format='%Y-%m-%d')

test_xgboost = pd.read_csv("./out/test_xgboost.csv", index_col=0)
test_xgboost.loc[:, 'date'] = pd.to_datetime(test_xgboost['date'],format='%Y-%m-%d')

test_lstm= pd.read_csv("./out/test_lstm.csv", index_col=0)
test_lstm.loc[:, 'date'] = pd.to_datetime(test_lstm['date'],format='%Y-%m-%d')

colname_mov = test_mov_avg.columns[-1]
colname_lin = test_lin_reg.columns[-1]

# Plot all methods together to compare
rcParams['figure.figsize'] = 10, 8 # width 10, height 8
matplotlib.rcParams.update({'font.size': 14})

ax = test.plot(x='date', y='adj_close', style='g-', grid=True)
ax = test_mov_avg.plot(x='date', y=colname_mov, style='b-', grid=True, ax=ax)
ax = test_lin_reg.plot(x='date', y=colname_lin, style='m-', grid=True, ax=ax)
ax = test_knn.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax = test_xgboost.plot(x='date', y='est', style='y-', grid=True, ax=ax)
ax = test_lstm.plot(x='date', y='est', style='o-', grid=True, ax=ax)

ax.legend(['test', 
           'predictions using moving average', 
           'predictions using linear regression',
		   'predictions using KNN',
           'predictions using XGBoost',
		   'predictions using LSTM'
           ], loc='lower left')
ax.set_xlabel("date")
ax.set_ylabel("USD")
# ax.set_xlim([date(2018, 4, 23), date(2018, 11, 23)])
ax.set_xlim([date(2019, 4, 1), date(2019, 4, 30)])
ax.set_title('Prediction Performance of various models')
plt.show()