import math
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import seaborn as sns

from datetime import date
from pylab import rcParams
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Turning off warnings
import warnings
warnings.filterwarnings("ignore")

#### Input params ##################
try:
    stk_path = sys.argv[1]
except IndexError:
    #stk_path = ".\data\AMZN.csv"   #Anaconda env
	stk_path = "./data/AMZN.csv" #Linux env


def new_dataset(dataset, step_size):
#Creating dataset for analysis
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
        a = dataset[i:(i+step_size), ]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, ])
    return np.array(data_X), np.array(data_Y)

def get_mape(y_true, y_pred): 
#Compute mean absolute percentage error (MAPE)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#MAIN
#####DATA Read part #######
raw_data = pd.read_csv(stk_path, sep = ",")
raw_data.head()

# Convert Date column to datetime
raw_data['Date'] = pd.to_datetime(raw_data.Date,format='%Y-%m-%d')
raw_data.index = raw_data['Date']

#Creating Prediction Data
Closing_Price = np.asarray(raw_data['Close'])
X_Prediction_Data, y_Prediction_Data = new_dataset(Closing_Price, 1)


#Spliting Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_Prediction_Data, y_Prediction_Data, test_size=0.20, shuffle=False,random_state=1)


Dates = raw_data['Date']
Train_Dates = Dates[:len(y_train)]
Test_Dates = Dates[len(raw_data)-len(y_test):]

# Create the LSTM network
knn_without_model_selection = KNeighborsRegressor(n_neighbors=1)
    
#fit the model and make predictions
knn_without_model_selection.fit(X_train.reshape(-1,1),y_train)
prediction = knn_without_model_selection.predict(X_test.reshape(-1,1))

print("HyperParameter Selection")
# Calculate RMSE
rmse_bef_tuning = math.sqrt(mean_squared_error(y_test, prediction))
print("RMSE = %0.3f" % rmse_bef_tuning)

# Calculate MAPE
mape_pct_bef_tuning = get_mape(y_test, prediction)
print("MAPE = %0.3f%%" % mape_pct_bef_tuning)

#using gridsearch to find the best parameter
params = {'n_neighbors':np.arange(1,25)}
knn = KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=10)     

#fit the model and make predictions
model.fit(X_train.reshape(-1,1),y_train)
preds = model.predict(X_test.reshape(-1,1))
# means = model.cv_results_['mean_train_score']
print(model.best_params_)

print("Prediction Results with Optimized HyperParameter")
# Calculate RMSE
rmse_bef_tuning = math.sqrt(mean_squared_error(y_test, preds))
print("RMSE = %0.3f" % rmse_bef_tuning)

# Calculate MAPE
mape_pct_bef_tuning = get_mape(y_test, preds)
print("MAPE = %0.3f%%" % mape_pct_bef_tuning)

#Write to text file
myfile = open('%s_knn_error.txt' % stk_path,'a+')		
myfile.write("RMSE and MAPE for " + stk_path + '\n')
myfile.write("RMSE = %0.3f\n" % rmse_bef_tuning)
myfile.write("MAPE = %0.3f%%\n" % mape_pct_bef_tuning)

# Plot adjusted close over time
# rcParams['figure.figsize'] = 10, 8 # width 10, height 8
# plt.figure(figsize=(12, 8), dpi=80)
# plt.plot(raw_data['Date'], raw_data['Close'], color='green', label='Predicted Data')
# plt.plot(Test_Dates, preds, color='blue', label='Predicted Data')
# plt.grid()
# plt.xlabel('Dates')
# plt.ylabel('Adjusted Close Price')
# plt.title("Predicted Adjusted Close Price of %s" % stk_path)
# plt.savefig('%s_knn_adj_close.jpg' % stk_path)
# plt.show()

est_df = pd.DataFrame({'est': preds.reshape(-1), 
                       'date': raw_data[len(raw_data)-len(y_test):]['Date']})
rcParams['figure.figsize'] = 10, 8 # width 10, height 8

ax = raw_data.plot(x='Date', y = 'Adj Close',color='green', label='Predicted Data')
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_title('Prediction with Test Data - KNN')
plt.show()

# Plot adjusted close over time, for test set only
rcParams['figure.figsize'] = 10, 8 # width 10, height 8
ax = raw_data.plot(x='Date', y = 'Adj Close',color='green', label='Predicted Data')
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_title('Prediction with Test Data - KNN')
ax.set_xlim([date(2019, 4, 1), date(2019, 4, 30)])
#ax.set_ylim([1250, 2100])
ax.set_title("Zoom in to test set")
plt.show()

test_mov_avg = est_df
test_mov_avg.to_csv("./out/test_knn.csv")