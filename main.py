import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pylab import rcParams

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

rcParams['figure.figsize'] = 20, 10

try:   #read in file
    stk_path = sys.argv[1]
except IndexError:
    #stk_path = ".\data\AMZN.csv"   #Anaconda env
	stk_path = "./data/AMZN.csv" #Linux env
	

command_mov = 'python moving_average.py '+str(stk_path)
command_lin = 'python linear_regression.py '+str(stk_path)
command_knn = 'python KNN.py '+str(stk_path)
command_xgb = 'python xgBoost.py '+str(stk_path)
command_lstm = 'python lstm.py '+str(stk_path)
command_plot = 'python plot.py '+str(stk_path)

print('************Moving Average**********')
fig = plt.figure()        #Display Text
ax = fig.add_axes([0,0,1,1])
ax.text(0.5*(left+right), 0.5*(bottom+top), 'Predictions with Moving Average',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=30, color='blue',
        transform=ax.transAxes)
		
ax.set_axis_off()
plt.show()
os.system(command_mov) #Run prediction

print('************Linear Regression**********')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0.5*(left+right), 0.5*(bottom+top), 'Predictions with Linear Regression',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=30, color='blue',
        transform=ax.transAxes)
		
ax.set_axis_off()
plt.show()
os.system(command_lin)

print('************KNN**********')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0.5*(left+right), 0.5*(bottom+top), 'Predictions with KNN',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=30, color='blue',
        transform=ax.transAxes)
		
ax.set_axis_off()
plt.show()
os.system(command_knn)

print('************Extreme Gradient Boosting**********')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0.5*(left+right), 0.5*(bottom+top), 'Predictions with Extreme Gradient Boosting',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=30, color='blue',
        transform=ax.transAxes)
		
ax.set_axis_off()
plt.show()
os.system(command_xgb)

print('************Long Short Term Memory**********')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0.5*(left+right), 0.5*(bottom+top), 'Predictions with Long Short Term Memory',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=30, color='blue',
        transform=ax.transAxes)
		
ax.set_axis_off()
plt.show()
os.system(command_lstm)

print('************Prediction Results Summary**********')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0.5*(left+right), 0.5*(bottom+top), 'Plot results comparing predictions from different models',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=30, color='blue',
        transform=ax.transAxes)
		
ax.set_axis_off()
plt.show()
os.system(command_plot)
