************************Stock Prices Prediction*******************
EECE5644 Introduction to Machine Learning and Pattern Recognition
		         CourseWork Project
*****************************************************************
Folder Structure:

Contains following code files:
				1. main.py - Main file that runs other models on the given stock data
				2. moving_average.py - Performs Moving Average based Stock Price Prediction
				3. linear_regression.py - Performs Linear Regression based Stock Price Prediction
				4. KNN.py - Performs K Nearest Neighbors based Stock Price Prediction
				5. xgBoost.py - Performs Extreme Gradient Boosting based Stock Price Prediction
				6. LSTM.py - Performs Long Short Term Memory based Stock Price Prediction
				7. plot.py - Gathers prediction results from all files and plots them in a single file for comparison

data folder - Has 5 years Historical Data of stock prices from companies from various sectors like Software, Oil, Finance etc
out folder - Will host the prediction result csv file created by the python files during execution


Instructions to run:

pip install -r requirements.txt
python main.py ./data/anyname.csv
