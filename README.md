# LSTM-for-cryptocurrency-forecasting
This repository contains code that receives historical data on the BTCUSDT trading pair with the Binance API and trains the LSTM model to predict changes in the Bitcoin closing price based on the previous 60 closing price values.
This code downloads and processes data on bitcoin quotes from the Binance exchange, then uses the LSTM model to predict bitcoin price changes based on previous price values. Specifically, it performs the following actions:
- Imports the necessary libraries: requests, pandas, numpy, tensorflow.keras and matplotlib.pyplot.
- Sets Binance API settings: URL, symbol, interval and limit.
- Retrieves data using the Binance API and creates a DataFrame.
- Removes unnecessary columns and converts values to numeric format.
- Divides the data into training and test samples.
- Scales the data.
- Converts data into a three-dimensional array to use the LSTM model.
- Creates and compiles an LSTM model using keras.
- Trains the model on a training sample.
- Visualizes model training on a graph.
- Predicts values on a test sample and scales them.
- Displays a graph with actual and predicted values on the test sample.
- Evaluates the quality of the model on a test sample using the mean absolute error (MAE) and the mean square error (MSE).
- Gets the last 60 values from the test sample, converts them into a three-dimensional array and predicts the next price value.
- Scales the predicted value and displays a graph with the actual and predicted values on the test sample, as well as the predicted value for an hour ahead.
