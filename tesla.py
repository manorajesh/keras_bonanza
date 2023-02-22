import yfinance as yf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Download the historical stock prices of Tesla
ticker = "TSLA"
data = yf.download(ticker, start="2010-01-01", end="2022-02-22")

# Extract the closing prices as the output (Y)
Y = data['Close'].values

# Normalize the data to be between 0 and 1
Y = Y / np.max(Y)

# Create the input (X) as a sequence of integers
X = np.arange(len(Y))

# Define the neural network model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the data
model.fit(X, Y, epochs=100, batch_size=10)

# Predict the output for new input data
new_X = np.array([len(Y), len(Y) + 1, len(Y) + 2])  # Example input data
predictions = model.predict(new_X)

# Denormalize the predictions to get the actual stock prices
predictions = predictions * np.max(Y)

print(predictions)