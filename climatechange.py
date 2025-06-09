# Example: Simple LSTM model to predict temperature trends
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load climate data CSV with 'temperature' column
data = pd.read_csv('climate_data.csv')
temperature = data['temperature'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
temperature_scaled = scaler.fit_transform(temperature)

# Prepare sequences
def create_dataset(data, look_back=10):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X, Y = create_dataset(temperature_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back,1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Plot
plt.plot(temperature[look_back+1:], label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
