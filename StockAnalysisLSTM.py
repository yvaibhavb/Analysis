import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Fetch historical data for Microsoft (MSFT)
def fetch_data():
    msft = yf.Ticker("GOOG")
    data = msft.history(period="1mo", interval="1d")
    return data[['Close']]

# Prepare data for LSTM
def prepare_data(data, lookback=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM
    return X, y, scaler

# Build LSTM model
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def optimize_lookback(data, lookback_values):
    best_lookback = None
    best_rmse = float('inf')
    
    for lookback in lookback_values:
        X, y, scaler = prepare_data(data, lookback=lookback)
        
        # Split into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train the model
        model = build_lstm((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
        
        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        print(f"Lookback: {lookback}, RMSE: {rmse:.2f}")
        
        # Update best lookback
        if rmse < best_rmse:
            best_rmse = rmse
            best_lookback = lookback
    
    print(f"Best Lookback: {best_lookback}, Best RMSE: {best_rmse:.2f}")
    return best_lookback

# Run the optimization
data = fetch_data()
lookback_values = [3, 5, 7, 10]
best_lookback = optimize_lookback(data, lookback_values)

# Use the best lookback value for final prediction
def predict_stock_prices_lstm(best_lookback):
    data = fetch_data()
    
    # Prepare data
    X, y, scaler = prepare_data(data, lookback=best_lookback)
    
    # Split into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train the model
    model = build_lstm((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Predict the next day's price
    last_sequence = X[-1].reshape(1, best_lookback, 1)
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(next_day_scaled.reshape(-1, 1))[0, 0]
    print(f"Predicted next day price: {next_day_price:.2f}")
    
    # Visualize results
    dates = data.index[-len(actual):]  # Get the dates for the test set
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, color='blue', label='Actual Prices')
    plt.plot(dates, predictions, color='red', label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Google Stock Prices: Actual vs Predicted')
    plt.legend()
    plt.show()

# Run the final prediction with the best lookback value
predict_stock_prices_lstm(best_lookback)
