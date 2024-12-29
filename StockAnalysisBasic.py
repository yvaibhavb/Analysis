import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch historical data for Microsoft (MSFT)
def fetch_data():
    msft = yf.Ticker("MSFT")
    # Get the last 30 days of data
    data = msft.history(period="1mo", interval="1d")
    return data[['Close']]

# Prepare data for regression
def prepare_data(data):
    data['Day'] = np.arange(1, len(data) + 1)  # Day numbers
    X = data[['Day']].values  # Independent variable
    y = data['Close'].values  # Dependent variable
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict using linear regression
def predict_stock_prices():
    data = fetch_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    
    # Predict for the next day
    next_day = np.array([[len(data) + 1]])  # Next day number
    next_day_prediction = model.predict(next_day)
    print(f"Predicted stock price for next day: {next_day_prediction[0]:.2f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Day'], data['Close'], color='blue', label='Actual Prices')
    plt.plot(data['Day'], model.predict(data[['Day']]), color='red', label='Regression Line')
    plt.xlabel('Day')
    plt.ylabel('Closing Price')
    plt.title('Microsoft Stock Prices Prediction')
    plt.legend()
    plt.show()

# Run the prediction
predict_stock_prices()
