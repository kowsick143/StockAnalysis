from flask import Flask, request, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

app = Flask(__name__)

template = """
<html>
  <head>
    <title>Stock Analysis</title>
  </head>
  <body>
    <h1>Stock Analysis</h1>
    <form method="post">
      <label for="user_name">Enter your name:</label><br>
      <input type="text" id="user_name" name="user_name"><br>
      <label for="stock_name">Enter the stock ticker name:</label><br>
      <input type="text" id="stock_name" name="stock_name"><br>
      <label for="predict_year">Enter the year to analyze from:</label><br>
      <input type="number" id="predict_year" name="predict_year"><br>
      <input type="submit" value="Submit">
    </form>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_name = request.form["user_name"]
        stock_name = request.form["stock_name"]
        predict_year = request.form["predict_year"]

        stock_data = yf.download(stock_name, period='1y')
        current_year = time.localtime().tm_year
        year_difference = abs(current_year - int(predict_year))

        stock_data['Day'] = np.arange(len(stock_data))
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))

        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]
                dataX.append(a)
                dataY.append(dataset[(i+time_step), 0])
            return np.array(dataX), np.array(dataY)

        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=1)

        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        train_rmse = np.sqrt(np.mean(np.power((train_predict - y_train), 2)))
        test_rmse = np.sqrt(np.mean(np.power((test_predict - y_test), 2)))
        print(f'Train RMSE: {train_rmse}')
        print(f'Test RMSE: {test_rmse}')

        future_days = np.arange(len(stock_data), len(stock_data) + 365)
        future_data = pd.DataFrame({'Day': future_days})
        input_data = scaled_data[-100:]
        predicted_values = []
        for i in range(len(future_days)):
            predicted_value = model.predict(np.array([input_data]))
            predicted_values.append(predicted_value[0][0])
            input_data = np.roll(input_data, -1)
            input_data[-1] = predicted_value[0][0]
        future_data['Close'] = predicted_values

        start_date = datetime.now() + timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=len(future_data))
        future_data['Date'] = future_dates

        best_buy_day = future_data.loc[future_data['Close'].idxmin(), 'Date'].date()
        best_sell_day = future_data.loc[future_data['Close'].idxmax(), 'Date'].date()

        future_return_percentage = (future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]) / future_data['Close'].iloc[0] * 100
    future_return_inr = (future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]) * future_data['Close'].iloc[0]
    future_grow_date = future_data['Date'].iloc[-1].date()

    revenue = future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]
    roi = ((future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]) / future_data['Close'].iloc[0]) * 100
    roe = ((future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]) / future_data['Close'].iloc[0]) * 100

    future_data['Moving_Avg'] = future_data['Close'].rolling(window=50).mean()
    future_data['Std_Dev'] = future_data['Close'].rolling(window=50).std()
    stability = future_data['Std_Dev'].mean()

    resistance = future_data['Close'].max()
    if roi > 10 and future_data['Close'].iloc[-1] < resistance * 0.8:
        opinion = "Buy"
    elif roi < -10 and future_data['Close'].iloc[-1] > resistance * 0.8:
        opinion = "Sell"
    else:
        opinion = "Hold"

    print("Stock Analysis Report:")
    print(f"Stock Name: {stock_name}")
    print(f"Date Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best Buy Day: {best_buy_day}")
    print(f"Best Sell Day: {best_sell_day}")
    print(f"Future Return Percentage: {future_return_percentage}%")
    print(f"Future Return INR: ₹{future_return_inr}")
    print(f"Future Grow Date: {future_grow_date}")
    print(f"Revenue: ₹{revenue}")
    print(f"ROI: {roi}%")
    print(f"ROE: {roe}%")
    print(f"Stability: {stability}")
    print(f"Resistance: ₹{resistance}")
    print(f"Opinion: {opinion}")

    data = {
        'User Name': [user_name],
        'Stock Name': [stock_name],
        'Date Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Best Buy Day': [best_buy_day],
        'Best Sell Day': [best_sell_day],
        'Future Return Percentage': [future_return_percentage],
        'Future Return INR': [future_return_inr],
        'Future Grow Date': [future_grow_date],
        'Revenue': [revenue],
        'ROI': [roi],
        'ROE': [roe],
        'Stability': [stability],
        'Resistance': [resistance],
        'Opinion': [opinion]
    }
    df = pd.DataFrame(data)

    if os.path.isfile('stock_analysis.csv'):
        df.to_csv('stock_analysis.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('stock_analysis.csv', index=False)

    return "Data received!"
    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ... (your code here)
        return "Data received!"
    else:
        return render_template_string(template)

if __name__ == "__main__":
    app.run(debug=True)