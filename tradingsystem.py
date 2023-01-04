# is my code correct and complete 

import argparse
import json
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import mt5
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

class NeuralNetwork:
    def __init__(self, ticker, start_date, end_date, backcandles, split_ratio, epochs):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.backcandles = backcandles
        self.split_ratio = split_ratio
        self.epochs = epochs

        self.feature_columns = ['Open', 'High', 'Low', 'Adj Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']
        self.target_column = 'TargetClass'

    def download_data(self, symbol):
        # Download stock data from Yahoo Finance API
        data = yf.download(tickers=symbol, start=self.start_date, end=self.end_date)

        # Adding indicators
        data['RSI'] = ta.rsi(data.Close, length=15)
        data['EMAF'] = ta.ema(data.Close, length=20)
        data['EMAM'] = ta.ema(data.Close, length=100)
        data['EMAS'] = ta.ema(data.Close, length=150)

        data['Target'] = data['Adj Close'] - data.Open
        data['Target'] = data['Target'].shift(-1)

        data['TargetClass'] = [1 if data.Target[i] > 0 else 0 for i in range(len(data))]

        data['TargetNextClose'] = data['Adj Close'].shift(-1)

        data.dropna(inplace=True)
        data.reset_index(inplace=True)
        data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

        return data

    def split_data(self, X, y):
        # Split data into training and test sets
        split_index = int(len(X) * self.split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, X_test, y_test):
        model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=32, validation_data=(X_test, y_test))

        return model

class Prediction:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def get_predictions(self, data, symbol):
        data = self.scaler.transform(data)
        data = np.expand_dims(data, axis=0)
        prediction = self.model.predict(data)[0][0]
        return 1 if prediction > 0.5 else 0

class ForexScalpingSystem:
    def __init__(self, ticker, start_date, end_date, backcandles, split_ratio, epochs, max_trades, balance, risk_reward_ratio, max_risk_percent):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.backcandles = backcandles
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.max_trades = max_trades
        self.balance = balance
        self.risk_reward_ratio = risk_reward_ratio
        self.max_risk_percent = max_risk_percent

        # Calculate the maximum allowed risk per trade based on the balance and the maximum risk percentage
        self.max_risk = balance * (max_risk_percent / 100)

        self.nn = NeuralNetwork(self.ticker, self.start_date, self.end_date, self.backcandles, self.split_ratio, self.epochs)
        self.mt5 = mt5.MT5Connection()

    def get_predictions(self, data, symbol):
        scaler = MinMaxScaler()
        scaler.fit(data[self.nn.feature_columns])
        data[self.nn.feature_columns] = scaler.transform(data[self.nn.feature_columns])
        X, y = self._create_dataset(data, self.backcandles)
        X_train, X_test, y_train, y_test = self.nn.split_data(X, y)
        model = self.nn.train(X_train, y_train, X_test, y_test)
        prediction = Prediction(model, scal
    
    def get_lots(self, balance, risk_reward_ratio, max_risk):
        # Calculate the number of lots based on the balance, risk-reward ratio, and maximum risk
        stop_loss_pips = self.mt5.symbol_info(self.ticker).point * risk_reward_ratio
        lots = max_risk / stop_loss_pips
        return lots

    def run(self, symbol, login, password, server):
        # Connect to the MT5 server
        self.mt5.connect(server=server, login=login, password=password)

        # Download the data for the specified symbol and time period
        data = self.nn.download_data(symbol, self.start_date, self.end_date)

        # Get the predictions for each candle
        predictions = []
        for i in range(self.backcandles, len(data)):
            candle_data = data[i-self.backcandles:i]
            prediction = self.get_predictions(candle_data, symbol)
            predictions.append(prediction)

        # Place trades based on the predictions
        trades = 0
        for i, prediction in enumerate(predictions):
            # Check if we have reached the maximum number of trades
            if trades >= self.max_trades:
                break

            # Check if we have enough balance to place a trade
            if self.balance < self.mt5.symbol_info(self.ticker).min_deposit:
                break

            # Calculate the number of lots based on the balance, risk-reward ratio, and maximum risk
            lots = self.get_lots(self.balance, self.risk_reward_ratio, self.max_risk)

            # Place a trade based on the prediction
            if prediction == 1:
                self.mt5.order_send(symbol=symbol, cmd=mt5.ORDER_TYPE_BUY, volume=lots)
            elif prediction == 0:
                self.mt5.order_send(symbol=symbol, cmd=mt5.ORDER_TYPE_SELL, volume=lots)

            # Update the balance and trade count
            self.balance -= self.mt5.symbol_info(self.ticker).min_deposit
            trades += 1

        # Disconnect from the MT5 server
        self.mt5.disconnect()

if __name__ == '__main__':
    # Parse the command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True, help='The symbol to trade')
    parser.add_argument('--login', required=True, help='The login for the MT5 account')
    parser.add_argument('--password', required=True, help='The password for the MT5 account')
    parser.add_argument('--server', required=True, help='The server for the MT5 account')
   

