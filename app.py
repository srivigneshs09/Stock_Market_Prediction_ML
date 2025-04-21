import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Streamlit config
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>📊 Stock Market Predictor</h1>", unsafe_allow_html=True)

# Load the model
model = load_model('Stock Predictions Model.keras')

# User inputs
stock = st.text_input('🔍 Enter Stock Symbol', 'GOOG')
start = st.date_input("📅 From Date", value=pd.to_datetime("2012-01-01"))
end = st.date_input("📅 To Date", value=pd.to_datetime("2022-12-31"))

if start >= end:
    st.error("❌ End date must be after start date.")
else:
    data = yf.download(stock, start=start, end=end)

    if data.empty:
        st.error("❌ No data found for the selected stock or date range.")
    else:
        st.subheader(f"📂 Stock Data for {stock}")
        st.dataframe(data.tail())

        # Moving averages
        ma_50 = data['Close'].rolling(50).mean()
        ma_200 = data['Close'].rolling(200).mean()

        # Buy/Sell signal generation using crossover strategy
        def generate_ma_signals(price, ma_short, ma_long):
            signals = []
            trigger = 0
            for i in range(len(price)):
                if np.isnan(ma_short[i]) or np.isnan(ma_long[i]):
                    signals.append("Hold")
                elif ma_short[i] > ma_long[i]:
                    if trigger != 1:
                        signals.append("Buy")
                        trigger = 1
                    else:
                        signals.append("Hold")
                elif ma_short[i] < ma_long[i]:
                    if trigger != -1:
                        signals.append("Sell")
                        trigger = -1
                    else:
                        signals.append("Hold")
                else:
                    signals.append("Hold")
            return signals

        ma_signals = generate_ma_signals(data['Close'].values, ma_50.values, ma_200.values)

        # Plot MA signals
        st.subheader('📉 Price + MA50 + MA200 + Buy/Sell Signals')
        fig_ma = plt.figure(figsize=(12,6))
        plt.plot(data['Close'].values, label='Price', color='gray')
        plt.plot(ma_50.values, label='MA50', color='red')
        plt.plot(ma_200.values, label='MA200', color='blue')

        for i in range(len(ma_signals)):
            if ma_signals[i] == "Buy":
                plt.scatter(i, data['Close'].values[i], marker='^', color='green', label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif ma_signals[i] == "Sell":
                plt.scatter(i, data['Close'].values[i], marker='v', color='black', label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig_ma)

        # Split and scale data
        data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
        data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

        scaler = MinMaxScaler(feature_range=(0,1))
        past_100 = data_train.tail(100)
        final_test = pd.concat([past_100, data_test], ignore_index=True)
        final_scaled = scaler.fit_transform(final_test)

        x_test, y_test = [], []
        for i in range(100, final_scaled.shape[0]):
            x_test.append(final_scaled[i-100:i])
            y_test.append(final_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict
        predicted = model.predict(x_test)
        scale = 1 / scaler.scale_
        predicted = predicted * scale
        y_test = y_test * scale

        # LSTM-based signal generation (very basic logic)
        lstm_signals = []
        for i in range(1, len(predicted)):
            if predicted[i] > predicted[i-1] * 1.01:
                lstm_signals.append("Buy")
            elif predicted[i] < predicted[i-1] * 0.99:
                lstm_signals.append("Sell")
            else:
                lstm_signals.append("Hold")
        lstm_signals.insert(0, "Hold")

        # Plot predictions and LSTM signals
        st.subheader('🤖 Predicted vs Actual Price + LSTM Buy/Sell')
        fig_pred = plt.figure(figsize=(12,6))
        plt.plot(y_test, label='Actual Price', color='green')
        plt.plot(predicted, label='Predicted Price', color='red')

        for i in range(len(lstm_signals)):
            if lstm_signals[i] == "Buy":
                plt.scatter(i, predicted[i], marker='^', color='blue', label='Buy Signal' if 'Buy Signal' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif lstm_signals[i] == "Sell":
                plt.scatter(i, predicted[i], marker='v', color='black', label='Sell Signal' if 'Sell Signal' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig_pred)
