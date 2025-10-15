import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Forecasting — Stock Price Prediction (ARIMA & LSTM)")

st.sidebar.header("Model & Data Options")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, TCS.NS)", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Model", ["ARIMA", "LSTM", "Compare Both"])
forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=60, value=7)
train_button = st.sidebar.button("Fetch & Train")

@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    return df[['Open','High','Low','Close','Volume']]

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    return model.fit()

if train_button:
    df = load_data(ticker, start_date, end_date)
    if df is None:
        st.error("No data fetched.")
        st.stop()
    series = df['Close'].dropna()
    st.line_chart(series)

    results = {}

    if model_choice in ("ARIMA", "Compare Both"):
        st.subheader("ARIMA Model")
        arima_fit = fit_arima(series)
        arima_forecast = arima_fit.forecast(steps=forecast_days)
        results['ARIMA'] = arima_forecast

       # Ensure same length and 1D data before plotting
arima_forecast = np.array(arima_forecast).flatten()
forecast_df = pd.DataFrame({
    'Actual': series[-len(arima_forecast):].values,
    'ARIMA Forecast': arima_forecast
})
st.line_chart(forecast_df)

    if model_choice in ("LSTM", "Compare Both"):
        st.subheader("LSTM Model")
        data_close = series.values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data_close)

        seq_len = 60
        X, y = create_sequences(scaled, seq_length=seq_len)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        last_seq = scaled[-seq_len:]
        forecast_scaled = []
        cur_seq = last_seq.copy()
        for _ in range(forecast_days):
            pred = model.predict(cur_seq.reshape(1, seq_len, 1), verbose=0)[0,0]
            forecast_scaled.append(pred)
            cur_seq = np.vstack([cur_seq[1:], [[pred]]])
        forecast_scaled = np.array(forecast_scaled).reshape(-1,1)
        forecast_inv = scaler.inverse_transform(forecast_scaled)
        results['LSTM'] = forecast_inv.flatten()

        st.line_chart(pd.DataFrame({'Actual': series[-100:], 'LSTM Forecast': forecast_inv.flatten()}))

st.markdown("**References:** ARIMA (statsmodels), LSTM (TensorFlow/Keras), yfinance for data.")
