import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit App Config
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Forecasting — Stock Price Prediction (ARIMA & LSTM)")

# Sidebar Inputs
st.sidebar.header("Model & Data Options")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, TCS.NS)", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Model", ["ARIMA", "LSTM", "Compare Both"])
forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=60, value=7)
train_button = st.sidebar.button("Fetch & Train")

# Cache Data
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Utility functions
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
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

def fit_arima(series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    return model.fit()

# Main
if train_button:
    df = load_data(ticker, start_date, end_date)
    if df is None:
        st.error("No data fetched. Please check ticker symbol or date range.")
        st.stop()

    series = df['Close'].dropna()
    st.subheader("📊 Historical Closing Prices")
    st.line_chart(series)

    results = {}

    # ============= ARIMA MODEL =============
    if model_choice in ("ARIMA", "Compare Both"):
        st.subheader("🔹 ARIMA Model Forecast")

        try:
            arima_fit = fit_arima(series)
            arima_forecast = arima_fit.forecast(steps=forecast_days)
            arima_forecast = np.array(arima_forecast).flatten()

            # Create future index for forecast
            future_index = pd.date_range(start=series.index[-1], periods=forecast_days + 1, freq='B')[1:]
            forecast_series = pd.Series(arima_forecast, index=future_index)

            # Combine actual + forecast
            combined_df = pd.concat([series[-100:], forecast_series])
            st.line_chart(combined_df)
            results["ARIMA"] = arima_forecast

        except Exception as e:
            st.error(f"Error in ARIMA model: {e}")

    # ============= LSTM MODEL =============
    if model_choice in ("LSTM", "Compare Both"):
        st.subheader("🔹 LSTM Model Forecast")

        try:
            data = series.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(data)

            seq_len = 60
            X, y = create_sequences(scaled, seq_len)
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            lstm = build_lstm((X_train.shape[1], X_train.shape[2]))
            lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

            # Forecast future
            last_seq = scaled[-seq_len:]
            preds_scaled = []
            cur_seq = last_seq.copy()

            for _ in range(forecast_days):
                pred = lstm.predict(cur_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
                preds_scaled.append(pred)
                cur_seq = np.vstack([cur_seq[1:], [[pred]]])

            preds_scaled = np.array(preds_scaled).reshape(-1, 1)
            preds = scaler.inverse_transform(preds_scaled).flatten()
            results["LSTM"] = preds

            # Create future index for forecast
            future_index = pd.date_range(start=series.index[-1], periods=forecast_days + 1, freq='B')[1:]
            forecast_series = pd.Series(preds, index=future_index)

            # ✅ Flatten actuals before combining
            last_actual = series[-100:].values.flatten()
            actual_series = pd.Series(last_actual, index=series.index[-100:], name="Actual")

            # Combine actual + forecast
            combined = pd.concat([actual_series, forecast_series.rename("LSTM Forecast")])
            st.line_chart(combined)

        except Exception as e:
            st.error(f"Error in LSTM model: {e}")

# Footer
st.markdown("---")
st.markdown("**References:** ARIMA (statsmodels), LSTM (TensorFlow/Keras), yfinance for stock data.")
