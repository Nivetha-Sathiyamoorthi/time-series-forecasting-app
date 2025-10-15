import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Streamlit Page Config
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Forecasting App")

# Sidebar
st.sidebar.header("🔧 Model & Data Options")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, MSFT, TCS.NS)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "LSTM", "Compare Both"])
forecast_days = st.sidebar.number_input("Forecast Horizon (days)", min_value=1, max_value=60, value=7)
train_button = st.sidebar.button("Fetch & Train")

# ------------------ Helper Functions ------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    return df[['Close']]

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
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

# ------------------ Main Execution ------------------
if train_button:
    df = load_data(ticker, start_date, end_date)
    if df is None:
        st.error("❌ No data fetched. Please check ticker or date range.")
        st.stop()

    st.subheader("📊 Historical Data")
    st.line_chart(df["Close"])

    series = df["Close"].dropna()

    results = {}

    # ============= ARIMA MODEL =============
    if model_choice in ("ARIMA", "Compare Both"):
        st.subheader("🔹 ARIMA Model Forecast")

        try:
            arima_model = ARIMA(series, order=(5, 1, 0))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=forecast_days)

            # Ensure shape correctness
            arima_forecast = np.array(arima_forecast).flatten()
            actual = series[-forecast_days:].values.flatten()

            # Create a proper DataFrame
            forecast_df = pd.DataFrame({
                "Actual": pd.Series(actual, index=series.index[-forecast_days:]),
                "ARIMA Forecast": pd.Series(arima_forecast, index=pd.date_range(series.index[-1], periods=forecast_days+1, freq='B')[1:])
            })

            st.line_chart(forecast_df)

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

            # Future index
            future_index = pd.date_range(start=series.index[-1], periods=forecast_days+1, freq='B')[1:]
            forecast_series = pd.Series(preds, index=future_index)

            # Combine last 100 actuals + forecast
            combined = pd.concat([
                pd.Series(series[-100:], name="Actual"),
                pd.Series(forecast_series, name="LSTM Forecast")
            ])

            st.line_chart(combined)

        except Exception as e:
            st.error(f"Error in LSTM model: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, TensorFlow & Statsmodels")
