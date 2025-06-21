import gradio as gr
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import tempfile

def get_stock_data(symbol='AAPL', period='6mo', interval='1d'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    return df[['Close']]

def prepare_data(data, seq_len=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

def predict_future(model, data, n_days, scaler):
    prediction_input = data[-60:].reshape(1, 60, 1)
    preds = []
    for _ in range(n_days):
        pred = model.predict(prediction_input, verbose=0)[0]
        preds.append(pred)
        prediction_input = np.append(prediction_input[:, 1:, :], [[pred]], axis=1)
    return scaler.inverse_transform(preds)

def predict_stock(symbol='AAPL'):
    try:
        df = get_stock_data(symbol)
        if df.empty:
            return "Invalid symbol or no data.", None
        X, y, scaler = prepare_data(df.values)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)

        preds = predict_future(model, df.values, 7, scaler)

        last_date = df.index[-1]
        forecast_dates = pd.date_range(last_date, periods=8, freq='B')[1:]
        result_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': preds.flatten()})

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index[-60:], df['Close'][-60:], label='Past Prices')
        ax.plot(result_df['Date'], result_df['Predicted Close'], label='Predicted', marker='o')
        ax.set_title(f"{symbol.upper()} Stock Price Prediction (Next 7 Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)

        # Save to temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            image_path = tmpfile.name

        return result_df.to_markdown(index=False), image_path

    except Exception as e:
        return f"Error: {str(e)}", None


# Gradio Interface
demo = gr.Interface(
    fn=predict_stock,
    inputs=gr.Textbox(label="Enter Stock Symbol", placeholder="e.g. AAPL, TSLA, GOOGL"),
    outputs=[
        gr.Textbox(label="7-Day Price Forecast"),
        gr.Image(label="Prediction Chart")
    ],
    title="📈 Live Stock Price Predictor",
    description="Enter any valid stock symbol (like AAPL, TSLA, GOOGL) to forecast the next 7 days of stock closing prices using an LSTM model."
)

if __name__ == "__main__":
    demo.launch()
