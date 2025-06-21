# Intern_task_Project

---

# 📈 Live Stock Price Predictor

This project is a **Gradio web app** that predicts the next 7 days of stock closing prices using an **LSTM neural network** trained on historical data fetched from Yahoo Finance.

## 🚀 Features

* Fetches live stock data using `yfinance`
* Preprocesses data with `MinMaxScaler`
* Trains a simple LSTM model on the fly
* Predicts the next 7 business days of stock prices
* Visualizes both historical and predicted prices
* Interactive UI using Gradio

## 🛠️ Tech Stack

* Python
* Gradio
* TensorFlow / Keras
* Scikit-learn
* yfinance
* matplotlib
* pandas / numpy

## ⚡ Quick Start

1️⃣ **Clone the repository**

```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
```

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

> Example `requirements.txt`:
>
> ```
> gradio
> yfinance
> numpy
> pandas
> scikit-learn
> tensorflow
> matplotlib
> ```

3️⃣ **Run the app**

```bash
python app.py
```

4️⃣ **Open the Gradio interface in your browser**
You’ll see a link in the terminal like:

```
Running on http://127.0.0.1:7860
```

Click the link or open it manually.

## 📌 Example

![Sample Prediction Chart](https://via.placeholder.com/600x300.png?text=Sample+Prediction+Chart)

## 💡 How It Works

* The app downloads the last 6 months of daily stock data for the input symbol.
* It normalizes the data and creates sequences for LSTM training.
* A simple 2-layer LSTM is trained for 3 epochs.
* The trained model predicts the next 7 business days.
* Results are displayed in both table and chart form.


