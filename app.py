import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model

st.title("Stock Price Prediction App")

# Load model
model = load_model("Latest_stock_price_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

stock = st.text_input("Enter Stock Symbol", "GOOG")

if st.button("Predict"):

    data = yf.download(stock, start="2015-01-01", end="2024-01-01", auto_adjust=True)
    data = data[['Close']]

    scaled_data = scaler.transform(data)

    x_test = []
    for i in range(100, len(scaled_data)):
        x_test.append(scaled_data[i-100:i])

    x_test = np.array(x_test)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    st.line_chart(predictions)