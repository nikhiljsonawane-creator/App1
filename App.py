import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(
    page_title="Stock Price Forecast using ARIMA",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Forecast using ARIMA")
st.write(
    "Fetches the last 5 years of stock prices from Yahoo Finance and "
    "forecasts the stock price for June 2027 using ARIMA."
)

# User Input
ticker = st.text_input(
    "Enter Stock Ticker",
    value="RELIANCE.NS"
)

if st.button("Generate Forecast"):

    try:
        # Download last 5 years of data
        end_date = datetime.today()
        start_date = end_date.replace(year=end_date.year - 5)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        if data.empty:
            st.error("No data found for this ticker.")
            st.stop()

        close_prices = data["Close"].dropna()

        st.subheader("Last 5 Years Closing Prices")

        # Line Chart
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            close_prices.index,
            close_prices.values,
            linewidth=2
        )
        ax.set_title(f"{ticker} Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ARIMA Model
        st.subheader("ARIMA Forecast")

        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast until June 2027
        forecast_end = pd.Timestamp("2027-06-30")

        months_ahead = (
            (forecast_end.year - close_prices.index[-1].year) * 12
            + (forecast_end.month - close_prices.index[-1].month)
        )

        if months_ahead <= 0:
            months_ahead = 1

        forecast = model_fit.forecast(
            steps=months_ahead
        )

        june_2027_price = forecast.iloc[-1]

        st.success(
            f"Predicted {ticker} Price for June 2027: "
            f"₹{june_2027_price:.2f}"
        )

        # Forecast Plot
        forecast_dates = pd.date_range(
            start=close_prices.index[-1],
            periods=months_ahead + 1,
            freq="M"
        )[1:]

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(
            close_prices.index,
            close_prices.values,
            label="Historical Prices"
        )

        ax2.plot(
            forecast_dates,
            forecast.values,
            label="ARIMA Forecast"
        )

        ax2.set_title(
            f"{ticker} Historical and Forecast Prices"
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()

        st.pyplot(fig2)

        st.write("### Recent Data")
        st.dataframe(data.tail())

    except Exception as e:
        st.error(f"Error: {e}")
