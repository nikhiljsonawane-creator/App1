import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Stock Price Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #B0B0B0;
    margin-bottom: 40px;
}

.prediction-box {
    background: linear-gradient(90deg,#11998e,#38ef7d);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
}

div[data-testid="metric-container"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    '<p class="main-title">📈 Stock Price Forecast Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Yahoo Finance + ARIMA Forecasting Model</p>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("⚙️ Dashboard Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    "RELIANCE.NS"
)

ticker = ticker.strip().upper()

run = st.sidebar.button("🚀 Generate Forecast")

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
if run:

    try:

        with st.spinner("Fetching stock data..."):

            data = yf.download(
                ticker,
                period="5y",
                auto_adjust=True,
                progress=False
            )

        if data.empty:
            st.error("No data found for this ticker.")
            st.stop()

        # -----------------------------------------
        # FIX FOR SERIES ERROR
        # -----------------------------------------
        close_prices = data["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = close_prices.dropna()

        # -----------------------------------------
        # METRICS
        # -----------------------------------------
        current_price = float(close_prices.iloc[-1])
        highest_price = float(close_prices.max())
        lowest_price = float(close_prices.min())

        st.subheader("📊 Stock Statistics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "💰 Current Price",
            f"₹{current_price:,.2f}"
        )

        c2.metric(
            "📈 5 Year High",
            f"₹{highest_price:,.2f}"
        )

        c3.metric(
            "📉 5 Year Low",
            f"₹{lowest_price:,.2f}"
        )

        st.divider()

        # -----------------------------------------
        # HISTORICAL CHART
        # -----------------------------------------
        st.subheader("📈 Last 5 Years Price Chart")

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        ax.plot(
            close_prices.index,
            close_prices.values,
            linewidth=2,
            color="#38ef7d"
        )

        ax.set_title(
            f"{ticker} Closing Price"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        st.divider()

        # -----------------------------------------
        # MONTHLY DATA FOR ARIMA
        # -----------------------------------------
        monthly_prices = (
            close_prices
            .resample("ME")
            .last()
        )

        # -----------------------------------------
        # ARIMA MODEL
        # -----------------------------------------
        model = ARIMA(
            monthly_prices,
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        # -----------------------------------------
        # FORECAST TO JUNE 2027
        # -----------------------------------------
        target_date = pd.Timestamp(
            "2027-06-30"
        )

        months_ahead = (
            (target_date.year -
             monthly_prices.index[-1].year) * 12
            +
            (target_date.month -
             monthly_prices.index[-1].month)
        )

        if months_ahead <= 0:
            months_ahead = 1

        forecast = model_fit.forecast(
            steps=months_ahead
        )

        forecast_dates = pd.date_range(
            start=monthly_prices.index[-1],
            periods=months_ahead + 1,
            freq="ME"
        )[1:]

        predicted_price = float(
            forecast.iloc[-1]
        )

        # -----------------------------------------
        # FORECAST CARD
        # -----------------------------------------
        st.subheader("🤖 ARIMA Forecast")

        st.markdown(
            f"""
            <div class="prediction-box">
                Predicted Price for June 2027
                <br><br>
                ₹ {predicted_price:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # -----------------------------------------
        # FORECAST CHART
        # -----------------------------------------
        st.subheader(
            "📈 Historical vs Forecast"
        )

        fig2, ax2 = plt.subplots(
            figsize=(14, 6)
        )

        ax2.plot(
            monthly_prices.index,
            monthly_prices.values,
            linewidth=2,
            label="Historical Prices"
        )

        ax2.plot(
            forecast_dates,
            forecast.values,
            marker="o",
            linewidth=2,
            label="Forecast"
        )

        ax2.set_title(
            f"{ticker} Forecast"
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig2)

        st.divider()

        # -----------------------------------------
        # RECENT DATA
        # -----------------------------------------
        with st.expander(
            "📄 View Recent Stock Data"
        ):
            st.dataframe(
                data.tail(10),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")
