import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# PROFESSIONAL STYLING
# -----------------------------
st.markdown("""
<style>

.stApp {
    background-color: #F3F6FA;
}

.big-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #0F172A;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #64748B;
    margin-bottom: 30px;
}

div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

.forecast-box {
    background: linear-gradient(
        90deg,
        #0F766E,
        #14B8A6
    );
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<p class="big-title">📈 Stock Price Forecast Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Yahoo Finance + ARIMA Forecasting Model</p>',
    unsafe_allow_html=True
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Dashboard Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    "RELIANCE.NS"
)

ticker = ticker.strip().upper()

run = st.sidebar.button("🚀 Generate Forecast")

# -----------------------------
# MAIN APP
# -----------------------------
if run:

    try:

        with st.spinner("Downloading stock data..."):

            data = yf.download(
                ticker,
                period="5y",
                auto_adjust=True,
                progress=False
            )

        if data.empty:
            st.error("No data found.")
            st.stop()

        close_prices = data["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = close_prices.dropna()

        # -----------------------------
        # METRICS
        # -----------------------------
        current_price = float(close_prices.iloc[-1])
        high_price = float(close_prices.max())
        low_price = float(close_prices.min())

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "💰 Current Price",
            f"₹{current_price:,.2f}"
        )

        c2.metric(
            "📈 5Y High",
            f"₹{high_price:,.2f}"
        )

        c3.metric(
            "📉 5Y Low",
            f"₹{low_price:,.2f}"
        )

        st.divider()

        # -----------------------------
        # HISTORICAL CHART
        # -----------------------------
        st.subheader("📈 Last 5 Years Price Chart")

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        ax.plot(
            close_prices.index,
            close_prices.values,
            linewidth=2
        )

        ax.set_title(
            f"{ticker} Closing Price"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        ax.grid(
            alpha=0.3,
            linestyle="--"
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        st.divider()

        # -----------------------------
        # MONTHLY DATA
        # -----------------------------
        monthly_prices = (
            close_prices
            .resample("ME")
            .last()
        )

        # -----------------------------
        # ARIMA MODEL
        # -----------------------------
        model = ARIMA(
            monthly_prices,
            order=(5, 1, 0)
        )

        model_fit = model.fit()

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

        # -----------------------------
        # FORECAST CARD
        # -----------------------------
        st.subheader("🤖 ARIMA Forecast")

        st.markdown(
            f"""
            <div class="forecast-box">
                Predicted Price for June 2027
                <br><br>
                ₹ {predicted_price:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # -----------------------------
        # FORECAST CHART
        # -----------------------------
        st.subheader(
            "📈 Historical vs Forecast"
        )

        fig2, ax2 = plt.subplots(
            figsize=(14, 6)
        )

        ax2.plot(
            monthly_prices.index,
            monthly_prices.values,
            label="Historical",
            linewidth=2
        )

        ax2.plot(
            forecast_dates,
            forecast.values,
            marker="o",
            linewidth=2,
            label="Forecast"
        )

        ax2.legend()
        ax2.grid(
            alpha=0.3,
            linestyle="--"
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig2)

        st.divider()

        with st.expander(
            "📄 View Recent Stock Data"
        ):
            st.dataframe(
                data.tail(10),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")
