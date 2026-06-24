import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    font-size: 18px;
    margin-bottom: 40px;
}

.metric-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
}

.forecast-card {
    background: linear-gradient(
        135deg,
        #16A34A,
        #166534
    );
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
    margin-top: 20px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    '<div class="title">📈 Stock Price Forecast Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Yahoo Finance + ARIMA Forecasting Model'
    '</div>',
    unsafe_allow_html=True
)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    "RELIANCE.NS"
)

run = st.sidebar.button("🚀 Generate Forecast")

# ----------------------------
# MAIN APP
# ----------------------------
if run:

    try:
        # Dates
        end_date = datetime.today()
        start_date = end_date.replace(
            year=end_date.year - 5
        )

        # Download Data
        with st.spinner("Fetching stock data..."):

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

        if data.empty:
            st.error("No data found.")
            st.stop()

        close_prices = data["Close"].dropna()

        # ----------------------------
        # METRICS
        # ----------------------------
        latest_price = float(close_prices.iloc[-1])
        highest_price = float(close_prices.max())
        lowest_price = float(close_prices.min())

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "💰 Current Price",
            f"₹{latest_price:,.2f}"
        )

        col2.metric(
            "📈 5Y High",
            f"₹{highest_price:,.2f}"
        )

        col3.metric(
            "📉 5Y Low",
            f"₹{lowest_price:,.2f}"
        )

        st.divider()

        # ----------------------------
        # HISTORICAL CHART
        # ----------------------------
        st.subheader("📊 Last 5 Years Price Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices.values,
                mode="lines",
                name="Closing Price"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ----------------------------
        # MONTHLY DATA
        # ----------------------------
        monthly_prices = (
            close_prices
            .resample("ME")
            .last()
        )

        # ----------------------------
        # ARIMA MODEL
        # ----------------------------
        model = ARIMA(
            monthly_prices,
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        # ----------------------------
        # FORECAST TO JUNE 2027
        # ----------------------------
        forecast_end = pd.Timestamp(
            "2027-06-30"
        )

        months_ahead = (
            (forecast_end.year -
             monthly_prices.index[-1].year) * 12
            +
            (forecast_end.month -
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

        # ----------------------------
        # FORECAST CARD
        # ----------------------------
        st.subheader("🤖 ARIMA Forecast")

        st.markdown(
            f"""
            <div class="forecast-card">
                Predicted {ticker} Price<br>
                for June 2027
                <br><br>
                ₹ {predicted_price:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # ----------------------------
        # FORECAST CHART
        # ----------------------------
        st.subheader(
            "📈 Historical vs Forecast"
        )

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices.values,
                mode="lines",
                name="Historical Prices"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast.values,
                mode="lines+markers",
                name="Forecast"
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.divider()

        # ----------------------------
        # RECENT DATA
        # ----------------------------
        with st.expander(
            "📄 View Recent Stock Data"
        ):
            st.dataframe(
                data.tail(10),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")
