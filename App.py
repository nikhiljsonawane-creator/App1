import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# PROFESSIONAL STYLING
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #F5F7FA;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #1E293B;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #64748B;
    margin-bottom: 40px;
}

div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

.prediction-card {
    background: linear-gradient(
        135deg,
        #0F766E,
        #14B8A6
    );
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 8px 25px rgba(20,184,166,0.3);
}

.prediction-title {
    font-size: 22px;
}

.prediction-price {
    font-size: 48px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<p class="main-title">📈 Stock Price Forecast Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Yahoo Finance + ARIMA Forecasting Model</p>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Dashboard Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    "RELIANCE.NS"
).strip().upper()

run = st.sidebar.button(
    "🚀 Generate Forecast"
)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
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
            st.error("No data found.")
            st.stop()

        # ------------------------------------------
        # CLOSE PRICE FIX
        # ------------------------------------------
        close_prices = data["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = close_prices.dropna()

        # ------------------------------------------
        # METRICS
        # ------------------------------------------
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
            "📈 5Y High",
            f"₹{highest_price:,.2f}"
        )

        c3.metric(
            "📉 5Y Low",
            f"₹{lowest_price:,.2f}"
        )

        st.markdown("---")

        # ------------------------------------------
        # INTERACTIVE HISTORICAL CHART
        # ------------------------------------------
        st.subheader(
            "📈 Last 5 Years Closing Price"
        )

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
            template="plotly_white",
            height=550,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # ------------------------------------------
        # MONTHLY DATA
        # ------------------------------------------
        monthly_prices = (
            close_prices
            .resample("ME")
            .last()
        )

        # ------------------------------------------
        # ARIMA MODEL
        # ------------------------------------------
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
             monthly_prices.index[-1].year)
            * 12
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

        # ------------------------------------------
        # FORECAST CARD
        # ------------------------------------------
        st.subheader(
            "🤖 ARIMA Forecast"
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">
                    Predicted {ticker} Price
                    <br>
                    June 2027
                </div>

                <br>

                <div class="prediction-price">
                    ₹ {predicted_price:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ------------------------------------------
        # FORECAST CHART
        # ------------------------------------------
        st.subheader(
            "📈 Historical vs Forecast"
        )

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices.values,
                mode="lines",
                name="Historical"
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
            template="plotly_white",
            height=600,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.markdown("---")

        # ------------------------------------------
        # DATA TABLE
        # ------------------------------------------
        with st.expander(
            "📄 View Recent Stock Data"
        ):
            st.dataframe(
                data.tail(10),
                use_container_width=True
            )

    except Exception as e:
        st.error(
            f"Error: {e}"
        )
