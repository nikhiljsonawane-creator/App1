import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Investor Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

div[data-testid="metric-container"]{
    background-color:#111827;
    border:1px solid #1F2937;
    border-radius:18px;
    padding:20px;
}

div[data-testid="metric-container"]:hover{
    border:1px solid #2563EB;
}

.forecast-box{
    background:linear-gradient(
        135deg,
        #1E3A8A,
        #2563EB
    );
    padding:35px;
    border-radius:20px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
# 📈 Investor Dashboard
#### Yahoo Finance + ARIMA Forecasting
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙️ Dashboard")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    "RELIANCE.NS"
).strip().upper()

run = st.sidebar.button(
    "🚀 Generate Dashboard"
)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if run:

    try:

        with st.spinner("Fetching stock data..."):

            data = yf.download(
                ticker,
                period="5y",
                auto_adjust=True,
                progress=False
            )

            stock = yf.Ticker(ticker)

        if data.empty:
            st.error("No data found.")
            st.stop()

        close_prices = data["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = close_prices.dropna()

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------
        current_price = float(close_prices.iloc[-1])
        high_price = float(close_prices.max())
        low_price = float(close_prices.min())

        returns = close_prices.pct_change()

        volatility = (
            returns.std()
            * np.sqrt(252)
            * 100
        )

        cagr = (
            (
                close_prices.iloc[-1]
                /
                close_prices.iloc[0]
            ) ** (1 / 5)
            - 1
        ) * 100

        st.subheader("📊 Key Metrics")

        c1, c2, c3, c4, c5 = st.columns(5)

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

        c4.metric(
            "🚀 CAGR",
            f"{cagr:.2f}%"
        )

        c5.metric(
            "⚠️ Volatility",
            f"{volatility:.2f}%"
        )

        st.divider()

        # ---------------------------------------------------
        # COMPANY INFORMATION
        # ---------------------------------------------------
        info = stock.info

        st.subheader("🏢 Company Information")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Sector",
            info.get(
                "sector",
                "N/A"
            )
        )

        c2.metric(
            "Industry",
            info.get(
                "industry",
                "N/A"
            )
        )

        market_cap = info.get(
            "marketCap",
            None
        )

        if market_cap:
            market_cap = f"{market_cap:,.0f}"
        else:
            market_cap = "N/A"

        c3.metric(
            "Market Cap",
            market_cap
        )

        st.divider()

        # ---------------------------------------------------
        # MOVING AVERAGES
        # ---------------------------------------------------
        ma50 = (
            close_prices
            .rolling(50)
            .mean()
        )

        ma200 = (
            close_prices
            .rolling(200)
            .mean()
        )

        # ---------------------------------------------------
        # INTERACTIVE CHART
        # ---------------------------------------------------
        st.subheader(
            "📈 Last 5 Years Price Chart"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices,
                mode="lines",
                name="Close Price"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ma50.index,
                y=ma50,
                mode="lines",
                name="50 DMA"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ma200.index,
                y=ma200,
                mode="lines",
                name="200 DMA"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#111827",
            hovermode="x unified",
            height=600,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------------------------
        # SIGNAL
        # ---------------------------------------------------
        if ma50.iloc[-1] > ma200.iloc[-1]:
            st.success(
                "🟢 Golden Cross: Bullish Trend"
            )
        else:
            st.warning(
                "🔴 Death Cross: Bearish Trend"
            )

        st.divider()

        # ---------------------------------------------------
        # 52 WEEK HIGH LOW
        # ---------------------------------------------------
        high_52 = close_prices.tail(
            252
        ).max()

        low_52 = close_prices.tail(
            252
        ).min()

        c1, c2 = st.columns(2)

        c1.metric(
            "📈 52 Week High",
            f"₹{high_52:,.2f}"
        )

        c2.metric(
            "📉 52 Week Low",
            f"₹{low_52:,.2f}"
        )

        st.divider()

        # ---------------------------------------------------
        # ARIMA FORECAST
        # ---------------------------------------------------
        monthly_prices = (
            close_prices
            .resample("ME")
            .last()
        )

        model = ARIMA(
            monthly_prices,
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        target_date = pd.Timestamp(
            "2027-06-30"
        )

        months_ahead = (
            (
                target_date.year
                -
                monthly_prices.index[-1].year
            ) * 12
            +
            (
                target_date.month
                -
                monthly_prices.index[-1].month
            )
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

        upside = (
            (
                predicted_price
                -
                current_price
            )
            /
            current_price
        ) * 100

        # ---------------------------------------------------
        # FORECAST CARD
        # ---------------------------------------------------
        st.subheader(
            "🤖 Forecast"
        )

        st.markdown(
            f"""
            <div class="forecast-box">
                <h3>
                Predicted Price
                </h3>

                <h1>
                ₹{predicted_price:,.2f}
                </h1>

                <h4>
                June 2027
                </h4>

                <h4>
                Forecast Upside:
                {upside:.2f}%
                </h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # ---------------------------------------------------
        # FORECAST CHART
        # ---------------------------------------------------
        st.subheader(
            "📊 Historical vs Forecast"
        )

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices,
                mode="lines",
                name="Historical"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast,
                mode="lines+markers",
                name="Forecast"
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#111827",
            hovermode="x unified",
            height=600,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.divider()

        # ---------------------------------------------------
        # NEWS
        # ---------------------------------------------------
        st.subheader("📰 Latest News")

        news = stock.news[:5]

        if news:

            for item in news:

                title = item.get(
                    "title",
                    "No Title"
                )

                link = item.get(
                    "link",
                    None
                )

                if link:
                    st.markdown(
                        f"• [{title}]({link})"
                    )
                else:
                    st.write(
                        f"• {title}"
                    )

        else:
            st.info(
                "No recent news available."
            )

        st.divider()

        # ---------------------------------------------------
        # RECENT DATA
        # ---------------------------------------------------
        with st.expander(
            "📄 View Recent Stock Data"
        ):
            st.dataframe(
                data.tail(10),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Data Source: Yahoo Finance | Forecast: ARIMA Model | Educational Purposes Only"
        )

    except Exception as e:
        st.error(
            f"Error: {e}"
        )
