import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ────────────────────────────────────────────────
# App Config
# ────────────────────────────────────────────────
st.set_page_config(page_title="Custom Trading Dashboard MVP", layout="wide")
st.title("🚀 Custom Trading Dashboard - MVP v0.3 (with Backtesting)")
st.markdown("Multi-ticker charts, risk calculator, and now basic SMA crossover backtesting. Built for Sh in Corona, CA – Feb 2026")

# ────────────────────────────────────────────────
# Sidebar Controls
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Data & Chart Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", "AAPL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
    interval = st.selectbox("Chart Interval", ["1d", "1wk", "1mo"], index=0)

    show_volume = st.checkbox("Show Volume", value=True)
    show_sma = st.checkbox("Show SMA (20 & 50)", value=True)
    show_rsi = st.checkbox("Show RSI (14)", value=True)
    show_signals = st.checkbox("Show Strategy Signals", value=True)

    st.header("Risk Calculator (Mock)")
    capital = st.number_input("Account Capital ($)", value=10000.0, min_value=100.0)
    risk_pct = st.number_input("Risk % per Trade", value=1.0, min_value=0.1, max_value=5.0)
    entry_price = st.number_input("Entry Price", value=150.0)
    stop_price = st.number_input("Stop Loss Price", value=140.0)
    if stop_price != entry_price:
        shares = (capital * (risk_pct / 100)) / abs(entry_price - stop_price)
        st.info(f"Recommended Position Size: {shares:.2f} shares")

    theme = st.selectbox("Theme", ["Light", "Dark"], index=1)
    template = "plotly_dark" if theme == "Dark" else "plotly_white"

if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

# ────────────────────────────────────────────────
# Data Fetching
# ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

data = {}
for t in tickers:
    df = fetch_data(t, period, interval)
    if df is not None:
        data[t] = df

if not data:
    st.error("No data fetched.")
    st.stop()

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📈 Comparison", "🗃 Data", "🔍 Backtest"])

with tab1:
    st.subheader("Candlestick Charts")
    for ticker, df in data.items():
        with st.expander(f"{ticker} – {period}", expanded=True):
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name=ticker, increasing_line_color='green', decreasing_line_color='red'
            ))

            if show_volume:
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker_color='rgba(100,100,100,0.4)'))

            if show_sma:
                df['SMA20'] = df['Close'].rolling(20).mean()
                df['SMA50'] = df['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='purple')))

                if show_signals:
                    df['Signal'] = 0
                    df.loc[df.index[20:], 'Signal'] = np.where(df['SMA20'][20:] > df['SMA50'][20:], 1, 0)
                    df['Position'] = df['Signal'].diff()
                    buys = df[df['Position'] == 1]
                    sells = df[df['Position'] == -1]
                    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=12, color='lime')))
                    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=12, color='red')))

            fig.update_layout(
                title=f"{ticker} Price", yaxis_title="Price (USD)",
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                xaxis_rangeslider_visible=True, height=500, template=template
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_rsi:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                st.line_chart(rsi.rename("RSI (14)"), height=250)

with tab2:
    st.subheader("Normalized Comparison")
    if len(data) > 1:
        closes = pd.DataFrame({t: df['Close'] for t, df in data.items()})
        norm = closes / closes.iloc[0] * 100
        st.line_chart(norm, height=600)
    else:
        st.info("Add more tickers for comparison.")

with tab3:
    st.subheader("Raw Data")
    selected = st.selectbox("Ticker", list(data.keys()))
    st.dataframe(data[selected].tail(100).style.format(precision=2))

with tab4:
    st.subheader("Backtest: SMA Crossover Strategy (Long Only)")

    # Backtest settings
    bt_ticker = st.selectbox("Select ticker to backtest", list(data.keys()))
    initial_capital = st.number_input("Initial Capital ($)", value=10000.0, min_value=1000.0)
    df = data[bt_ticker].copy()

    if len(df) < 50:
        st.warning("Not enough data for SMA 50. Choose longer period.")
    else:
        # Prepare signals
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df = df.dropna()

        df['Signal'] = 0
        df['Signal'] = np.where(df['SMA20'] > df['SMA50'], 1, 0)
        df['Position'] = df['Signal'].diff()

        # Simulate trades
        in_position = False
        trades = []
        equity = [initial_capital]
        positions = []

        for i in range(len(df)):
            price = df['Close'].iloc[i]
            date = df.index[i]

            if df['Position'].iloc[i] == 1:  # Buy signal
                if not in_position:
                    shares = equity[-1] // price
                    if shares > 0:
                        cost = shares * price
                        in_position = True
                        trades.append({"Date": date, "Action": "BUY", "Price": price, "Shares": shares, "Cash": equity[-1] - cost})
                        positions.append(shares)

            elif df['Position'].iloc[i] == -1 and in_position:  # Sell signal
                proceeds = shares * price
                pnl = proceeds - cost
                trades.append({"Date": date, "Action": "SELL", "Price": price, "Shares": shares, "Cash": equity[-1] + proceeds, "PnL": pnl})
                equity.append(equity[-1] + pnl)
                in_position = False
                positions.append(0)

            else:
                # Hold
                if in_position:
                    current_value = equity[-1] - cost + shares * price
                    equity.append(current_value)
                    positions.append(shares)
                else:
                    equity.append(equity[-1])
                    positions.append(0)

        # Final sell if still in position
        if in_position:
            price = df['Close'].iloc[-1]
            proceeds = shares * price
            pnl = proceeds - cost
            trades.append({"Date": df.index[-1], "Action": "SELL (end)", "Price": price, "Shares": shares, "Cash": equity[-1] + proceeds, "PnL": pnl})
            equity.append(equity[-1] + pnl)

        # Results
        equity_series = pd.Series(equity, index=df.index[:len(equity)])
        returns = equity_series.pct_change().fillna(0)
        total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
        num_trades = len([t for t in trades if t['Action'].startswith('SELL')])
        wins = len([t for t in trades if t.get('PnL', 0) > 0])
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

        # Drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        max_dd = drawdown.min()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{total_return:.2f}%")
        col2.metric("Max Drawdown", f"{max_dd:.2f}%")
        col3.metric("Trades", num_trades)
        col4.metric("Win Rate", f"{win_rate:.1f}%")

        # Equity curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_series.index, y=equity_series, name="Equity", line=dict(color='blue')))
        fig_eq.update_layout(title="Equity Curve", yaxis_title="Portfolio Value ($)", template=template, height=400)
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill='tozeroy', fillcolor='rgba(255,0,0,0.3)'))
        fig_dd.update_layout(title="Drawdown", yaxis_title="% Drawdown", template=template, height=300)
        st.plotly_chart(fig_dd, use_container_width=True)

        if trades:
            st.subheader("Trade Log")
            trade_df = pd.DataFrame(trades)
            st.dataframe(trade_df.style.format({
                'Price': '${:.2f}', 'Cash': '${:.2f}', 'PnL': '${:.2f}'
            }), use_container_width=True)
        else:
            st.info("No trades triggered in this period with current SMA params.")

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.caption(f"MVP v0.3 • {datetime.now().strftime('%Y-%m-%d %H:%M PST')} • Extend with: commissions, stop-loss, multi-strategy selector, parameter optimization, real-time data via Polygon")
