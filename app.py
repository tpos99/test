# ==== app.py ====
import streamlit as st
import pandas as pd
from data_loader import get_data
from lstm_model import train_lstm_model, predict_next_return
from optimizer import optimize_portfolio
from backtest import backtest_portfolio

st.title("Real-Time Portfolio Diversification Dashboard")

assets = ['BTC-USD', 'GC=F', 'SPY', 'ASII.JK']
data = get_data(assets)
returns = data.pct_change().dropna()
st.line_chart(data)

predicted = []
for ticker in assets:
    series = returns[ticker].dropna().values
    model = train_lstm_model(series)
    pred = predict_next_return(model, series)
    predicted.append(pred)

cov_matrix = returns.cov().values
weights = optimize_portfolio(np.array(predicted), cov_matrix)

st.subheader("Optimal Portfolio Weights")
st.bar_chart(pd.Series(weights, index=assets))

st.subheader("Backtest Result")
backtest = backtest_portfolio(weights, returns)
st.line_chart(backtest)
