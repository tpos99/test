import streamlit as st
from data_loader import get_data
from lstm_model import train_lstm_model, predict_next_return
from optimizer import optimize_portfolio
from backtest import backtest_portfolio

import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Smart Portfolio Diversification with ML")

# Aset yang digunakan
assets = ["BTC-USD", "GC=F", "SPY", "ASII.JK"]

# Ambil data historis
with st.spinner("📥 Mengambil data..."):
    data = get_data(assets)
    returns = data.pct_change().dropna()

# Cek data yang berhasil dimuat
available_assets = [ticker for ticker in assets if ticker in returns.columns]
if not available_assets:
    st.error("Gagal memuat data untuk semua aset. Periksa ticker!")
    st.stop()

st.sidebar.header("⚙️ Pengaturan")
risk_aversion = st.sidebar.slider("Risk Aversion (λ)", 0.0, 1.0, 0.5)

# Prediksi return dengan LSTM
st.subheader("📈 Prediksi Return Berikutnya (LSTM)")
predicted = []
valid_assets = []

for ticker in available_assets:
    series = returns[ticker].dropna().values
    if len(series) > 60:
        model = train_lstm_model(series)
        pred = predict_next_return(model, series)
        predicted.append(pred)
        valid_assets.append(ticker)
        st.write(f"{ticker}: {pred:.5f}")
    else:
        st.warning(f"Data {ticker} terlalu sedikit untuk LSTM")

if not predicted:
    st.error("Tidak ada aset yang berhasil diprediksi.")
    st.stop()

# Optimisasi portofolio
st.subheader("🧠 Optimisasi Portofolio")
weights = optimize_portfolio(predicted, returns[valid_assets], risk_aversion)

for ticker, weight in zip(valid_assets, weights):
    st.write(f"{ticker}: {weight*100:.2f}%")

# Backtesting
st.subheader("🔁 Backtesting Portofolio")
perf = backtest_portfolio(returns[valid_assets], weights)
st.line_chart(perf)
