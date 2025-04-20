# ==== data_loader.py ====
import yfinance as yf
import pandas as pd

def get_data(tickers, period="2y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval)['Adj Close']
    return data.dropna()
