# ==== data_loader.py ====
import yfinance as yf
import pandas as pd

def get_data(tickers, period="2y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker')
    data = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers if 'Adj Close' in data[ticker]})
    return data.dropna()
