# ==== data_loader.py ====
import yfinance as yf
import pandas as pd

def get_data(tickers, period="1y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True)

    all_close = {}
    for ticker in tickers:
        try:
            close_series = data[ticker]["Close"]
            if not close_series.empty:
                all_close[ticker] = close_series
        except Exception as e:
            print(f"❌ Gagal mengambil data untuk {ticker}: {e}")

    if not all_close:
        return pd.DataFrame()  # Tidak ada data valid

    df = pd.DataFrame(all_close)
    df.dropna(inplace=True)
    return df
