# ==== lstm_model.py ====
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(series, lookback=60):
    if len(series) <= lookback:
        raise ValueError(f"Series terlalu pendek untuk lookback={lookback}, panjang={len(series)}")

    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def predict_next_return(model, series, lookback=60):
    if len(series) < lookback:
        return 0  # Default prediksi jika data tidak cukup
    last_sequence = series[-lookback:].reshape((1, lookback, 1))
    return model.predict(last_sequence, verbose=0)[0][0]
