# ==== optimizer.py ====
import numpy as np
import cvxpy as cp

def optimize_portfolio(expected_returns, returns_data, risk_aversion=0.5):
    n = len(expected_returns)
    
    if returns_data.shape[1] != n:
        raise ValueError(f"Mismatch: {n} prediksi, tapi data return hanya {returns_data.shape[1]} kolom.")

    # Variabel bobot
    w = cp.Variable(n)

    # Matriks kovarians
    cov_matrix = np.cov(returns_data.T)

    # Fungsi objektif: maximize expected return - λ * risk
    expected = expected_returns
    risk = cp.quad_form(w, cov_matrix)
    ret = expected @ w
    objective = cp.Maximize(ret - risk_aversion * risk)

    # Batasan: total bobot = 1, dan semua >= 0 (long only)
    constraints = [cp.sum(w) == 1, w >= 0]

    # Optimisasi
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value

