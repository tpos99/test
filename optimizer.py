# ==== optimizer.py ====
import cvxpy as cp
import numpy as np

def optimize_portfolio(predicted_returns, cov_matrix, gamma=0.1):
    n = len(predicted_returns)
    w = cp.Variable(n)
    ret = predicted_returns @ w
    risk = cp.quad_form(w, cov_matrix)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
    prob.solve()
    return w.value
