# ==== backtest.py ====
def backtest_portfolio(weights, returns):
    daily_portfolio_return = (returns * weights).sum(axis=1)
    cumulative_return = (1 + daily_portfolio_return).cumprod()
    return cumulative_return
