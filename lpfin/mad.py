"""
Mean absolute deviation portfolios (Konno and Yamazaki, 1991).

Minimising the mean absolute deviation of portfolio returns from their mean
is a linear program, where minimising variance is quadratic. Over T
historical months and N assets, with d_t the absolute deviation in month t:

    minimise   (1/T) sum_t d_t
    subject to d_t >= +(r_t - rbar) . w
               d_t >= -(r_t - rbar) . w
               rbar . w >= target return      (optional)
               sum w = 1, w >= 0

MAD needs no covariance matrix and treats losses linearly rather than
squaring them. Under normal returns it picks the same portfolios as
mean-variance; on real returns the question is whether it does as well out of
sample, which is what backtest() measures.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


def min_mad(returns: np.ndarray, target=None):
    T, N = returns.shape
    dev = returns - returns.mean(axis=0)
    cost = np.concatenate([np.zeros(N), np.full(T, 1.0 / T)])
    A_ub = np.vstack([np.hstack([dev, -np.eye(T)]),
                      np.hstack([-dev, -np.eye(T)])])
    b_ub = np.zeros(2 * T)
    if target is not None:
        A_ub = np.vstack([A_ub, np.concatenate([-returns.mean(axis=0), np.zeros(T)])])
        b_ub = np.append(b_ub, -target)
    A_eq = np.concatenate([np.ones(N), np.zeros(T)])[None, :]
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (N + T), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:N]


def min_variance(returns: np.ndarray, target=None):
    """Long-only minimum variance, solved with SLSQP; small enough for it."""
    N = returns.shape[1]
    cov = np.cov(returns, rowvar=False)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        mu = returns.mean(axis=0)
        cons.append({"type": "ineq", "fun": lambda w: w @ mu - target})
    res = minimize(lambda w: w @ cov @ w, np.full(N, 1 / N), jac=lambda w: 2 * cov @ w,
                   bounds=[(0, 1)] * N, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x


def frontier(returns: np.ndarray, points=15):
    """In-sample frontiers for both risk measures, from the minimum-risk
    portfolio up to the highest-return asset."""
    mu = returns.mean(axis=0)
    lo = min(min_mad(returns) @ mu, min_variance(returns) @ mu)
    rows = []
    for target in np.linspace(lo, mu.max() * 0.999, points):
        for name, solver in (("MAD", min_mad), ("variance", min_variance)):
            w = solver(returns, target)
            p = returns @ w
            rows.append({"target": target, "model": name, "mean": p.mean(),
                         "mad": np.abs(p - p.mean()).mean(), "vol": p.std()})
    return pd.DataFrame(rows)


def backtest(returns: pd.DataFrame, window=60, tx_cost=0.001):
    """Re-estimate each month on the previous `window` months and hold the
    result for the next month. Every strategy pays the same cost on
    turnover."""
    strategies = {"minimum MAD": min_mad, "minimum variance": min_variance,
                  "equal weight": lambda r: np.full(r.shape[1], 1 / r.shape[1])}
    out = {k: [] for k in strategies}
    weights = {k: [] for k in strategies}
    dates = returns.index[window:]
    for i, date in enumerate(dates):
        hist = returns.iloc[i:i + window].to_numpy()
        nxt = returns.loc[date].to_numpy()
        for name, f in strategies.items():
            w = f(hist)
            prev = weights[name][-1] if weights[name] else w
            turnover = np.abs(w - prev).sum()
            out[name].append(w @ nxt - tx_cost * turnover)
            weights[name].append(w)
    rets = pd.DataFrame(out, index=dates)
    w_frames = {k: pd.DataFrame(v, index=dates, columns=returns.columns) for k, v in weights.items()}
    return rets, w_frames


def summarise(rets: pd.DataFrame, weights: dict) -> pd.DataFrame:
    rows = {}
    for name in rets:
        r = rets[name]
        nav = (1 + r).cumprod()
        w = weights[name]
        rows[name] = {
            "annual_return": nav.iloc[-1] ** (12 / len(r)) - 1,
            "annual_vol": r.std() * np.sqrt(12),
            "mad_monthly": (r - r.mean()).abs().mean(),
            "worst_month": r.min(),
            "max_drawdown": (1 - nav / nav.cummax()).max(),
            "annual_turnover": w.diff().abs().sum(axis=1).mean() * 12,
            "avg_holdings": (w > 0.01).sum(axis=1).mean(),
        }
    return pd.DataFrame(rows).T
