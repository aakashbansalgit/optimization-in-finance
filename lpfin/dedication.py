"""
Funding a stream of liabilities with Treasury bonds: dedication against
immunization.

Dedication buys the cheapest portfolio whose cash flows, with any surplus
carried forward at a conservative reinvestment rate, pay every liability
when it falls due. It is a linear program, and its dual prices are the
implied value today of a dollar needed in each year.

Immunization only matches present value and duration, which is cheaper in
constraints but leaves the portfolio exposed to the curve changing shape.

The bond universe is a set of annual-pay bonds built off the Treasury par
curve on one date from FRED, one at each benchmark maturity (1, 2, 3, 5, 7,
10, 20 and 30 years), each with a coupon equal to its par yield so it prices
at 100. With gaps between maturities, dedication cannot match each year
exactly and has to carry cash forward.
"""
from math import exp

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import linprog

TENORS = {"DGS1": 1, "DGS2": 2, "DGS3": 3, "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}


def load_par_curve(date, cache=None):
    """Par yields (decimal) by tenor in years from FRED, on the last trading
    day on or before `date`."""
    if cache is not None and cache.exists():
        s = pd.read_csv(cache, index_col=0)["par_yield"]
        s.index = s.index.astype(float)
        return s
    out = {}
    for sid, years in TENORS.items():
        d = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                        index_col=0, parse_dates=True).iloc[:, 0]
        d = pd.to_numeric(d, errors="coerce").dropna()
        out[float(years)] = d[:date].iloc[-1] / 100
    s = pd.Series(out, name="par_yield").sort_index()
    if cache is not None:
        cache.parent.mkdir(exist_ok=True)
        s.to_frame().to_csv(cache)
    return s


ON_THE_RUN = (1, 2, 3, 5, 7, 10, 20, 30)


def bonds_from_curve(par, maturities=ON_THE_RUN):
    """One par bond per whole-year maturity, coupon set to the interpolated
    par yield, so each prices at 100."""
    f = interp1d(par.index, par.values)
    return pd.DataFrame([{"maturity": float(m), "coupon": float(f(m)), "price": 100.0}
                         for m in maturities])


def cash_flow_matrix(bonds, horizon):
    """Rows are years 1..horizon, columns are bonds, per 100 face."""
    C = np.zeros((horizon, len(bonds)))
    for j, b in bonds.iterrows():
        for yr in range(1, int(b.maturity) + 1):
            C[yr - 1, j] = 100 * b.coupon
        C[int(b.maturity) - 1, j] += 100
    return C


def dedicate(bonds, liabilities, reinvest=0.0):
    """Cheapest bond portfolio that meets every liability.

    Variables are bond quantities x_j and the surplus z_t carried from year t
    to t+1 at the reinvestment rate. Each year: bond cash + carried surplus
    from last year - surplus carried on = liability."""
    # bonds that outlive the liabilities would have to be sold at an unknown
    # price, so dedication only buys bonds that mature within the horizon
    bonds = bonds[bonds.maturity <= len(liabilities)].reset_index(drop=True)
    H, J = len(liabilities), len(bonds)
    C = cash_flow_matrix(bonds, H)
    A = np.zeros((H, J + H))
    A[:, :J] = C
    for t in range(H):
        A[t, J + t] = -1.0
        if t > 0:
            A[t, J + t - 1] = 1.0 + reinvest
    cost = np.concatenate([bonds["price"].to_numpy(), np.zeros(H)])
    res = linprog(cost, A_eq=A, b_eq=liabilities, bounds=[(0, None)] * (J + H), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return {"cost": res.fun, "bonds": bonds, "quantities": res.x[:J], "surplus": res.x[J:],
            "dual_discount": res.eqlin.marginals, "cash": C @ res.x[:J]}


def immunize(bonds, liabilities, dfs):
    """Cheapest portfolio that matches the liabilities' present value and
    duration on the zero curve, with convexity at least theirs. Duration and
    convexity here are the Fisher-Weil versions, sums of t and t^2 times the
    discounted cash flows, so every condition is linear in the holdings."""
    H = len(liabilities)
    C = cash_flow_matrix(bonds, max(H, int(bonds.maturity.max())))
    t = np.arange(1, C.shape[0] + 1)
    df = dfs[:C.shape[0]]
    pv_b, dur_b, cvx_b = df @ C, (t * df) @ C, (t * t * df) @ C
    L = np.zeros(C.shape[0])
    L[:H] = liabilities
    pv_l, dur_l, cvx_l = L @ df, (t * L) @ df, (t * t * L) @ df
    res = linprog(bonds["price"].to_numpy(),
                  A_eq=np.vstack([pv_b, dur_b]), b_eq=[pv_l, dur_l],
                  A_ub=[-cvx_b], b_ub=[-cvx_l], bounds=[(0, None)] * len(bonds), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return {"cost": res.fun, "quantities": res.x, "cash": C @ res.x}


def shifted(dfs, shift):
    """Discount factors after moving the zero rate at maturity t by shift(t)."""
    t = np.arange(1, len(dfs) + 1)
    return dfs * np.exp(-shift(t) * t)


def surplus(asset_cash, liabilities, dfs):
    """Present value of assets minus liabilities."""
    n = len(dfs)
    a = np.zeros(n)
    a[:len(asset_cash)] = asset_cash
    L = np.zeros(n)
    L[:len(liabilities)] = liabilities
    return float((a - L) @ dfs)


def zero_discount_factors(par, horizon):
    """Annual zero discount factors bootstrapped from annual-pay par yields on
    an interpolated curve. Used to value both sides under curve shifts."""
    f = interp1d(par.index, par.values, fill_value="extrapolate")
    dfs = []
    for n in range(1, horizon + 1):
        c = float(f(max(n, par.index.min())))
        dfs.append((1 - c * sum(dfs)) / (1 + c))
    return np.array(dfs)
