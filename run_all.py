"""
Runs the three studies and writes tables to results/ and charts to figures/.

    python run_all.py

The Treasury curve and ETF prices are downloaded on the first run and cached
in data/, so later runs reproduce the same numbers.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lpfin import cash_financing, dedication, mad

ROOT = Path(__file__).parent
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
CURVE_DATE = "2026-10-06"
ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "AGG"]


def cash_study():
    rows = []
    for r in (0.015, 0.025):
        p = cash_financing.solve(r)
        for t in range(6):
            rows.append({"line_rate": r, "month": t + 1, "cash": p.cash[t], "line": p.line[t],
                         "value_of_1_in_month_6_cash": p.month_value[t]})
    pd.DataFrame(rows).to_csv(RES / "cash_plan.csv", index=False, float_format="%.4f")
    sweep = pd.DataFrame(cash_financing.rate_sweep(np.round(np.arange(0, 0.0601, 0.0025), 4)))
    sweep.to_csv(RES / "cash_rate_sweep.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.line_rate * 100, sweep.final_cash, "o-")
    ax2 = ax.twinx()
    ax2.bar(sweep.line_rate * 100, sweep.peak_line, width=0.15, alpha=0.3, color="C1")
    ax.set_xlabel("credit line rate, % a month")
    ax.set_ylabel("cash at end of month 6")
    ax2.set_ylabel("peak line balance (bars)")
    ax.set_title("The line is used until paper is cheaper")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "cash_rate_sweep.png", dpi=120)
    return sweep


def bond_study():
    par = dedication.load_par_curve(CURVE_DATE, cache=DATA / f"par_curve_{CURVE_DATE}.csv")
    bonds = dedication.bonds_from_curve(par)
    liabilities = 10 * 1.03 ** np.arange(10)       # $10m a year, growing 3%
    dfs = dedication.zero_discount_factors(par, 30)
    pv = liabilities @ dfs[:10]

    ded0 = dedication.dedicate(bonds, liabilities, reinvest=0.0)
    ded3 = dedication.dedicate(bonds, liabilities, reinvest=0.03)
    imm = dedication.immunize(bonds, liabilities, dfs)

    holdings = pd.DataFrame({
        "maturity": bonds.maturity,
        "dedicated_no_reinvestment": pd.Series(ded0["quantities"], index=ded0["bonds"].maturity).reindex(bonds.maturity).fillna(0).to_numpy() * 100,
        "dedicated_3pct_reinvestment": pd.Series(ded3["quantities"], index=ded3["bonds"].maturity).reindex(bonds.maturity).fillna(0).to_numpy() * 100,
        "immunized": imm["quantities"] * 100,
    })
    holdings.to_csv(RES / "bond_holdings_face_musd.csv", index=False, float_format="%.3f")

    years = pd.DataFrame({"year": np.arange(1, 11), "liability": liabilities,
                          "curve_discount_factor": dfs[:10],
                          "dual_no_reinvestment": ded0["dual_discount"],
                          "carried_cash_no_reinvestment": ded0["surplus"]})
    years.to_csv(RES / "dedication_duals.csv", index=False, float_format="%.4f")

    scenarios = {
        "no change": lambda t: 0 * t,
        "parallel -100bp": lambda t: -0.01 + 0 * t,
        "parallel +100bp": lambda t: 0.01 + 0 * t,
        "steepener": lambda t: 0.001 * (t - 5),     # 10bp a year around 5y
        "flattener": lambda t: -0.001 * (t - 5),
    }
    rows = []
    for name, f in scenarios.items():
        d = dedication.shifted(dfs, f)
        rows.append({"scenario": name,
                     "dedicated": dedication.surplus(ded0["cash"], liabilities, d),
                     "dedicated_3pct": dedication.surplus(ded3["cash"], liabilities, d),
                     "immunized": dedication.surplus(imm["cash"], liabilities, d)})
    stress = pd.DataFrame(rows)
    stress.to_csv(RES / "bond_stress_surplus_musd.csv", index=False, float_format="%.3f")
    costs = pd.Series({"pv_of_liabilities": pv, "dedicated_no_reinvestment": ded0["cost"],
                       "dedicated_3pct_reinvestment": ded3["cost"], "immunized": imm["cost"]})
    costs.to_frame("cost_musd").to_csv(RES / "bond_costs_musd.csv", float_format="%.3f")
    par.to_frame().to_csv(RES / "par_curve.csv")
    return costs, stress, years


def load_etf_returns():
    path = DATA / "etf_monthly_returns.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(ETFS, start="2003-10-01", end="2025-10-01", interval="1mo",
                     auto_adjust=True, progress=False)["Close"][ETFS]
    rets = px.pct_change().dropna()
    DATA.mkdir(exist_ok=True)
    rets.to_csv(path)
    return rets


def portfolio_study():
    rets = load_etf_returns()
    front = mad.frontier(rets.iloc[-120:].to_numpy())
    front.to_csv(RES / "frontier_last_10y.csv", index=False, float_format="%.5f")
    oos, weights = mad.backtest(rets)
    table = mad.summarise(oos, weights)
    table.to_csv(RES / "portfolio_backtest.csv", float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in front.groupby("model"):
        axes[0].plot(g["vol"] * np.sqrt(12) * 100, g["mean"] * 1200, "o-", ms=3,
                     label=f"minimise {name}")
    axes[0].set_xlabel("annualised volatility, %")
    axes[0].set_ylabel("annualised mean return, %")
    axes[0].set_title("Frontiers on the last ten years")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    (1 + oos).cumprod().plot(ax=axes[1], logy=True)
    axes[1].set_title("Out of sample, 60-month window, 10 bp costs")
    axes[1].set_xlabel("")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "portfolios.png", dpi=120)
    w = weights["minimum MAD"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w.plot.area(ax=ax, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_title("Minimum-MAD weights through time")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG / "mad_weights.png", dpi=120)
    return table, oos


def main():
    for d in (DATA, RES, FIG):
        d.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    print(cash_study().round(4).to_string(), "\n")
    costs, stress, years = bond_study()
    print(costs.round(3), "\n", stress.round(3), "\n", years.round(4), "\n")
    table, oos = portfolio_study()
    print(table.round(4))
    print(f"\nout of sample {oos.index[0]:%b %Y} to {oos.index[-1]:%b %Y}, {len(oos)} months")


if __name__ == "__main__":
    main()
