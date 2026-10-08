"""Run from the repo root: python -m pytest tests  (or python tests/test_lpfin.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpfin import cash_financing, dedication, mad


def test_cash_plan_matches_the_gurobi_solution():
    assert np.isclose(cash_financing.solve(0.015).final_cash, 93.735)
    assert np.isclose(cash_financing.solve(0.025).final_cash, 92.725)


def test_cash_duals_are_funding_costs():
    p = cash_financing.solve(0.015)
    # cash arriving in months 3-5 saves issuing the 2% paper that runs to month 6
    assert np.allclose(p.month_value[2:5], 1.02)
    assert np.isclose(p.month_value[5], 1.0)
    # every month's balance equation holds
    assert np.all(p.cash >= -1e-9) and np.all(p.line <= 100 + 1e-9) and abs(p.line[-1]) < 1e-9


CURVE = pd.Series({1.0: 0.045, 2.0: 0.047, 3.0: 0.048, 5.0: 0.050, 7.0: 0.051,
                   10.0: 0.053, 20.0: 0.057, 30.0: 0.056}, name="par_yield")
LIAB = 10 * 1.03 ** np.arange(10)


def test_bonds_price_at_par_on_their_own_curve():
    bonds = dedication.bonds_from_curve(CURVE, maturities=range(1, 31))
    dfs = dedication.zero_discount_factors(CURVE, 30)
    C = dedication.cash_flow_matrix(bonds, 30)
    assert np.allclose(dfs @ C, 100.0)


def test_dedication_with_every_maturity_costs_exactly_present_value():
    bonds = dedication.bonds_from_curve(CURVE, maturities=range(1, 11))
    dfs = dedication.zero_discount_factors(CURVE, 10)
    ded = dedication.dedicate(bonds, LIAB)
    assert np.isclose(ded["cost"], LIAB @ dfs)
    assert np.allclose(ded["dual_discount"], dfs, atol=1e-6)


def test_dedication_with_gaps_overfunds_and_pays_every_liability():
    bonds = dedication.bonds_from_curve(CURVE)
    ded = dedication.dedicate(bonds, LIAB)
    dfs = dedication.zero_discount_factors(CURVE, 30)
    assert ded["cost"] > LIAB @ dfs[:10]
    carried = np.concatenate([[0.0], ded["surplus"][:-1]])
    assert np.allclose(ded["cash"] + carried - ded["surplus"], LIAB, atol=1e-6)


def test_immunized_portfolio_matches_pv_and_duration():
    bonds = dedication.bonds_from_curve(CURVE)
    dfs = dedication.zero_discount_factors(CURVE, 30)
    imm = dedication.immunize(bonds, LIAB, dfs)
    t = np.arange(1, 31)
    L = np.zeros(30)
    L[:10] = LIAB
    assert np.isclose(imm["cash"] @ dfs, L @ dfs)
    assert np.isclose((t * imm["cash"]) @ dfs, (t * L) @ dfs)


def _returns():
    rng = np.random.default_rng(0)
    return rng.multivariate_normal([0.005, 0.007, 0.004, 0.003],
                                   np.diag([0.04, 0.05, 0.03, 0.01]) ** 2, size=120)


def test_mad_lp_minimises_mean_absolute_deviation():
    r = _returns()
    w = mad.min_mad(r)
    p = r @ w
    best = np.abs(p - p.mean()).mean()
    eq = r.mean(axis=1)
    assert np.isclose(w.sum(), 1) and w.min() >= -1e-9
    assert best <= np.abs(eq - eq.mean()).mean() + 1e-12
    for _ in range(200):
        x = np.random.default_rng(_).dirichlet(np.ones(4))
        q = r @ x
        assert best <= np.abs(q - q.mean()).mean() + 1e-10


def test_return_target_is_respected():
    r = _returns()
    target = r.mean(axis=0).max() * 0.9
    for solver in (mad.min_mad, mad.min_variance):
        assert r.mean(axis=0) @ solver(r, target) >= target - 1e-8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
