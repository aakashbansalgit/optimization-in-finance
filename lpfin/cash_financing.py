"""
Six-month cash financing plan as a linear program.

A firm knows its operating cash flows for the next six months. It can draw on
a revolving credit line, up to 100, at a monthly rate r, and issue five
commercial papers, each with its own issue month, maturity and coupon. Cash
earns nothing. Maximise cash at the end of month 6, keeping cash
non-negative every month and the credit line repaid by the end.

Variables: s_t cash at the end of month t, b_t line balance at the end of
month t, y_j face of paper j. Each month's balance equation is

    s_t = s_{t-1} + CF_t + new paper - maturing paper (1 + c_j) + (b_t - b_{t-1}) - r b_{t-1}

The dual price on month t's equation is what one more unit of cash arriving
in month t is worth in month-6 cash, which is the firm's marginal cost of
funding between t and 6.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

CF = [50, -100, -150, 200, -300, 400]
CAPACITY = 100.0
PAPERS = [  # (issue month, maturity month, coupon for the whole term)
    (1, 2, 0.020),
    (1, 4, 0.025),
    (2, 5, 0.035),
    (3, 6, 0.020),
    (4, 6, 0.030),
]


@dataclass
class Plan:
    rate: float
    final_cash: float
    cash: np.ndarray
    line: np.ndarray
    paper: np.ndarray
    month_value: np.ndarray = field(repr=False)   # dual price of each month's cash


def solve(rate, cash_flows=CF, capacity=CAPACITY, papers=PAPERS):
    T, J = len(cash_flows), len(papers)
    n = 2 * T + J                      # s_1..s_T, b_1..b_T, y_1..y_J
    s = lambda t: t - 1
    b = lambda t: T + t - 1
    y = lambda j: 2 * T + j

    A, rhs = np.zeros((T + 1, n)), np.zeros(T + 1)
    for t in range(1, T + 1):
        row = A[t - 1]
        row[s(t)] = 1.0
        row[b(t)] = -1.0
        if t > 1:
            row[s(t - 1)] = -1.0
            row[b(t - 1)] = 1.0 + rate
        for j, (issue, mature, coupon) in enumerate(papers):
            if issue == t:
                row[y(j)] -= 1.0
            if mature == t:
                row[y(j)] += 1.0 + coupon
        rhs[t - 1] = cash_flows[t - 1]
    A[T, b(T)] = 1.0                   # line repaid at the end

    cost = np.zeros(n)
    cost[s(T)] = -1.0                  # maximise final cash
    bounds = [(0, None)] * T + [(0, capacity)] * T + [(0, None)] * J
    res = linprog(cost, A_eq=A, b_eq=rhs, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x
    return Plan(rate, x[s(T)], x[:T], x[T:2 * T], x[2 * T:],
                -res.eqlin.marginals[:T])


def rate_sweep(rates):
    """Final cash and the financing mix as the credit line gets dearer."""
    rows = []
    for r in rates:
        p = solve(r)
        rows.append({"line_rate": r, "final_cash": p.final_cash,
                     "peak_line": p.line.max(), "paper_issued": p.paper.sum(),
                     **{f"paper_{j + 1}": v for j, v in enumerate(p.paper)}})
    return rows
