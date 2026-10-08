# optimization-in-finance

Three linear programs from corporate finance and asset management, each solved with open-source solvers and checked against something independent. They grew out of Optimization in Finance at the University of Illinois. `coursework/` keeps the original Gurobi notebook and the LaTeX write-up of the first problem.

## 1. Financing six months of cash flows

`lpfin/cash_financing.py`. A firm's operating cash flows over six months are 50, -100, -150, 200, -300 and 400. It can draw up to 100 on a credit line at a monthly rate, and issue five commercial papers with different issue months, maturities and coupons. Cash earns nothing. The program maximises cash at the end of month 6, with cash never negative and the line repaid at the end.

At a 1.5% line rate the best plan ends with 93.735. It draws 50 in month 2, issues 200.75 of the 2% paper that runs from month 3 to month 6, and draws the full line in month 5. At 2.5% it ends with 92.725 and covers month 5 with more of the same paper. Both match the Gurobi solution in `coursework/`.

The dual prices on each month's cash equation are the useful part. A dollar arriving in months 3 to 5 is worth exactly 1.02 at month 6, because it replaces a dollar of the 2% paper. A dollar arriving in month 1 or 2 is worth 1.0353 at a 1.5% line rate, the paper's 2% plus the line interest it saves, rising to 1.0455 at 2.5%. They are the firm's marginal cost of funding from each month to the end.

`results/cash_rate_sweep.csv` reprices the plan for line rates from 0% to 6%. The line is fully used up to 2%, half used at 2.25% and 2.5%, and dropped from 2.75%, at which point the 2.5% three-month paper takes its place. From 2.5% up, final cash stays at 92.725, because the line no longer changes the plan's cost.

## 2. Funding a pension stream with Treasuries

`lpfin/dedication.py`. The liability is $10m a year for ten years, growing 3% a year, worth $86.85m on the Treasury curve of 6 October 2026, built from FRED par yields. The bonds available are par bonds at the eight benchmark maturities: 1, 2, 3, 5, 7, 10, 20 and 30 years.

Dedication is the cheapest portfolio whose cash flows pay every liability when due, with any surplus carried forward. Immunization is the cheapest portfolio that only matches present value and duration on the curve, with convexity at least the liabilities'.

| Strategy | Cost, $m |
|---|---|
| Present value of the liabilities | 86.85 |
| Dedication, surplus earns nothing | 88.86 |
| Dedication, surplus earns 3% | 87.73 |
| Immunization | 86.85 |

Dedication costs $2.0m more because the maturity ladder has gaps. Money needed in year 4 has to be bought with the 3-year bond and held, and the duals show it: the dual for year 4 equals year 3's, 0.8666, against a curve discount factor of 0.8237. Years 8 and 9 sit on year 7's dual for the same reason.

Immunization buys a barbell of the 1-year and 20-year bonds at exactly present value. Then the curve moves, in $m of surplus:

| Scenario | Dedicated | Immunized |
|---|---|---|
| No change | +2.00 | 0.00 |
| Parallel -100bp | +1.74 | +0.19 |
| Parallel +100bp | +2.22 | +0.16 |
| Steepener, 10bp a year around 5 years | +2.28 | -2.98 |
| Flattener | +1.71 | +4.16 |

Immunization does what it promises for parallel moves, and convexity even leaves a small gain either way. A steepener, the move it does not hedge, costs it $3m, 3.4% of the liabilities, because the barbell's 20-year leg falls while the liabilities sit in the middle of the curve. Dedication never runs short.

## 3. Mean absolute deviation portfolios

`lpfin/mad.py`. Konno and Yamazaki's model minimises the mean absolute deviation of returns, which makes portfolio choice a linear program with no covariance matrix. It is compared with long-only minimum variance on the nine original SPDR sector funds plus AGG, using monthly returns since 2003.

On the last ten years in sample, the two frontiers nearly coincide. At any target return, the MAD portfolio's volatility is within 2.6% of the minimum-variance portfolio's.

Out of sample, each strategy re-estimates every month on the previous 60 months and pays 10bp on turnover. From November 2008 to September 2025:

| | Return | Volatility | Worst month | Max drawdown | Turnover a year | Funds held |
|---|---|---|---|---|---|---|
| Minimum MAD | 3.7% | 4.7% | -4.4% | 15.7% | 0.35 | 2.7 |
| Minimum variance | 3.6% | 4.7% | -4.4% | 15.6% | 0.18 | 2.6 |
| Equal weight | 12.0% | 13.5% | -13.2% | 21.1% | 0 | 10 |

The two minimum-risk portfolios deliver almost exactly the same risk. The LP buys that without estimating a covariance matrix, but it trades twice as much: its solutions sit on corners of the feasible set and jump between them as the window rolls, where the quadratic solution moves smoothly. Both put about 90% in AGG on average, which is why they hold fewer than three funds and earn a bond-like return. Equal weight is shown for scale; its monthly rebalancing back to equal weights is not charged.

## Running it

```
pip install numpy scipy pandas matplotlib yfinance
python -m pytest tests          # or: python tests/test_lpfin.py
python run_all.py               # about ten seconds once the data is cached
```

The Treasury curve and ETF returns are cached in `data/` after the first run.
