# Systematic Multi-Factor Equity Strategy — Research Project

A full research pipeline for building, backtesting and **falsifying** a systematic
multi-factor equity strategy on the S&P 500, using point-in-time index membership,
realistic trading frictions, and an out-of-sample protocol designed to make the
strategy fail if it deserves to.

**Headline finding: the hypothesis is not supported.** The composite does not select
stocks better than chance within its universe, and its factor exposures fully explain
its returns. Details in [`REPORT.md`](REPORT.md).

The report is also published as a single self-contained page:
**[read the illustrated version](https://claude.ai/code/artifact/d6d5b284-7b34-4fbd-b6ec-15afe6d17f15)**.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

python scripts/01_download_data.py     # ~10 min, caches to data/interim/*.parquet
python scripts/02_factor_analysis.py   # in-sample factor efficacy + survivorship audit
python scripts/03_backtest.py          # headline backtests + attribution
python scripts/04_robustness.py        # controls, sweeps, walk-forward, bootstrap
python scripts/05_figures.py           # all figures -> results/figures/
python scripts/06_fundamentals_demo.py # PIT schema demo + synthetic guard
python scripts/07_build_artifact.py    # render REPORT.md -> report_artifact.html

python -m pytest tests/ -v             # 49 tests
```

Every script is idempotent; re-running uses the parquet cache in `data/interim/`.

---

## Repository layout

```
src/
  config.py        BacktestConfig -- a run is fully specified by this dataclass
  data.py          point-in-time membership, prices, Fama-French factors, survivorship audit
  factors.py       factor mathematics, winsorisation, z-scoring, compositing
  fundamentals.py  value/quality schema + PIT requirements (NOT backtested -- see below)
  portfolio.py     universe screens, weighting schemes, position caps
  backtest.py      event-timed engine with costs, liquidity limits, IC and decile tests
  metrics.py       performance stats, Newey-West, PSR / Deflated Sharpe
  attribution.py   CAPM / FF3 / Carhart / FF5 / FF5+MOM regressions with HAC errors
  robustness.py    sweeps, regimes, cost break-even, walk-forward, bootstrap, null test
  plots.py         figures
  specs.py         FROZEN strategy specifications (weights fixed before OOS was touched)
scripts/           01..07, run in order
tests/
  test_engine.py     engine correctness -- look-ahead, costs, caps, PIT guard
  test_inference.py  attribution, HAC, Deflated Sharpe, controls, frozen specs
results/
  tables/          every CSV referenced in the report
  figures/         every PNG referenced in the report
```

`data/interim/` is a derived cache and is not tracked; `scripts/01_download_data.py`
rebuilds it from the tracked sources in `data/raw/` plus Yahoo Finance (~10 min).

---

## Data

| Dataset | Source | Status |
|---|---|---|
| S&P 500 point-in-time membership, 1996–2026 | `fja05680/sp500` (index change records) | real |
| Daily split/dividend-adjusted prices | Yahoo Finance via `yfinance` | real, 767 tickers |
| Fama-French 5 factors + momentum, daily | Ken French Data Library | real, through 2026-06 |
| Benchmark | SPY adjusted close | real |
| **Point-in-time fundamentals** | — | **NOT AVAILABLE — see below** |

### Two data limitations that are stated, measured, and not worked around

**1. Survivorship bias is present and quantified, not eliminated.**
Point-in-time membership tells us exactly which tickers were in the index on any date,
so the *universe* has no look-ahead. But Yahoo Finance serves no price history for
delisted securities — verified directly: `SIVB`, `FRC`, `ATVI`, `TWTR`, `CERN`, `XLNX`,
`LEH` all return zero rows. Of 1,206 tickers that were ever S&P 500 members, only 767
can be priced.

`data.audit_survivorship()` measures the resulting hole at every rebalance:
coverage runs from **46% in 1999 to 99% in 2026**, averaging 71%. The early sample is
therefore materially biased toward survivors, and — critically — the bias is *strongest
in the in-sample window*, which is where the strategy appears to work. This is treated
as a primary finding, not a footnote.

**2. Value and quality factors are specified but not backtested.**
A non-look-ahead value factor needs fundamentals keyed by **filing date** (not fiscal
period end) and **never restated**. `yfinance` supplies 5 annual periods keyed by period
end with no filing date and post-restatement values — verified directly. That is
unusable for a 27-year backtest.

Rather than report an inflated Sharpe from a look-ahead-contaminated value factor,
`src/fundamentals.py` specifies the required schema, implements the full factor
mathematics against it, and ships a clearly-labelled synthetic generator with a hard
guard (`assert_not_synthetic`) that raises if synthetic output is ever used for a
reported result. To activate value/quality, connect a real PIT feed — Sharadar SF1
(`datekey`, retains delisted tickers) or Compustat PIT via WRDS.

---

## Method summary

- **Universe** — S&P 500 members on the rebalance date, ≥$5 close, ≥$5m 60-day median
  dollar volume, ≥252 days of history. Mean eligible universe: 348 names.
- **Factors** — momentum (12-1), residual momentum, low volatility, low beta,
  short-term reversal. All price-derived, so all strictly point-in-time.
- **Compositing** — winsorise at 1%/99%, cross-sectionally z-score, weighted sum,
  re-standardise.
- **Portfolio** — top 50 by composite score, score-tilted weights, 5% position cap,
  monthly rebalance.
- **Timing** — signal uses data through the close of *t*; trades execute at the close of
  *t+1*. Positions drift with prices between rebalances.
- **Costs** — 8 bps one-way (1 commission + 5 spread + 2 slippage), charged on realised
  turnover. Trades are throttled to 5% of 60-day ADV against a $100m AUM assumption;
  the unfilled remainder stays in the prior position.

### Protocol against overfitting

| Guard | Implementation |
|---|---|
| In-sample / out-of-sample split | design on 1999–2012; 2013–2026 untouched until step 3 |
| Frozen specifications | `src/specs.py` stores weights as literals, not recomputed |
| A priori control | equal-weighted composite uses **no** fitted information at all |
| Walk-forward | weights re-fitted on 6y train, traded on next 2y, rolled forward |
| Random-portfolio null | 500 random 50-stock portfolios from the same universe |
| Equal-weight universe control | isolates factor *selection* from the equal-weighting tilt |
| Multiple-testing correction | Deflated Sharpe Ratio against every trial in the sweep |
| HAC inference | Newey-West throughout; stationary block bootstrap for Sharpe CIs |

---

## Reproducibility

- Randomness is confined to `robustness.random_portfolio_null` and
  `stationary_bootstrap_sharpe`, both seeded (`seed=42`).
- A backtest is fully determined by one frozen `BacktestConfig` plus the cached data.
- Package versions pinned in `requirements.txt`.
- Data caches are content-stable: re-running `01_download_data.py` will pick up any
  new Yahoo history, so pin `DOWNLOAD_END` in `src/config.py` to reproduce exactly.

## Known limitations

See [`REPORT.md`](REPORT.md) § Limitations. The four that matter most: survivorship
bias concentrated in the in-sample window; no point-in-time fundamentals; a single
market (US large cap) and therefore one macro history; and transaction costs modelled
as a constant spread rather than a volatility- and size-dependent impact function.
