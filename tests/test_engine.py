"""
Engine correctness tests.

These target the failure modes that silently invalidate a backtest rather than
crashing it: look-ahead leakage, mis-stated costs, degenerate factor definitions.

Run:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import backtest as B, factors as F, portfolio as P
from src.config import BacktestConfig
from src import fundamentals as FU


# ---------------------------------------------------------------------------
# fixtures: a small synthetic price panel with known structure
# ---------------------------------------------------------------------------
@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", "2021-12-31")
    tickers = [f"T{i:02d}" for i in range(40)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.015, (len(dates), len(tickers))),
                        index=dates, columns=tickers)
    prices = 100 * (1 + rets).cumprod()
    volume = pd.DataFrame(1e7, index=dates, columns=tickers)
    mkt = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
    rf = pd.Series(0.00005, index=dates)
    return prices, rets, volume, mkt, rf


CFG = BacktestConfig(n_long=10, min_dollar_volume=0.0, min_price=0.0)


# ---------------------------------------------------------------------------
# 1. Look-ahead
# ---------------------------------------------------------------------------
def test_factors_ignore_future_data(panel):
    """Factor values at date d must not change when data after d is deleted."""
    prices, rets, _, mkt, rf = panel
    d = prices.index[1500]

    full = F.compute_raw_factors(prices, rets, mkt, rf, d, CFG)
    truncated = F.compute_raw_factors(prices.loc[:d], rets.loc[:d],
                                      mkt.loc[:d], rf.loc[:d], d, CFG)
    pd.testing.assert_frame_equal(full, truncated, check_exact=False, atol=1e-12)


def test_eligible_universe_ignores_future(panel):
    prices, _, volume, _, _ = panel
    d = prices.index[1500]
    members = pd.Series(True, index=prices.columns)
    dv = (prices * volume).rolling(5, min_periods=1).mean()

    a = P.eligible_universe(d, prices, prices, dv, members, CFG)
    b = P.eligible_universe(d, prices.loc[:d], prices.loc[:d], dv.loc[:d], members, CFG)
    assert list(a) == list(b)


# ---------------------------------------------------------------------------
# 2. Residual momentum must NOT collapse into short-term reversal
# ---------------------------------------------------------------------------
def test_residual_momentum_is_not_reversal(panel):
    """
    Regression test for a real bug found in this project.

    OLS residuals sum to zero over the estimation window. If residuals are
    accumulated over that SAME window minus the skipped final month, the result is
    algebraically -1 x (the skipped month's residual) -- i.e. a reversal signal
    wearing a momentum label. The fix is a strictly longer estimation window.
    Here we assert the two signals stay economically distinct.
    """
    prices, rets, _, mkt, rf = panel
    d = prices.index[1600]
    raw = F.compute_raw_factors(prices, rets, mkt, rf, d, CFG)
    both = raw[["residual_momentum", "reversal"]].dropna()
    assert len(both) > 20
    corr = both["residual_momentum"].corr(both["reversal"], method="spearman")
    assert abs(corr) < 0.6, f"residual momentum has collapsed into reversal (rho={corr:.3f})"


def test_residual_momentum_degenerates_when_windows_coincide(panel):
    """The bug reproduces when the estimation window equals the accumulation window."""
    prices, rets, _, mkt, rf = panel
    d = prices.index[1600]
    bad = CFG.variant(resmom_beta_window=252, resmom_lookback=252)
    raw = F.compute_raw_factors(prices, rets, mkt, rf, d, bad)
    both = raw[["residual_momentum", "reversal"]].dropna()
    corr = both["residual_momentum"].corr(both["reversal"], method="spearman")
    assert corr > 0.8, "expected the degenerate configuration to mimic reversal"


# ---------------------------------------------------------------------------
# 3. Portfolio construction invariants
# ---------------------------------------------------------------------------
def test_weights_sum_to_one_and_respect_cap():
    scores = pd.Series(np.linspace(-3, 3, 60), index=[f"S{i}" for i in range(60)])
    vols = pd.Series(0.2, index=scores.index)
    for scheme in ("equal", "score_tilt", "inverse_vol"):
        cfg = CFG.variant(weighting=scheme, n_long=20, max_weight=0.08)
        w = P.build_weights(scores, vols, cfg)
        assert len(w) == 20
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w <= 0.08 + 1e-9).all(), f"{scheme} breached the position cap"
        assert (w > 0).all()


def test_cap_falls_back_when_infeasible():
    """A cap below 1/N is infeasible; the builder must degrade to equal weight."""
    scores = pd.Series(np.arange(10, dtype=float), index=[f"S{i}" for i in range(10)])
    vols = pd.Series(0.2, index=scores.index)
    cfg = CFG.variant(n_long=10, max_weight=0.05)   # 10 x 0.05 = 0.5 < 1
    w = P.build_weights(scores, vols, cfg)
    assert w.sum() == pytest.approx(1.0)
    assert w.nunique() == 1


def test_higher_score_gets_higher_weight_under_score_tilt():
    scores = pd.Series([3.0, 2.0, 1.0, 0.0], index=list("ABCD"))
    vols = pd.Series(0.2, index=scores.index)
    cfg = CFG.variant(weighting="score_tilt", n_long=4, max_weight=1.0)
    w = P.build_weights(scores, vols, cfg)
    assert w["A"] > w["B"] > w["C"] > w["D"]


# ---------------------------------------------------------------------------
# 4. Cross-sectional standardisation
# ---------------------------------------------------------------------------
def test_zscore_is_standardised():
    s = pd.Series(np.random.default_rng(1).normal(5, 3, 500))
    z = F.zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-10)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-10)


def test_winsorize_caps_outliers():
    s = pd.Series(list(np.zeros(98)) + [1e6, -1e6])
    w = F.winsorize(s, 0.01)
    assert w.max() < 1e6 and w.min() > -1e6


def test_composite_requires_minimum_factor_coverage():
    """A stock missing most factors must not receive a score."""
    idx = [f"S{i}" for i in range(30)]
    z = pd.DataFrame(np.random.default_rng(3).normal(size=(30, 5)),
                     index=idx, columns=F.FACTOR_NAMES)
    z.loc["S0", F.FACTOR_NAMES[1:]] = np.nan     # only 1 of 5 factors present
    score = F.composite_score(z, CFG.variant(
        factor_weights={k: 0.2 for k in F.FACTOR_NAMES}))
    assert np.isnan(score["S0"])
    assert score.drop("S0").notna().all()


# ---------------------------------------------------------------------------
# 5. Cost accounting
# ---------------------------------------------------------------------------
def test_zero_cost_run_matches_gross(panel):
    prices, _, volume, mkt, rf = panel
    ff = pd.DataFrame({"Mkt-RF": mkt, "RF": rf})
    cfg = CFG.variant(start="2017-01-01", end="2019-12-31",
                      commission_bps=0, spread_bps=0, slippage_bps=0)
    res = _run_isolated(prices, volume, ff, cfg)
    pd.testing.assert_series_equal(res.returns, res.gross_returns,
                                   check_names=False, atol=1e-12)


def test_costs_reduce_returns_monotonically(panel):
    prices, _, volume, mkt, rf = panel
    ff = pd.DataFrame({"Mkt-RF": mkt, "RF": rf})
    outs = []
    for bps in (0, 10, 40):
        cfg = CFG.variant(start="2017-01-01", end="2019-12-31",
                          commission_bps=bps, spread_bps=0, slippage_bps=0)
        outs.append((1 + _run_isolated(prices, volume, ff, cfg).returns).prod())
    assert outs[0] > outs[1] > outs[2]


def test_cost_equals_turnover_times_bps(panel):
    prices, _, volume, mkt, rf = panel
    ff = pd.DataFrame({"Mkt-RF": mkt, "RF": rf})
    cfg = CFG.variant(start="2017-01-01", end="2019-12-31",
                      commission_bps=10, spread_bps=0, slippage_bps=0)
    res = _run_isolated(prices, volume, ff, cfg)
    expected = res.turnover * 10 / 1e4
    pd.testing.assert_series_equal(res.costs, expected, check_names=False, atol=1e-12)


def _run_isolated(prices, volume, ff, cfg):
    """Run the engine with membership stubbed out to 'everything is a member'."""
    import src.backtest as bt
    original = bt.membership_matrix
    bt.membership_matrix = lambda dates: pd.DataFrame(
        True, index=dates, columns=prices.columns)
    try:
        return bt.run_backtest(prices, prices, volume, ff, cfg)
    finally:
        bt.membership_matrix = original


# ---------------------------------------------------------------------------
# 6. Execution timing
# ---------------------------------------------------------------------------
def test_execution_lag_delays_trades(panel):
    """With a lag, the first rebalance's weights cannot earn that day's return."""
    prices, _, volume, mkt, rf = panel
    ff = pd.DataFrame({"Mkt-RF": mkt, "RF": rf})
    cfg = CFG.variant(start="2017-01-01", end="2017-06-30", execution_lag=1)
    res = _run_isolated(prices, volume, ff, cfg)
    first_rebal = res.weights.index[0]
    # nothing is held before the first execution, so the return that day is zero
    assert res.gross_returns.loc[:first_rebal].abs().sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. Point-in-time fundamentals
# ---------------------------------------------------------------------------
def test_as_of_respects_filing_date():
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "period_end": pd.to_datetime(["2020-12-31", "2021-03-31"]),
        "filing_date": pd.to_datetime(["2021-02-15", "2021-05-10"]),
        "book_equity": [100.0, 200.0],
    })
    # on 2021-03-01 only the first filing is public, even though Q1 has ended
    snap = FU.as_of(df, pd.Timestamp("2021-03-01"))
    assert snap.loc["A", "book_equity"] == 100.0
    snap2 = FU.as_of(df, pd.Timestamp("2021-06-01"))
    assert snap2.loc["A", "book_equity"] == 200.0


def test_load_rejects_filing_before_period_end(tmp_path):
    df = pd.DataFrame({k: [1] for k in FU.REQUIRED_SCHEMA})
    df["ticker"] = ["A"]
    df["period_end"] = pd.to_datetime(["2021-03-31"])
    df["filing_date"] = pd.to_datetime(["2021-01-01"])   # impossible
    p = tmp_path / "f.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="look-ahead"):
        FU.load_fundamentals(p)


def test_synthetic_guard_blocks_reporting():
    df = FU.make_synthetic_fundamentals(["A", "B"], "2020-01-01", "2021-01-01")
    assert FU.SYNTHETIC_FLAG in df.columns
    with pytest.raises(FU.SyntheticDataError):
        FU.assert_not_synthetic(df, "unit test")


# ---------------------------------------------------------------------------
# 8. Metrics
# ---------------------------------------------------------------------------
def test_metrics_on_known_series():
    from src import metrics as M
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    assert M.cagr(r) == pytest.approx((1.001 ** 252) - 1, rel=1e-6)
    assert M.ann_vol(r) == pytest.approx(0.0, abs=1e-12)
    assert M.max_drawdown(r) == pytest.approx(0.0, abs=1e-12)
    assert M.win_rate(r) == 1.0


def test_drawdown_matches_hand_calculation():
    from src import metrics as M
    r = pd.Series([0.5, -0.5, 0.0], index=pd.bdate_range("2020-01-01", periods=3))
    # equity: 1.5, 0.75, 0.75 -> peak 1.5, trough 0.75 -> -50%
    assert M.max_drawdown(r) == pytest.approx(-0.5, abs=1e-12)
