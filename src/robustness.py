r"""
Robustness and overfitting diagnostics.

A backtest that only works at one parameter setting is a curve fit. These tests ask
whether the result is a *plateau* (robust) or a *spike* (fitted):

    param_sweep          vary one knob at a time, hold everything else fixed
    subperiod_analysis   split the sample into regimes and re-measure
    cost_breakeven       find the trading cost at which alpha disappears
    walk_forward         re-fit factor weights on a rolling window, trade the next
                         window only -- no parameter is ever informed by its own
                         evaluation period
    bootstrap_sharpe     stationary block bootstrap CI for the Sharpe ratio
    trial_sharpes        collect every Sharpe examined, for the Deflated Sharpe Ratio
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import backtest as B
from . import factors as F
from . import metrics as M
from .config import BacktestConfig, TRADING_DAYS


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------
def param_sweep(adj, cl, vol, ff, base: BacktestConfig,
                grid: dict[str, list], benchmark: pd.Series | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """One-at-a-time sensitivity: for each parameter, run every value in its list."""
    rows = []
    for param, values in grid.items():
        for v in values:
            cfg = base.variant(**{param: v})
            try:
                res = B.run_backtest(adj, cl, vol, ff, cfg)
            except Exception as exc:
                rows.append({"param": param, "value": str(v), "error": str(exc)[:60]})
                continue
            row = {
                "param": param, "value": str(v),
                "CAGR_net": M.cagr(res.returns),
                "CAGR_gross": M.cagr(res.gross_returns),
                "Sharpe_net": M.sharpe(res.returns, ff["RF"]),
                "Sharpe_gross": M.sharpe(res.gross_returns, ff["RF"]),
                "MaxDD": M.max_drawdown(res.returns),
                "ann_turnover": res.turnover.mean() * len(res.turnover) / res.n_years,
            }
            if benchmark is not None:
                b = benchmark.reindex(res.returns.index).fillna(0.0)
                row["IR_vs_bmk"] = M.information_ratio(res.returns, b)
            rows.append(row)
            if verbose:
                print(f"  {param}={v}: net Sharpe {row['Sharpe_net']:.3f}, "
                      f"turnover {row['ann_turnover']:.1f}x", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime / subperiod analysis
# ---------------------------------------------------------------------------
REGIMES = {
    "Dot-com bust 1999-2002": ("1999-01-01", "2002-12-31"),
    "Recovery 2003-2007": ("2003-01-01", "2007-12-31"),
    "GFC 2008-2009": ("2008-01-01", "2009-12-31"),
    "QE bull 2010-2019": ("2010-01-01", "2019-12-31"),
    "COVID 2020": ("2020-01-01", "2020-12-31"),
    "Inflation/bear 2022": ("2022-01-01", "2022-12-31"),
    "AI bull 2023-2026": ("2023-01-01", "2026-06-30"),
}


def subperiod_analysis(strategy: pd.Series, benchmark: pd.Series,
                       rf: pd.Series, regimes: dict | None = None) -> pd.DataFrame:
    rows = []
    for name, (a, b) in (regimes or REGIMES).items():
        s = strategy.loc[a:b].dropna()
        if len(s) < 60:
            continue
        bm = benchmark.reindex(s.index).fillna(0.0)
        rows.append({
            "regime": name,
            "n_days": len(s),
            "strat_CAGR": M.cagr(s),
            "bmk_CAGR": M.cagr(bm),
            "excess": M.cagr(s) - M.cagr(bm),
            "strat_Sharpe": M.sharpe(s, rf),
            "bmk_Sharpe": M.sharpe(bm, rf),
            "strat_MaxDD": M.max_drawdown(s),
            "bmk_MaxDD": M.max_drawdown(bm),
            "IR": M.information_ratio(s, bm),
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------
def cost_breakeven(adj, cl, vol, ff, base: BacktestConfig,
                   bps_grid=(0, 2, 5, 10, 15, 20, 30, 50),
                   benchmark: pd.Series | None = None) -> pd.DataFrame:
    """
    Re-run holding the signal fixed but varying total one-way trading cost.
    The break-even cost is where net excess return over the benchmark hits zero --
    the single most useful number for judging whether a high-turnover strategy is
    implementable.
    """
    rows = []
    for bps in bps_grid:
        cfg = base.variant(commission_bps=float(bps), spread_bps=0.0, slippage_bps=0.0)
        res = B.run_backtest(adj, cl, vol, ff, cfg)
        row = {
            "total_bps_1way": bps,
            "CAGR_net": M.cagr(res.returns),
            "Sharpe_net": M.sharpe(res.returns, ff["RF"]),
            "ann_turnover": res.turnover.mean() * len(res.turnover) / res.n_years,
        }
        if benchmark is not None:
            b = benchmark.reindex(res.returns.index).fillna(0.0)
            row["excess_CAGR"] = M.cagr(res.returns) - M.cagr(b)
            row["IR"] = M.information_ratio(res.returns, b)
        rows.append(row)
        print(f"  {bps} bps: net CAGR {row['CAGR_net']:.4f}", flush=True)
    return pd.DataFrame(rows)


def implied_breakeven_bps(df: pd.DataFrame, col: str = "excess_CAGR") -> float:
    """Linear interpolation of the cost level at which `col` crosses zero."""
    d = df.dropna(subset=[col]).sort_values("total_bps_1way")
    x, y = d["total_bps_1way"].to_numpy(float), d[col].to_numpy(float)
    sign = np.sign(y)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return np.nan if y[0] < 0 else np.inf
    i = idx[0]
    return float(x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    returns: pd.Series
    weights_history: pd.DataFrame
    folds: pd.DataFrame


def walk_forward(adj, cl, vol, ff, base: BacktestConfig,
                 train_years: int = 6, test_years: int = 2,
                 start: str = "1999-01-01", end: str = "2026-06-30",
                 benchmark: pd.Series | None = None) -> WalkForwardResult:
    r"""
    Anchored-length rolling walk-forward.

    For each fold: estimate each factor's IC information ratio on the TRAIN window,
    set w_k proportional to max(IC_IR_k, 0), then run the TEST window with those
    weights and keep only the test-window returns. Concatenating the test windows
    gives a return series in which every observation was generated by parameters
    fitted strictly on prior data.

    This is the strongest generalisation test available without new data: it
    reproduces the full research procedure (including factor selection) at every
    point in time, rather than freezing one set of weights chosen with hindsight.
    """
    fold_rows, seg_returns, wt_hist = [], [], {}
    t0 = pd.Timestamp(start)
    t_end = pd.Timestamp(end)

    while True:
        tr_a = t0
        tr_b = tr_a + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        te_a = tr_b + pd.Timedelta(days=1)
        te_b = te_a + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if te_a > t_end:
            break
        te_b = min(te_b, t_end)

        train_cfg = base.variant(start=str(tr_a.date()), end=str(tr_b.date()))
        ic = B.factor_ic(adj, cl, vol, ff, train_cfg)
        if ic.empty:
            break
        ir = ic["IC_IR"].drop(index="COMPOSITE", errors="ignore")
        floored = ir.clip(lower=0.0)
        if floored.sum() <= 0:
            weights = {k: 1.0 / len(F.FACTOR_NAMES) for k in F.FACTOR_NAMES}
            degenerate = True
        else:
            weights = (floored / floored.sum()).to_dict()
            degenerate = False
        weights = {k: float(weights.get(k, 0.0)) for k in F.FACTOR_NAMES}
        wt_hist[te_a] = weights

        test_cfg = base.variant(start=str(te_a.date()), end=str(te_b.date()),
                                factor_weights=weights)
        res = B.run_backtest(adj, cl, vol, ff, test_cfg)
        seg_returns.append(res.returns)

        row = {
            "train": f"{tr_a.date()}..{tr_b.date()}",
            "test": f"{te_a.date()}..{te_b.date()}",
            "test_CAGR": M.cagr(res.returns),
            "test_Sharpe": M.sharpe(res.returns, ff["RF"]),
            "test_MaxDD": M.max_drawdown(res.returns),
            "all_IC_IR_negative": degenerate,
            **{f"w_{k}": v for k, v in weights.items()},
        }
        if benchmark is not None:
            b = benchmark.reindex(res.returns.index).fillna(0.0)
            row["bmk_CAGR"] = M.cagr(b)
            row["excess"] = row["test_CAGR"] - row["bmk_CAGR"]
        fold_rows.append(row)
        print(f"  fold {row['test']}: Sharpe {row['test_Sharpe']:.3f}", flush=True)

        t0 = t0 + pd.DateOffset(years=test_years)

    stitched = pd.concat(seg_returns).sort_index() if seg_returns else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated()]
    return WalkForwardResult(
        returns=stitched,
        weights_history=pd.DataFrame(wt_hist).T.sort_index(),
        folds=pd.DataFrame(fold_rows),
    )


# ---------------------------------------------------------------------------
# Bootstrap and multiple-testing
# ---------------------------------------------------------------------------
def stationary_bootstrap_sharpe(r: pd.Series, n_boot: int = 2000,
                                mean_block: int = 21, seed: int = 42) -> dict:
    """
    Politis-Romano stationary bootstrap: resample geometric-length blocks so that
    serial dependence in returns is preserved. Returns the sampling distribution of
    the annualised Sharpe ratio.
    """
    rng = np.random.default_rng(seed)
    x = r.dropna().to_numpy(float)
    T = len(x)
    if T < 100:
        return {}
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for t in range(T):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(T)
            else:
                i = (i + 1) % T
        s = x[idx]
        sd = s.std(ddof=1)
        out[b] = np.sqrt(TRADING_DAYS) * s.mean() / sd if sd > 0 else np.nan
    out = out[np.isfinite(out)]
    actual = M.sharpe(r)
    return {
        "sharpe": actual,
        "boot_mean": float(out.mean()),
        "ci_2.5%": float(np.percentile(out, 2.5)),
        "ci_97.5%": float(np.percentile(out, 97.5)),
        "p_sharpe_le_0": float((out <= 0).mean()),
        "n_boot": len(out),
    }


# ---------------------------------------------------------------------------
# Controls: is the *factor selection* doing anything at all?
# ---------------------------------------------------------------------------
def _chain(period_rets: list[float], ppy: int) -> pd.Series:
    return pd.Series(period_rets, dtype=float)


def equal_weight_universe(panel, cost_bps: float = 8.0,
                          ppy: int = 12) -> pd.Series:
    """
    Benchmark: hold EVERY eligible name at equal weight, rebalanced on the same
    schedule. This is the right control for a 50-stock equal-ish portfolio, because
    it strips out the equal-weighting tilt (a small-cap-within-large-cap bet) and
    isolates what the factor *selection* contributes. Comparing only against
    cap-weighted SPY confounds the two.
    """
    rets, prev = [], set()
    for _, _, fwd in panel:
        names = set(fwd.index)
        turn = 1.0 if not prev else len(names ^ prev) / max(len(names), 1)
        rets.append(float(fwd.mean()) - turn * cost_bps / 1e4)
        prev = names
    return pd.Series(rets, dtype=float)


def strategy_period_returns(panel, cfg: BacktestConfig, cost_bps: float = 8.0) -> pd.Series:
    """Period-frequency returns of the actual strategy, computed from the same panel
    as the null, so the comparison is exactly like-for-like."""
    rets, prev_w = [], pd.Series(dtype=float)
    for _, raw, fwd in panel:
        z = F.standardize_factors(raw, cfg)
        score = F.composite_score(z, cfg).dropna()
        common = score.index.intersection(fwd.index)
        if len(common) < cfg.n_long:
            continue
        sel = score.loc[common].nlargest(cfg.n_long).index
        w = pd.Series(1.0 / len(sel), index=sel)
        idx = w.index.union(prev_w.index)
        turn = float((w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())
        rets.append(float(fwd.loc[sel].mean()) - turn * cost_bps / 1e4)
        prev_w = w
    return pd.Series(rets, dtype=float)


def random_portfolio_null(panel, n_names: int = 50, n_sims: int = 500,
                          cost_bps: float = 8.0, seed: int = 42) -> pd.DataFrame:
    r"""
    Monte Carlo null: at each rebalance draw `n_names` at random from the SAME
    eligible universe, equal-weight them, and charge the same turnover cost.

    This is the sharpest available test of whether the factor model contributes
    anything. If the live strategy's performance sits inside the bulk of this
    distribution, the factors are not selecting stocks better than a coin flip --
    any apparent edge came from the universe and the weighting scheme, not the signal.

    Returns one row per simulation with its annualised return, volatility and Sharpe.
    """
    rng = np.random.default_rng(seed)
    ppy = 12
    rows = []
    universes = [list(fwd.index) for _, _, fwd in panel]
    fwds = [fwd for _, _, fwd in panel]

    for _ in range(n_sims):
        rets, prev = [], set()
        for uni, fwd in zip(universes, fwds):
            if len(uni) < n_names:
                continue
            pick = rng.choice(len(uni), size=n_names, replace=False)
            names = {uni[i] for i in pick}
            turn = 2.0 if not prev else len(names ^ prev) / n_names
            rets.append(float(fwd.iloc[pick].mean()) - turn * cost_bps / 1e4)
            prev = names
        s = pd.Series(rets, dtype=float)
        if len(s) < 12:
            continue
        mu = (1 + s).prod() ** (ppy / len(s)) - 1
        sd = s.std(ddof=1) * np.sqrt(ppy)
        rows.append({"CAGR": mu, "vol": sd, "Sharpe": (s.mean() / s.std(ddof=1) * np.sqrt(ppy))
                     if s.std(ddof=1) > 0 else np.nan})
    return pd.DataFrame(rows)


def null_percentile(actual: float, null: pd.Series) -> float:
    """Fraction of the null distribution the actual result exceeds."""
    n = null.dropna()
    return float((n < actual).mean()) if len(n) else np.nan


def collect_trial_sharpes(*frames: pd.DataFrame, col: str = "Sharpe_net") -> np.ndarray:
    """Gather every Sharpe ratio computed during research, for the Deflated Sharpe."""
    vals = []
    for f in frames:
        if isinstance(f, pd.DataFrame) and col in f.columns:
            vals.extend(f[col].dropna().tolist())
    return np.asarray(vals, dtype=float)
