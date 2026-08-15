r"""
Performance and risk statistics.

Formulas (r_t = daily net return, N = 252)

    CAGR          = (prod(1+r_t))^(N/T) - 1
    Volatility    = sqrt(N) * stdev(r_t)
    Sharpe        = sqrt(N) * mean(r_t - rf_t) / stdev(r_t - rf_t)
    Sortino       = sqrt(N) * mean(r_t - rf_t) / stdev(min(r_t - MAR, 0))
    Drawdown_t    = E_t / max_{s<=t} E_s - 1,   E_t = cumulative equity
    MaxDD         = min_t Drawdown_t
    Calmar        = CAGR / |MaxDD|
    Win rate      = #{r_t > 0} / T   (also reported monthly)
    Information ratio = sqrt(N) * mean(r_t - b_t) / stdev(r_t - b_t)

Statistical significance
------------------------
Daily strategy returns are autocorrelated and heteroskedastic, so a plain t-test
overstates significance. `newey_west_tstat` uses a HAC covariance estimator with
Bartlett kernel and lag L = floor(4 (T/100)^(2/9)) (Newey-West 1994 rule of thumb).

Multiple-testing correction
---------------------------
Any Sharpe ratio selected as the best of many trials is upward-biased. The
Probabilistic Sharpe Ratio and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
adjust for non-normality and for the number of independent trials:

    PSR(SR*) = Phi[ (SR - SR*) sqrt(T-1) / sqrt(1 - g1 SR + (g2-1)/4 SR^2) ]

where g1 is skewness and g2 kurtosis of returns. The DSR sets the benchmark SR*
to the expected maximum Sharpe obtainable from N independent random trials:

    SR* = sqrt(Var(SR_trials)) * [ (1-gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)) ]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import TRADING_DAYS

EULER = np.euler_gamma


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
def cagr(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    if len(r) == 0:
        return np.nan
    total = float((1 + r).prod())
    if total <= 0:
        return -1.0
    return total ** (periods / len(r)) - 1.0


def ann_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = r - rf if isinstance(rf, pd.Series) else r - rf / periods
    sd = ex.std(ddof=1)
    return float(np.sqrt(periods) * ex.mean() / sd) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, mar: float = 0.0,
            periods: int = TRADING_DAYS) -> float:
    ex = r - rf if isinstance(rf, pd.Series) else r - rf / periods
    downside = np.minimum(ex - mar, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    return float(np.sqrt(periods) * ex.mean() / dd) if dd > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def drawdown_details(r: pd.Series) -> dict:
    dd = drawdown_series(r)
    trough = dd.idxmin()
    eq = (1 + r).cumprod()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    recovery = rec.index[0] if len(rec) else None
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "drawdown_days": (trough - peak).days,
        "recovery_days": (recovery - trough).days if recovery is not None else np.nan,
    }


def calmar(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(r))
    return float(cagr(r, periods) / mdd) if mdd > 0 else np.nan


def win_rate(r: pd.Series) -> float:
    nz = r[r != 0]
    return float((nz > 0).mean()) if len(nz) else np.nan


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    """HAC t-statistic for H0: mean(x) = 0."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = x - x.mean()
    gamma0 = (e @ e) / T
    var = gamma0
    for l in range(1, lags + 1):
        cov = (e[l:] @ e[:-l]) / T
        var += 2 * (1 - l / (lags + 1)) * cov
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / T))


# ---------------------------------------------------------------------------
# Sharpe-ratio significance under non-normality and multiple testing
# ---------------------------------------------------------------------------
def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0,
                         periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    T = len(r)
    if T < 30:
        return np.nan
    sr = sharpe(r, 0.0, periods) / np.sqrt(periods)     # per-period SR
    srb = sr_benchmark / np.sqrt(periods)
    g1 = float(stats.skew(r))
    g2 = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1 - g1 * sr + (g2 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - srb) * np.sqrt(T - 1) / denom))


def deflated_sharpe(r: pd.Series, sr_trials: np.ndarray | list,
                    periods: int = TRADING_DAYS) -> tuple[float, float]:
    """
    Returns (DSR, expected_max_SR). `sr_trials` are the annualised Sharpe ratios
    of every configuration examined during research -- the honest denominator for
    multiple testing.
    """
    tr = np.asarray([s for s in np.asarray(sr_trials, dtype=float) if np.isfinite(s)])
    N = len(tr)
    if N < 2:
        return np.nan, np.nan
    var_sr = tr.var(ddof=1)
    if var_sr <= 0:
        return np.nan, np.nan
    z1 = stats.norm.ppf(1 - 1 / N)
    z2 = stats.norm.ppf(1 - 1 / (N * np.e))
    sr_star = np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2)
    return probabilistic_sharpe(r, sr_star, periods), float(sr_star)


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------
def information_ratio(r: pd.Series, b: pd.Series, periods: int = TRADING_DAYS) -> float:
    d = (r - b).dropna()
    sd = d.std(ddof=1)
    return float(np.sqrt(periods) * d.mean() / sd) if sd > 0 else np.nan


def capture_ratios(r: pd.Series, b: pd.Series) -> tuple[float, float]:
    up, dn = b > 0, b < 0
    u = float(r[up].mean() / b[up].mean()) if up.any() and b[up].mean() != 0 else np.nan
    d = float(r[dn].mean() / b[dn].mean()) if dn.any() and b[dn].mean() != 0 else np.nan
    return u, d


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summarize(
    r: pd.Series,
    benchmark: pd.Series | None = None,
    rf: pd.Series | None = None,
    turnover: pd.Series | None = None,
    periods: int = TRADING_DAYS,
    label: str = "strategy",
) -> pd.Series:
    r = r.dropna()
    rf_a = rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0
    dd = drawdown_details(r)
    monthly = (1 + r).resample("ME").prod() - 1

    out = {
        "CAGR": cagr(r, periods),
        "Ann. volatility": ann_vol(r, periods),
        "Sharpe": sharpe(r, rf_a, periods),
        "Sortino": sortino(r, rf_a, 0.0, periods),
        "Max drawdown": dd["max_drawdown"],
        "Calmar": calmar(r, periods),
        "Win rate (daily)": win_rate(r),
        "Win rate (monthly)": win_rate(monthly),
        "Skew": float(stats.skew(r)),
        "Excess kurtosis": float(stats.kurtosis(r)),
        "VaR 95% (daily)": float(np.percentile(r, 5)),
        "CVaR 95% (daily)": float(r[r <= np.percentile(r, 5)].mean()),
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "Drawdown length (days)": dd["drawdown_days"],
        "Recovery length (days)": dd["recovery_days"],
        "t-stat (Newey-West)": newey_west_tstat(r),
        "PSR (vs SR=0)": probabilistic_sharpe(r, 0.0, periods),
        "Years": len(r) / periods,
    }
    if turnover is not None and len(turnover):
        per_year = len(turnover) / (len(r) / periods)
        out["Turnover (ann., 1-way)"] = float(turnover.mean() * per_year)
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0.0)
        up, dn = capture_ratios(r, b)
        out.update({
            "Excess CAGR vs bmk": cagr(r, periods) - cagr(b, periods),
            "Information ratio": information_ratio(r, b, periods),
            "Tracking error": ann_vol(r - b, periods),
            "Up capture": up,
            "Down capture": dn,
        })
    return pd.Series(out, name=label)


def annual_returns(r: pd.Series) -> pd.Series:
    return (1 + r).resample("YE").prod() - 1


def monthly_return_table(r: pd.Series) -> pd.DataFrame:
    m = (1 + r).resample("ME").prod() - 1
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    return df.pivot(index="year", columns="month", values="ret")


def rolling_sharpe(r: pd.Series, window: int = TRADING_DAYS,
                   periods: int = TRADING_DAYS) -> pd.Series:
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    return np.sqrt(periods) * mu / sd
