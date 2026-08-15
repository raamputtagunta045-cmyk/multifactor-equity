r"""
Factor construction.

All factors here are computed from the price/volume panel alone, which makes them
strictly point-in-time: the value of factor f for stock i on rebalance date t uses
only data observable on or before t. Every factor is signed so that *higher is
better* (i.e. a higher score should predict a higher subsequent return under the
hypothesis), which lets the compositing step be a simple weighted sum.

Definitions
-----------
Let P_{i,t} be the split- and dividend-adjusted close and r_{i,t} = P_{i,t}/P_{i,t-1} - 1
the daily total return. Let r_{m,t} be the market return and r_{f,t} the risk-free rate.

1. Momentum (12-1)
       MOM_{i,t} = P_{i,t-s} / P_{i,t-L} - 1,      L = 252, s = 21
   The most recent month is skipped because 1-month returns exhibit reversal
   (Jegadeesh 1990); including it contaminates the momentum signal.

2. Residual momentum
   Estimate the market model over the window (t-L, t]:
       r_{i,u} - r_{f,u} = alpha_i + beta_i (r_{m,u} - r_{f,u}) + eps_{i,u}
   then
       RESMOM_{i,t} = ( sum_{u=t-L}^{t-s} eps_{i,u} ) / sigma(eps_i)
   Scaling by residual volatility makes the signal comparable across stocks and
   strips out the market-beta component of raw momentum, which is the main driver
   of momentum's crash risk (Blitz, Huij & Martens 2011).

3. Low volatility
       VOL_{i,t}    = sqrt(252) * stdev(r_{i,u}; u in (t-L, t])
       LOWVOL_{i,t} = -VOL_{i,t}
   Sign flipped: the low-volatility anomaly says low-vol stocks earn higher
   risk-adjusted returns (Ang, Hodrick, Xing & Zhang 2006).

4. Low beta
       beta_{i,t}   = Cov(r_i, r_m) / Var(r_m)  over (t-L, t]
       LOWBETA_{i,t} = -beta_{i,t}
   The betting-against-beta effect (Frazzini & Pedersen 2014).

5. Short-term reversal
       REV_{i,t} = -( P_{i,t} / P_{i,t-s} - 1 ),   s = 21
   Liquidity-provision premium; one-month losers tend to bounce.

Cross-sectional standardisation
-------------------------------
Raw factor values are not comparable across dates (volatility levels drift) or
across factors (different units). At each date, within the eligible universe:

    winsorise at the p / 1-p quantiles       -> caps outliers
    z_{i,t} = (f_{i,t} - mean_t(f)) / std_t(f)  -> mean 0, unit variance

The composite score is the weighted sum of z-scores, re-standardised:

    S_{i,t} = sum_k w_k * z^{(k)}_{i,t}

Weighting z-scores rather than ranks preserves information about *how* extreme a
stock is, at the cost of more sensitivity to outliers -- which winsorisation controls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig, TRADING_DAYS

FACTOR_NAMES = ["momentum", "residual_momentum", "low_volatility", "low_beta", "reversal"]


# ---------------------------------------------------------------------------
# Raw factor values at a single date
# ---------------------------------------------------------------------------
def _window(returns: pd.DataFrame, date: pd.Timestamp, length: int) -> np.ndarray:
    """Trailing `length` rows of returns strictly up to and including `date`."""
    sub = returns.loc[:date]
    return sub.iloc[-length:]


def _market_model(win: pd.DataFrame, mkt_excess: pd.Series, rf: pd.Series):
    """
    Vectorised OLS of each stock's excess return on the market excess return over
    the rows of `win`. NaNs are handled per-column so stocks with partial history
    still get an estimate from the observations they do have.

    Returns (alpha, beta, residuals, residual_sd, n_obs), all aligned to win.columns.
    """
    m = mkt_excess.reindex(win.index)
    rf_w = rf.reindex(win.index).fillna(0.0)
    m = m.fillna(0.0)

    R = win.sub(rf_w, axis=0).to_numpy(dtype=float)     # T x N excess returns
    x = m.to_numpy(dtype=float)                         # T market excess
    mask = (~np.isnan(R)).astype(float)
    n_obs = mask.sum(axis=0)
    Rz = np.nan_to_num(R, nan=0.0)
    n = np.maximum(n_obs, 1.0)

    xbar = (mask * x[:, None]).sum(axis=0) / n
    ybar = Rz.sum(axis=0) / n
    xc = (x[:, None] - xbar) * mask
    yc = (Rz - ybar) * mask
    varx = (xc * xc).sum(axis=0) / np.maximum(n - 1, 1)
    covxy = (xc * yc).sum(axis=0) / np.maximum(n - 1, 1)
    beta = np.where(varx > 0, covxy / varx, np.nan)
    alpha = ybar - beta * xbar

    resid = (Rz - (alpha + beta * x[:, None])) * mask
    resid_sd = np.sqrt((resid ** 2).sum(axis=0) / np.maximum(n - 2, 1))
    return alpha, beta, resid, resid_sd, n_obs


def compute_raw_factors(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    mkt_excess: pd.Series,
    rf: pd.Series,
    date: pd.Timestamp,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Raw (un-standardised) factor values for every ticker on one rebalance date.
    Returns a DataFrame indexed by ticker with one column per factor.
    """
    L = max(cfg.mom_lookback, cfg.vol_lookback, cfg.beta_lookback)
    win = _window(returns, date, L)
    if len(win) < cfg.min_history_days:
        return pd.DataFrame(columns=FACTOR_NAMES, dtype=float)

    px = prices.loc[:date]
    tickers = returns.columns

    # --- 1. momentum (12-1): price ratio, skipping the last month -----------
    def _px_lag(k: int) -> pd.Series:
        return px.iloc[-k] if len(px) >= k else pd.Series(np.nan, index=tickers)

    p_skip = _px_lag(cfg.mom_skip + 1)
    p_look = _px_lag(cfg.mom_lookback + 1)
    momentum = p_skip / p_look - 1.0

    # --- 5. short-term reversal --------------------------------------------
    p_now = px.iloc[-1]
    reversal = -(p_now / _px_lag(cfg.rev_lookback + 1) - 1.0)

    # --- 3. realised volatility --------------------------------------------
    vw = win.iloc[-cfg.vol_lookback:]
    low_volatility = -(vw.std(ddof=1) * np.sqrt(TRADING_DAYS))

    # --- 4. market-model beta -----------------------------------------------
    bw = _window(returns, date, cfg.beta_lookback)
    alpha_b, beta, _, _, obs_b = _market_model(bw, mkt_excess, rf)
    beta = np.where(obs_b < cfg.min_history_days, np.nan, beta)

    # --- 2. residual momentum ----------------------------------------------
    # Estimate the market model on a LONGER window than the accumulation window.
    # OLS forces residuals to sum to zero over the estimation window; if the two
    # windows coincided, the accumulated residual would be exactly minus the
    # skipped month's residual, turning this into a reversal signal.
    ew = _window(returns, date, cfg.resmom_beta_window)
    alpha_e, beta_e, resid_e, resid_sd_e, obs_e = _market_model(ew, mkt_excess, rf)

    # accumulate residuals over (t-resmom_lookback, t-resmom_skip]
    T = len(ew)
    keep = np.zeros(T, dtype=bool)
    lo = max(T - cfg.resmom_lookback, 0)
    hi = T - cfg.resmom_skip
    keep[lo:hi] = True
    cum_resid = (resid_e * keep[:, None]).sum(axis=0)
    residual_momentum = np.where(resid_sd_e > 0, cum_resid / resid_sd_e, np.nan)
    # require enough history for the estimation window to be meaningful
    residual_momentum = np.where(obs_e < cfg.resmom_beta_window * 0.8, np.nan,
                                 residual_momentum)

    out = pd.DataFrame(
        {
            "momentum": momentum.reindex(tickers),
            "residual_momentum": pd.Series(residual_momentum, index=tickers),
            "low_volatility": low_volatility.reindex(tickers),
            "low_beta": pd.Series(-beta, index=tickers),
            "reversal": reversal.reindex(tickers),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Cross-sectional standardisation and compositing
# ---------------------------------------------------------------------------
def winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0 or s.notna().sum() < 10:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def standardize_factors(raw: pd.DataFrame, cfg: BacktestConfig,
                        groups: pd.Series | None = None) -> pd.DataFrame:
    """
    Winsorise then z-score each factor cross-sectionally. If `groups` is supplied
    (e.g. GICS sector), standardisation is done within group, which makes the
    resulting scores sector-neutral by construction.
    """
    out = {}
    for col in raw.columns:
        s = raw[col]
        if groups is None:
            out[col] = zscore(winsorize(s, cfg.winsorize))
        else:
            g = groups.reindex(s.index)
            out[col] = s.groupby(g).transform(
                lambda x: zscore(winsorize(x, cfg.winsorize))
            )
    return pd.DataFrame(out, index=raw.index)


def composite_score(z: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Weighted sum of factor z-scores, re-standardised to unit cross-sectional variance."""
    w = pd.Series(cfg.factor_weights, dtype=float)
    w = w.reindex(z.columns).fillna(0.0)
    # Require a stock to have at least half the factors present, then renormalise
    # the weights over the factors it does have -- avoids dropping names for one
    # missing input while keeping the score on a comparable scale.
    present = z.notna()
    wsum = present.mul(w, axis=1).sum(axis=1)
    raw = z.fillna(0.0).mul(w, axis=1).sum(axis=1)
    score = np.where(wsum >= 0.5 * w.sum(), raw / wsum.replace(0, np.nan), np.nan)
    return zscore(pd.Series(score, index=z.index))
