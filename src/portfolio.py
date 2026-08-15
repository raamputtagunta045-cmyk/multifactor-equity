r"""
Universe screening and portfolio construction.

Eligibility (all evaluated with data available on or before the rebalance date):
    1. member of the S&P 500 on that date          (point-in-time membership)
    2. we have >= min_history_days of prices        (needed to score the stock)
    3. close >= min_price                           (avoid microstructure noise)
    4. 60-day median dollar volume >= min_dollar_volume  (tradability)

Weighting schemes
-----------------
equal        w_i = 1/N
score_tilt   w_i proportional to the stock's composite z-score shifted to be positive:
                 w_i = (S_i - min_j S_j + eps) / sum_k (S_k - min_j S_j + eps)
             This tilts toward higher-conviction names while keeping all weights
             non-negative and bounded.
inverse_vol  w_i proportional to 1/sigma_i, equalising each name's risk contribution
             under an assumption of equal correlations.

All schemes are then capped at `max_weight` and renormalised. Capping is applied
iteratively because renormalising after a cap can push another name above the cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig, TRADING_DAYS


def eligible_universe(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    members: pd.Series,
    cfg: BacktestConfig,
) -> pd.Index:
    """Tickers that pass every screen on `date`."""
    hist = prices.loc[:date]
    if len(hist) < cfg.min_history_days:
        return pd.Index([])

    in_index = members[members].index
    enough_history = hist.notna().sum() >= cfg.min_history_days
    has_price_today = hist.iloc[-1].notna()

    last_close = close.loc[:date].iloc[-1]
    price_ok = last_close >= cfg.min_price

    adv = dollar_volume.loc[:date].iloc[-60:].median()
    liquid = adv >= cfg.min_dollar_volume

    ok = (enough_history & has_price_today & price_ok.reindex(prices.columns, fill_value=False)
          & liquid.reindex(prices.columns, fill_value=False))
    uni = prices.columns[ok.values].intersection(in_index)

    if cfg.max_names is not None and len(uni) > cfg.max_names:
        uni = pd.Index(adv.reindex(uni).nlargest(cfg.max_names).index)
    return uni


def _apply_cap(w: pd.Series, cap: float, tol: float = 1e-10, max_iter: int = 100) -> pd.Series:
    """Iteratively cap weights at `cap` and redistribute the excess pro rata."""
    if cap is None or cap <= 0 or len(w) == 0:
        return w
    if cap * len(w) < 1 - tol:
        # Cap is infeasible (too few names); fall back to equal weight.
        return pd.Series(1.0 / len(w), index=w.index)
    w = w.clip(lower=0.0)
    for _ in range(max_iter):
        total = w.sum()
        if total <= 0:
            return pd.Series(1.0 / len(w), index=w.index)
        w = w / total
        over = w > cap + tol
        if not over.any():
            return w
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over
        if w[free].sum() <= tol:
            w[free] = excess / max(free.sum(), 1)
        else:
            w[free] = w[free] + excess * w[free] / w[free].sum()
    return w / w.sum()


def build_weights(
    scores: pd.Series,
    vols: pd.Series,
    cfg: BacktestConfig,
    prev_weights: pd.Series | None = None,
) -> pd.Series:
    """
    Turn a cross-section of composite scores into target portfolio weights.

    Long book = top `n_long` scores. If `n_short` > 0 the portfolio is
    dollar-neutral long/short: +1 gross on the long side, -1 on the short side.
    """
    s = scores.dropna().sort_values(ascending=False)
    if len(s) < 2:
        return pd.Series(dtype=float)

    n_long = min(cfg.n_long, len(s))
    longs = s.index[:n_long]

    # Optional rank buffer: a currently-held name is retained as long as it stays
    # inside the top n_long*(1+buffer) ranks. This cuts turnover caused by names
    # oscillating around the selection boundary.
    if cfg.turnover_buffer > 0 and prev_weights is not None and len(prev_weights):
        held = prev_weights[prev_weights > 0].index
        wide = int(round(n_long * (1 + cfg.turnover_buffer)))
        keep_zone = set(s.index[:wide])
        retained = [t for t in held if t in keep_zone]
        fresh = [t for t in s.index if t not in retained]
        longs = pd.Index((retained + fresh)[:n_long])

    def _raw(idx: pd.Index) -> pd.Series:
        if cfg.weighting == "equal":
            return pd.Series(1.0, index=idx)
        if cfg.weighting == "inverse_vol":
            v = vols.reindex(idx).abs()
            v = v.replace(0, np.nan).fillna(v.median())
            return 1.0 / v
        if cfg.weighting == "score_tilt":
            sc = scores.reindex(idx)
            return sc - sc.min() + 0.25   # shift to strictly positive
        raise ValueError(f"unknown weighting scheme {cfg.weighting!r}")

    w_long = _apply_cap(_raw(longs), cfg.max_weight)

    if cfg.n_short <= 0:
        return w_long

    n_short = min(cfg.n_short, len(s) - n_long)
    if n_short <= 0:
        return w_long
    shorts = s.index[-n_short:]
    w_short = _apply_cap(_raw(shorts).rank(ascending=False), cfg.max_weight)
    return pd.concat([w_long, -w_short])


def realised_vol(returns: pd.DataFrame, date: pd.Timestamp, lookback: int) -> pd.Series:
    win = returns.loc[:date].iloc[-lookback:]
    return win.std(ddof=1) * np.sqrt(TRADING_DAYS)
