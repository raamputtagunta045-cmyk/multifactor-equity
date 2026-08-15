r"""
Event-driven-ish vectorised backtest engine.

Timing convention (this is where most backtests leak information)
-----------------------------------------------------------------
    * Factor values on rebalance date t use prices up to and including the close of t.
    * Target weights are therefore known only *after* the close of t.
    * Trades execute at the close of t + `execution_lag` (default: 1 trading day).
    * The new weights earn returns from t + lag + 1 onward.
This guarantees no look-ahead: nothing observable only at t+1 is used to trade at t.

Between rebalances, positions drift with prices rather than being silently held at
constant weight:

    w_{i,d} = w_{i,d-1} (1 + r_{i,d}) / (1 + r_{p,d}),   r_{p,d} = sum_i w_{i,d-1} r_{i,d}

Costs
-----
One-way trading cost in basis points is commission + half-spread + slippage. On a
rebalance day the cost charged to the portfolio is

    c_d = (bps / 10000) * sum_i | w_target_i - w_drifted_i |

i.e. cost is proportional to one-way turnover. Liquidity is enforced *before*
costing: the weight change for name i on a single day cannot exceed

    dw_max_i = participation_cap * ADV_i / AUM

so a large fund simply cannot reach its target in illiquid names, and the
un-executed portion stays in the previous position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import factors as F
from . import portfolio as P
from .config import BacktestConfig, TRADING_DAYS
from .data import membership_matrix


PERIODS_PER_YEAR = {"ME": 12, "QE": 4, "W-FRI": 52, "2W-FRI": 26}


@dataclass
class BacktestResult:
    returns: pd.Series                 # daily net portfolio returns
    gross_returns: pd.Series           # before costs
    equity: pd.Series                  # cumulative net growth of $1
    weights: pd.DataFrame              # target weights at each rebalance
    turnover: pd.Series                # one-way turnover per rebalance
    costs: pd.Series                   # cost drag per rebalance day
    exposures: pd.DataFrame            # weighted-average factor z-score of the book
    universe_size: pd.Series           # eligible names per rebalance
    unfilled: pd.Series                # weight that liquidity prevented trading
    config: BacktestConfig

    @property
    def n_years(self) -> float:
        return len(self.returns) / TRADING_DAYS


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each period present in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.resample(freq).last().dropna().values)


def run_backtest(
    prices: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    ff: pd.DataFrame,
    cfg: BacktestConfig,
    sectors: pd.Series | None = None,
    verbose: bool = False,
) -> BacktestResult:
    prices = prices.sort_index()
    returns = prices.pct_change(fill_method=None)

    dollar_volume = (close * volume).rolling(5, min_periods=1).mean()

    mkt_excess = ff["Mkt-RF"].reindex(prices.index)
    rf = ff["RF"].reindex(prices.index).fillna(0.0)

    # Evaluation window, with warm-up available before it.
    sample = prices.loc[cfg.start: cfg.end].index
    if len(sample) == 0:
        raise ValueError("empty sample window")
    rebals = rebalance_dates(sample, cfg.rebalance)
    members_all = membership_matrix(rebals)

    all_days = sample
    day_pos = {d: i for i, d in enumerate(all_days)}

    w = pd.Series(dtype=float)              # current live weights
    pending: dict[int, pd.Series] = {}      # execution-day index -> target weights

    gross, net, dates = [], [], []
    tno, cst, unf, usz = {}, {}, {}, {}
    wt_hist, exp_hist = {}, {}

    for i, d in enumerate(all_days):
        # ---- 1. accrue today's return on yesterday's (drifted) weights -----
        if len(w):
            r_d = returns.loc[d].reindex(w.index)
            r_d = r_d.fillna(0.0)           # missing price -> treat as flat
            r_p = float((w * r_d).sum())
            w = w * (1.0 + r_d) / (1.0 + r_p) if (1.0 + r_p) != 0 else w
        else:
            r_p = 0.0

        cost_today = 0.0

        # ---- 2. execute any trade scheduled for today ----------------------
        if i in pending:
            target = pending.pop(i)
            idx = w.index.union(target.index)
            w_old = w.reindex(idx).fillna(0.0)
            w_new = target.reindex(idx).fillna(0.0)

            # liquidity throttle
            adv = dollar_volume.loc[:d].iloc[-60:].median().reindex(idx)
            dw_max = (cfg.participation_cap * adv / cfg.aum).fillna(0.0)
            desired = w_new - w_old
            executed = desired.clip(-dw_max, dw_max)
            shortfall = float((desired - executed).abs().sum())

            w = (w_old + executed)
            w = w[w.abs() > 1e-8]
            gross_lev = w.abs().sum()
            if gross_lev > 0:
                w = w / gross_lev            # keep gross exposure at 1

            turn = float(executed.abs().sum())
            cost_today = turn * cfg.total_cost_bps / 1e4
            tno[d] = turn
            cst[d] = cost_today
            unf[d] = shortfall

        gross.append(r_p)
        net.append(r_p - cost_today)
        dates.append(d)

        # ---- 3. generate a signal if today is a rebalance date -------------
        if d in members_all.index:
            uni = P.eligible_universe(d, prices, close, dollar_volume,
                                      members_all.loc[d], cfg)
            usz[d] = len(uni)
            if len(uni) >= max(10, cfg.n_long // 2):
                raw = F.compute_raw_factors(prices[uni], returns[uni], mkt_excess,
                                            rf, d, cfg)
                grp = sectors.reindex(uni) if (cfg.sector_neutral and sectors is not None) else None
                z = F.standardize_factors(raw, cfg, groups=grp)
                score = F.composite_score(z, cfg)
                vols = P.realised_vol(returns[uni], d, cfg.vol_lookback)
                target = P.build_weights(score, vols, cfg, prev_weights=w)
                if len(target):
                    wt_hist[d] = target
                    aligned = z.reindex(target.index)
                    exp_hist[d] = aligned.mul(target.abs(), axis=0).sum() / target.abs().sum()
                    exec_i = i + cfg.execution_lag
                    if exec_i < len(all_days):
                        pending[exec_i] = target
            if verbose and len(usz) % 24 == 0:
                print(f"    {d.date()}  universe={len(uni)}", flush=True)

    net_s = pd.Series(net, index=pd.DatetimeIndex(dates)).fillna(0.0)
    gross_s = pd.Series(gross, index=pd.DatetimeIndex(dates)).fillna(0.0)

    return BacktestResult(
        returns=net_s,
        gross_returns=gross_s,
        equity=(1 + net_s).cumprod(),
        weights=pd.DataFrame(wt_hist).T.sort_index(),
        turnover=pd.Series(tno).sort_index(),
        costs=pd.Series(cst).sort_index(),
        exposures=pd.DataFrame(exp_hist).T.sort_index(),
        universe_size=pd.Series(usz).sort_index(),
        unfilled=pd.Series(unf).sort_index(),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Single-factor decile spreads -- the standard first check on whether a signal
# has any cross-sectional information content at all.
# ---------------------------------------------------------------------------
def collect_factor_panel(
    prices: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    ff: pd.DataFrame,
    cfg: BacktestConfig,
) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.Series]]:
    """
    At every rebalance date, return (date, raw factor values, forward return to the
    next rebalance). The forward return is entered with `execution_lag` days of delay,
    matching how the live portfolio would actually trade.
    """
    returns = prices.pct_change(fill_method=None)
    dollar_volume = (close * volume).rolling(5, min_periods=1).mean()
    mkt_excess = ff["Mkt-RF"].reindex(prices.index)
    rf = ff["RF"].reindex(prices.index).fillna(0.0)

    sample = prices.loc[cfg.start: cfg.end].index
    rebals = rebalance_dates(sample, cfg.rebalance)
    members_all = membership_matrix(rebals)

    panel = []
    for k in range(len(rebals) - 1):
        d, d_next = rebals[k], rebals[k + 1]
        uni = P.eligible_universe(d, prices, close, dollar_volume, members_all.loc[d], cfg)
        if len(uni) < 25:
            continue
        raw = F.compute_raw_factors(prices[uni], returns[uni], mkt_excess, rf, d, cfg)
        fwd_start = prices.index[min(prices.index.get_loc(d) + cfg.execution_lag,
                                     len(prices.index) - 1)]
        fwd = prices.loc[d_next, uni] / prices.loc[fwd_start, uni] - 1.0
        panel.append((d, raw, fwd.dropna()))
    return panel


def factor_decile_test(prices, close, volume, ff, cfg, n_buckets: int = 5,
                       panel=None) -> pd.DataFrame:
    """
    Sort the eligible universe into `n_buckets` by each factor at every rebalance and
    measure the equal-weighted forward return of each bucket. Q1 = lowest factor value.
    Reports annualised bucket returns and the top-minus-bottom spread with a
    Newey-West t-statistic.
    """
    from .metrics import newey_west_tstat

    panel = panel if panel is not None else collect_factor_panel(prices, close, volume, ff, cfg)
    records = {f: {b: [] for b in range(n_buckets)} for f in F.FACTOR_NAMES}

    for _, raw, fwd in panel:
        for f in F.FACTOR_NAMES:
            s = raw[f].dropna()
            common = s.index.intersection(fwd.index)
            if len(common) < n_buckets * 5:
                continue
            s = s.loc[common]
            buckets = pd.qcut(s.rank(method="first"), n_buckets, labels=False)
            for b in range(n_buckets):
                records[f][b].append(fwd.loc[s.index[buckets == b]].mean())

    ppy = PERIODS_PER_YEAR.get(cfg.rebalance, 12)
    rows = []
    for f in F.FACTOR_NAMES:
        series = {b: pd.Series(v) for b, v in records[f].items() if len(v)}
        if len(series) < n_buckets:
            continue
        row = {"factor": f}
        for b in range(n_buckets):
            row[f"Q{b+1}"] = series[b].mean() * ppy
        spread = series[n_buckets - 1] - series[0]
        row["spread"] = spread.mean() * ppy
        row["t_stat"] = newey_west_tstat(spread)
        row["n_periods"] = len(spread)
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")


def factor_ic(prices, close, volume, ff, cfg, panel=None) -> pd.DataFrame:
    r"""
    Information coefficient analysis.

        IC_t = Spearman rank correlation( factor_{i,t} , forward return_{i,t->t+1} )

    Rank correlation is robust to the fat tails and outliers that dominate raw
    return cross-sections, so it is a more stable read on signal quality than a
    bucket spread. Reported:

        mean IC     average cross-sectional predictive power
        IC IR       mean(IC) / stdev(IC)  -- consistency of the signal
        t-stat      mean(IC) / stdev(IC) * sqrt(n_periods)
        hit rate    fraction of periods with IC > 0
    """
    from scipy import stats as sps

    panel = panel if panel is not None else collect_factor_panel(prices, close, volume, ff, cfg)
    ics: dict[str, list[float]] = {f: [] for f in F.FACTOR_NAMES}
    comp: list[float] = []

    for _, raw, fwd in panel:
        z = F.standardize_factors(raw, cfg)
        score = F.composite_score(z, cfg)
        for f in F.FACTOR_NAMES:
            s = raw[f].dropna()
            common = s.index.intersection(fwd.index)
            if len(common) >= 25:
                ics[f].append(sps.spearmanr(s.loc[common], fwd.loc[common]).statistic)
        sc = score.dropna()
        common = sc.index.intersection(fwd.index)
        if len(common) >= 25:
            comp.append(sps.spearmanr(sc.loc[common], fwd.loc[common]).statistic)
    ics["COMPOSITE"] = comp

    rows = []
    for f, vals in ics.items():
        v = pd.Series(vals).dropna()
        if len(v) < 10:
            continue
        sd = v.std(ddof=1)
        rows.append({
            "factor": f,
            "mean_IC": v.mean(),
            "std_IC": sd,
            "IC_IR": v.mean() / sd if sd > 0 else np.nan,
            "t_stat": v.mean() / sd * np.sqrt(len(v)) if sd > 0 else np.nan,
            "hit_rate": (v > 0).mean(),
            "n_periods": len(v),
        })
    return pd.DataFrame(rows).set_index("factor")


def factor_correlation(prices, close, volume, ff, cfg, panel=None) -> pd.DataFrame:
    """Average cross-sectional rank correlation between factor pairs -- shows how
    much genuine diversification the composite actually gets."""
    panel = panel if panel is not None else collect_factor_panel(prices, close, volume, ff, cfg)
    mats = []
    for _, raw, _ in panel:
        c = raw.rank().corr(method="pearson")
        if c.notna().all().all():
            mats.append(c.to_numpy())
    if not mats:
        return pd.DataFrame()
    return pd.DataFrame(np.mean(mats, axis=0), index=F.FACTOR_NAMES, columns=F.FACTOR_NAMES)
