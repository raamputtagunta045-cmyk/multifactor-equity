"""
Step 3: headline backtests.

Runs both frozen specifications over the in-sample, out-of-sample and full windows,
against a buy-and-hold SPY benchmark, and saves returns + summary tables.
"""
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src import attribution as A, backtest as B, config as C, data as D, metrics as M, specs as S

if __name__ == "__main__":
    pd.set_option("display.width", 240)
    px = D.download_prices([])
    ff = D.download_ff_factors()
    bm = D.download_benchmark()
    adj, cl, vol = px["adj_close"], px["close"], px["volume"]
    bench_ret = bm[C.BENCHMARK_TICKER].pct_change(fill_method=None)

    windows = {
        "IS 1999-2012": (S.IS_START, S.IS_END),
        "OOS 2013-2026": (S.OOS_START, S.OOS_END),
        "FULL 1999-2026": (S.FULL_START, S.FULL_END),
    }

    all_returns: dict[str, pd.Series] = {}
    summaries: list[pd.Series] = []
    store: dict[tuple[str, str], B.BacktestResult] = {}

    for spec_name, spec in S.SPECS.items():
        for win_name, (a, b) in windows.items():
            cfg = spec.variant(start=a, end=b)
            print(f"running {spec_name:26s} {win_name} ...", flush=True)
            res = B.run_backtest(adj, cl, vol, ff, cfg)
            key = f"{spec_name} | {win_name}"
            store[(spec_name, win_name)] = res
            all_returns[key] = res.returns

            bslice = bench_ret.reindex(res.returns.index).fillna(0.0)
            s_net = M.summarize(res.returns, benchmark=bslice, rf=ff["RF"],
                                turnover=res.turnover, label=key + " | NET")
            s_gross = M.summarize(res.gross_returns, benchmark=bslice, rf=ff["RF"],
                                  turnover=res.turnover, label=key + " | GROSS")
            summaries += [s_gross, s_net]

    # benchmark on each window
    for win_name, (a, b) in windows.items():
        bslice = bench_ret.loc[a:b].dropna()
        summaries.append(M.summarize(bslice, rf=ff["RF"], label=f"SPY | {win_name}"))
        all_returns[f"SPY | {win_name}"] = bslice

    table = pd.DataFrame(summaries).T
    table.to_csv(C.TABLES / "headline_performance.csv")
    pd.DataFrame(all_returns).to_parquet(C.INTERIM / "strategy_returns.parquet")

    print("\n=== HEADLINE PERFORMANCE ===")
    print(table.round(4).to_string())

    # ---- turnover / cost detail -------------------------------------------
    print("\n=== TURNOVER & COST ===")
    rows = []
    for (spec_name, win_name), res in store.items():
        yrs = res.n_years
        ann_turn = res.turnover.mean() * len(res.turnover) / yrs
        rows.append({
            "spec": spec_name, "window": win_name,
            "turnover_per_rebal": res.turnover.mean(),
            "ann_turnover_1way": ann_turn,
            "ann_cost_drag": M.cagr(res.gross_returns) - M.cagr(res.returns),
            "mean_universe": res.universe_size.mean(),
            "unfilled_per_rebal": res.unfilled.mean(),
        })
    tc = pd.DataFrame(rows)
    tc.to_csv(C.TABLES / "turnover_costs.csv", index=False)
    print(tc.round(4).to_string(index=False))

    # ---- factor attribution ------------------------------------------------
    print("\n=== RISK-FACTOR ATTRIBUTION (net returns) ===")
    for (spec_name, win_name), res in store.items():
        if win_name == "FULL 1999-2026":
            print(f"\n--- {spec_name} ---")
            at = A.attribution_table(res.returns, ff)
            print(at.round(4).to_string())
            at.to_csv(C.TABLES / f"attribution_{spec_name.split()[0]}.csv")

    # ---- realised factor exposures of the book -----------------------------
    print("\n=== AVERAGE FACTOR EXPOSURE OF THE LONG BOOK (z-score units) ===")
    for (spec_name, win_name), res in store.items():
        if win_name == "FULL 1999-2026" and len(res.exposures):
            print(f"{spec_name}:")
            print(res.exposures.mean().round(3).to_string())
            res.exposures.to_csv(C.TABLES / f"exposures_{spec_name.split()[0]}.csv")

    print(f"\nsaved -> {C.TABLES}")
