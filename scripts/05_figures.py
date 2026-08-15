"""Step 5: generate every figure in results/figures/."""
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src import attribution as A, backtest as B, config as C, data as D
from src import metrics as M, plots as PL, specs as S

if __name__ == "__main__":
    px = D.download_prices([])
    ff = D.download_ff_factors()
    bm = D.download_benchmark()
    adj, cl, vol = px["adj_close"], px["close"], px["volume"]
    bench = bm[C.BENCHMARK_TICKER].pct_change(fill_method=None)

    rets = pd.read_parquet(C.INTERIM / "strategy_returns.parquet")
    ew_full = rets["EW (a priori) | FULL 1999-2026"].dropna()
    ic_full = rets["IC-weighted (IS-fitted) | FULL 1999-2026"].dropna()
    spy_full = rets["SPY | FULL 1999-2026"].dropna()

    print("generating figures ...")

    # 1-3 core performance
    PL.cumulative_returns({
        "Strategy (EW composite)": ew_full,
        "Strategy (IC-weighted)": ic_full,
        "SPY buy & hold": spy_full,
    })
    PL.drawdowns({"Strategy (EW composite)": ew_full, "SPY buy & hold": spy_full})
    PL.rolling_sharpe({"Strategy (EW composite)": ew_full, "SPY buy & hold": spy_full})

    # 4-5 portfolio internals
    cfg_full = S.SPEC_EW.variant(start=S.FULL_START, end=S.FULL_END)
    res = B.run_backtest(adj, cl, vol, ff, cfg_full)
    PL.factor_exposures(res.exposures)
    PL.turnover_chart(res.turnover)

    # 6-7 return distribution
    PL.annual_returns_bar(ew_full, spy_full)
    PL.monthly_heatmap(ew_full, title="Monthly returns, strategy (EW composite)")

    # 8 factor efficacy
    ic = pd.read_csv(C.TABLES / "is_factor_ic.csv", index_col=0)
    PL.factor_ic_bar(ic)

    # 9 survivorship
    audit = pd.read_csv(C.TABLES / "survivorship_audit.csv", index_col=0,
                        parse_dates=[0])
    PL.survivorship_coverage(audit)

    # 10 null distribution
    null = pd.read_csv(C.TABLES / "null_distribution.csv")
    nullcmp = pd.read_csv(C.TABLES / "null_comparison.csv").set_index("portfolio")
    PL.null_histogram(
        null["Sharpe"],
        {"Strategy EW": float(nullcmp.loc["Strategy EW", "Sharpe"]),
         "Strategy IC-wtd": float(nullcmp.loc["Strategy IC-wtd", "Sharpe"]),
         "Equal-wt whole universe": float(nullcmp.loc["Equal-wt universe", "Sharpe"])},
        metric="Sharpe",
        title="Does factor selection beat chance? Strategy vs 500 random 50-stock portfolios",
    )

    # 11 cost break-even
    cb = pd.read_csv(C.TABLES / "cost_breakeven.csv")
    PL.cost_breakeven_chart(cb)

    # 12 walk-forward
    folds = pd.read_csv(C.TABLES / "walk_forward_folds.csv")
    PL.walk_forward_chart(folds)

    # 13 rolling betas
    rb = A.rolling_alpha_beta(ew_full, ff)
    PL.rolling_beta_chart(rb)

    # 14 IS vs OOS
    perf = pd.read_csv(C.TABLES / "headline_performance.csv", index_col=0)
    row = perf.loc["Excess CAGR vs bmk"]
    data = pd.DataFrame({
        "EW (a priori)": [
            float(row["EW (a priori) | IS 1999-2012 | NET"]),
            float(row["EW (a priori) | OOS 2013-2026 | NET"]),
        ],
        "IC-weighted (IS-fitted)": [
            float(row["IC-weighted (IS-fitted) | IS 1999-2012 | NET"]),
            float(row["IC-weighted (IS-fitted) | OOS 2013-2026 | NET"]),
        ],
    }, index=["In-sample\n1999-2012", "Out-of-sample\n2013-2026"])
    PL.is_oos_bar(data)

    print(f"\nall figures -> {C.FIGURES}")
