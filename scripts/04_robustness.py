"""
Step 4: robustness, generalisation and overfitting diagnostics.

Order matters here. The controls run FIRST: if a random portfolio drawn from the
same universe performs as well as the strategy, no amount of parameter-sweep
prettiness rescues it.
"""
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src import backtest as B, config as C, data as D, metrics as M
from src import robustness as R, specs as S

if __name__ == "__main__":
    pd.set_option("display.width", 240)
    px = D.download_prices([])
    ff = D.download_ff_factors()
    bm = D.download_benchmark()
    adj, cl, vol = px["adj_close"], px["close"], px["volume"]
    bench_ret = bm[C.BENCHMARK_TICKER].pct_change(fill_method=None)
    base = S.SPEC_EW.variant(start=S.FULL_START, end=S.FULL_END)

    # ================= 1. CONTROLS ==========================================
    print("=" * 70)
    print("1. CONTROLS: equal-weight universe and random-portfolio null")
    print("=" * 70)
    panel = B.collect_factor_panel(adj, cl, vol, ff, base)
    print(f"  {len(panel)} monthly cross-sections\n")

    ppy = 12
    ewu = R.equal_weight_universe(panel)
    strat_ew = R.strategy_period_returns(panel, S.SPEC_EW)
    strat_ic = R.strategy_period_returns(panel, S.SPEC_IC)

    def _ann(s):
        return {"CAGR": (1 + s).prod() ** (ppy / len(s)) - 1,
                "vol": s.std(ddof=1) * np.sqrt(ppy),
                "Sharpe": s.mean() / s.std(ddof=1) * np.sqrt(ppy)}

    ctrl = pd.DataFrame({
        "Equal-weight ENTIRE universe": _ann(ewu),
        "Strategy EW (top 50)": _ann(strat_ew),
        "Strategy IC-wtd (top 50)": _ann(strat_ic),
    }).T
    print(ctrl.round(4).to_string())
    ctrl.to_csv(C.TABLES / "controls_equalweight.csv")

    print("\n  running random-portfolio Monte Carlo null (500 sims) ...")
    null = R.random_portfolio_null(panel, n_names=base.n_long, n_sims=500)
    null.to_csv(C.TABLES / "null_distribution.csv", index=False)
    print("\n  Null distribution of RANDOM 50-stock portfolios from the same universe:")
    print(null.describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())

    rows = []
    for name, s in [("Strategy EW", strat_ew), ("Strategy IC-wtd", strat_ic),
                    ("Equal-wt universe", ewu)]:
        a = _ann(s)
        rows.append({
            "portfolio": name,
            "CAGR": a["CAGR"], "Sharpe": a["Sharpe"],
            "pctile_vs_null_CAGR": R.null_percentile(a["CAGR"], null["CAGR"]),
            "pctile_vs_null_Sharpe": R.null_percentile(a["Sharpe"], null["Sharpe"]),
        })
    nullcmp = pd.DataFrame(rows)
    nullcmp.to_csv(C.TABLES / "null_comparison.csv", index=False)
    print("\n  Where the real strategies sit in the random-portfolio null:")
    print(nullcmp.round(4).to_string(index=False))

    # ================= 2. PARAMETER SWEEPS ==================================
    print("\n" + "=" * 70)
    print("2. PARAMETER SENSITIVITY (one knob at a time, full sample)")
    print("=" * 70)
    grid = {
        "n_long": [25, 50, 75, 100, 150],
        "rebalance": ["ME", "QE"],
        "weighting": ["equal", "score_tilt", "inverse_vol"],
        "mom_lookback": [126, 189, 252, 378],
        "rev_lookback": [10, 21, 42],
        "max_weight": [0.03, 0.05, 0.10],
        "turnover_buffer": [0.0, 0.25, 0.5, 1.0],
        "min_dollar_volume": [1e6, 5e6, 2e7],
    }
    sweep = R.param_sweep(adj, cl, vol, ff, base, grid, benchmark=bench_ret)
    sweep.to_csv(C.TABLES / "param_sweep.csv", index=False)
    print("\n", sweep.round(4).to_string(index=False))

    # ================= 3. REGIME ANALYSIS ===================================
    print("\n" + "=" * 70)
    print("3. PERFORMANCE BY MARKET REGIME")
    print("=" * 70)
    rets = pd.read_parquet(C.INTERIM / "strategy_returns.parquet")
    strat_full = rets["EW (a priori) | FULL 1999-2026"].dropna()
    sub = R.subperiod_analysis(strat_full, bench_ret, ff["RF"])
    sub.to_csv(C.TABLES / "regime_analysis.csv")
    print(sub.round(4).to_string())

    # ================= 4. COST BREAK-EVEN ===================================
    print("\n" + "=" * 70)
    print("4. TRANSACTION-COST BREAK-EVEN")
    print("=" * 70)
    cb = R.cost_breakeven(adj, cl, vol, ff, base, benchmark=bench_ret)
    cb.to_csv(C.TABLES / "cost_breakeven.csv", index=False)
    print(cb.round(4).to_string(index=False))
    be = R.implied_breakeven_bps(cb, "excess_CAGR")
    print(f"\n  break-even one-way cost vs SPY: {be:.2f} bps"
          if np.isfinite(be) else "\n  strategy does not beat SPY at ANY cost level")

    # ================= 5. WALK-FORWARD ======================================
    print("\n" + "=" * 70)
    print("5. WALK-FORWARD (6y train / 2y test, weights re-fitted each fold)")
    print("=" * 70)
    wf = R.walk_forward(adj, cl, vol, ff, base, train_years=6, test_years=2,
                        start=S.FULL_START, end=S.FULL_END, benchmark=bench_ret)
    wf.folds.to_csv(C.TABLES / "walk_forward_folds.csv", index=False)
    wf.weights_history.to_csv(C.TABLES / "walk_forward_weights.csv")
    wf.returns.to_frame("wf_returns").to_parquet(C.INTERIM / "walk_forward_returns.parquet")
    print("\n", wf.folds.round(4).to_string(index=False))

    if len(wf.returns):
        b = bench_ret.reindex(wf.returns.index).fillna(0.0)
        wf_sum = M.summarize(wf.returns, benchmark=b, rf=ff["RF"], label="walk-forward")
        wf_sum.to_frame().to_csv(C.TABLES / "walk_forward_summary.csv")
        print("\n  Stitched walk-forward performance:")
        print(wf_sum.round(4).to_string())

    # ================= 6. BOOTSTRAP + DEFLATED SHARPE =======================
    print("\n" + "=" * 70)
    print("6. BOOTSTRAP CONFIDENCE INTERVAL AND DEFLATED SHARPE")
    print("=" * 70)
    boot = R.stationary_bootstrap_sharpe(strat_full, n_boot=2000)
    print("  stationary block bootstrap (full-sample net returns):")
    for k, v in boot.items():
        print(f"    {k:16s} {v:.4f}" if isinstance(v, float) else f"    {k:16s} {v}")

    trials = R.collect_trial_sharpes(sweep)
    dsr, sr_star = M.deflated_sharpe(strat_full, trials)
    print(f"\n  trials examined (parameter sweep): {len(trials)}")
    print(f"  expected max Sharpe from noise alone (SR*): {sr_star:.4f}")
    print(f"  actual full-sample Sharpe:                  {M.sharpe(strat_full, ff['RF']):.4f}")
    print(f"  Deflated Sharpe Ratio (P[SR > SR*]):        {dsr:.4f}")

    pd.Series({"boot_" + k: v for k, v in boot.items()}
              | {"n_trials": len(trials), "SR_star": sr_star, "DSR": dsr}
              ).to_csv(C.TABLES / "significance.csv")

    print(f"\nsaved -> {C.TABLES}")
