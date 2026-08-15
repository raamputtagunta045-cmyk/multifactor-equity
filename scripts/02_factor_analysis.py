"""
Step 2: single-factor efficacy, measured ON THE IN-SAMPLE WINDOW ONLY.

This is the design stage. Everything decided here -- which factors survive, how
they are weighted -- is fitted on 1999-2012 and then frozen. The 2013-2026 window
is not touched until step 3, so it remains a genuine out-of-sample test.
"""
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src import backtest as B, config as C, data as D

IS_START, IS_END = "1999-01-01", "2012-12-31"
OOS_START, OOS_END = "2013-01-01", "2026-06-30"

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    px = D.download_prices([])
    ff = D.download_ff_factors()
    adj, cl, vol = px["adj_close"], px["close"], px["volume"]

    cfg = C.DEFAULT.variant(start=IS_START, end=IS_END)
    print(f"IN-SAMPLE factor analysis {IS_START} -> {IS_END}")
    panel = B.collect_factor_panel(adj, cl, vol, ff, cfg)
    print(f"  {len(panel)} rebalance cross-sections\n")

    dec = B.factor_decile_test(adj, cl, vol, ff, cfg, panel=panel)
    ic = B.factor_ic(adj, cl, vol, ff, cfg, panel=panel)
    corr = B.factor_correlation(adj, cl, vol, ff, cfg, panel=panel)

    print("--- Quintile sort (annualised) ---")
    print(dec.round(4).to_string(), "\n")
    print("--- Information coefficients ---")
    print(ic.round(4).to_string(), "\n")
    print("--- Average cross-sectional factor rank correlation ---")
    print(corr.round(3).to_string(), "\n")

    dec.to_csv(C.TABLES / "is_factor_deciles.csv")
    ic.to_csv(C.TABLES / "is_factor_ic.csv")
    corr.to_csv(C.TABLES / "is_factor_correlation.csv")

    # --- survivorship audit -------------------------------------------------
    full = C.DEFAULT.variant(start=IS_START, end=OOS_END)
    rebals = B.rebalance_dates(adj.loc[full.start:full.end].index, full.rebalance)
    audit = D.audit_survivorship(adj, rebals)
    audit.to_csv(C.TABLES / "survivorship_audit.csv")
    print("--- Survivorship coverage (priced members / true index members) ---")
    print(audit["coverage"].resample("YE").mean().round(3).to_string())
    print(f"\n  overall mean coverage: {audit['coverage'].mean():.1%}")
    print(f"  worst year coverage:   {audit['coverage'].resample('YE').mean().min():.1%}")
