"""
Step 6: demonstrate that the value/quality pipeline is complete and correct, and
that it is wired to REFUSE to produce reportable results from synthetic data.

This script produces NO performance numbers. Its only job is to show that the
moment a real point-in-time fundamentals feed is dropped in, the value, quality,
size and investment factors flow through the existing compositing and portfolio
machinery with no further code changes.
"""
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src import config as C, data as D, fundamentals as FU

if __name__ == "__main__":
    pd.set_option("display.width", 200)

    print("=" * 72)
    print("REQUIRED POINT-IN-TIME FUNDAMENTALS SCHEMA")
    print("=" * 72)
    for k, v in FU.REQUIRED_SCHEMA.items():
        print(f"  {k:18s} {v}")

    px = D.download_prices([])
    adj = px["adj_close"]
    tickers = list(adj.columns[:120])

    print("\n" + "=" * 72)
    print("PIPELINE EXERCISE ON SYNTHETIC DATA (NOT A RESULT)")
    print("=" * 72)
    synth = FU.make_synthetic_fundamentals(tickers, "1999-01-01", "2026-06-30")
    print(f"  generated {len(synth):,} synthetic filings for {len(tickers)} tickers")

    date = pd.Timestamp("2020-06-30")
    snap = FU.as_of(synth, date)
    print(f"\n  point-in-time snapshot at {date.date()}: {len(snap)} tickers visible")
    print(f"  latest filing_date in snapshot: {snap['filing_date'].max().date()} "
          f"(must be <= {date.date()})")
    assert snap["filing_date"].max() <= date, "PIT violation"
    print("  PIT constraint holds.")

    price_on_date = adj.loc[:date].iloc[-1]
    ff = FU.compute_fundamental_factors(synth, price_on_date, date)
    print(f"\n  computed factors: {list(ff.columns)}")
    print(f"  non-null coverage:\n{ff.notna().sum().to_string()}")
    print("\n  cross-sectional summary (synthetic -- meaningless by construction):")
    print(ff.describe().round(3).to_string())

    print("\n" + "=" * 72)
    print("GUARD CHECK")
    print("=" * 72)
    try:
        FU.assert_not_synthetic(synth, "06_fundamentals_demo")
        print("  ERROR: guard failed to fire!")
        sys.exit(1)
    except FU.SyntheticDataError as exc:
        print(f"  guard fired as designed:\n    {exc}")

    print("\nTo activate value/quality on real data:")
    print("  1. obtain a PIT feed (Sharadar SF1 `datekey`, or Compustat PIT via WRDS)")
    print("  2. map it to REQUIRED_SCHEMA above")
    print("  3. df = FU.load_fundamentals('path.parquet')   # validates + rejects look-ahead")
    print("  4. add the returned columns to BacktestConfig.factor_weights")
