"""Step 1: acquire all raw data. Idempotent -- re-running uses the parquet cache."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import config as C

if __name__ == "__main__":
    t0 = time.time()

    print("[1/4] historical S&P 500 membership ...")
    mem = D.download_membership()
    tickers = D.all_historical_tickers()
    print(f"      {len(mem)} change dates, {mem['date'].min().date()} -> "
          f"{mem['date'].max().date()}")
    print(f"      {len(tickers)} distinct tickers ever in the index")

    print("[2/4] Fama-French factors ...")
    ff = D.download_ff_factors()
    print(f"      {ff.shape[0]} daily obs, {ff.index[0].date()} -> {ff.index[-1].date()}")
    print(f"      columns: {list(ff.columns)}")

    print("[3/4] benchmark (SPY) ...")
    bm = D.download_benchmark()
    print(f"      {bm.shape[0]} daily obs, {bm.index[0].date()} -> {bm.index[-1].date()}")

    print(f"[4/4] prices for {len(tickers)} tickers (this takes several minutes) ...")
    px = D.download_prices(tickers)
    for k, v in px.items():
        print(f"      {k}: {v.shape[0]} rows x {v.shape[1]} tickers")

    print(f"\ndone in {time.time()-t0:.0f}s -> {C.INTERIM}")
