"""
Data acquisition layer.

Three sources, all public:
  1. Point-in-time S&P 500 membership (fja05680/sp500, derived from index change
     announcements). Gives us, for any date, the set of tickers that were *actually*
     in the index on that date -- this is what removes look-ahead in universe choice.
  2. Split/dividend-adjusted daily prices (Yahoo Finance via yfinance).
  3. Fama-French 5 factors + momentum (Ken French data library) for attribution.

IMPORTANT DATA-QUALITY NOTE
---------------------------
Yahoo Finance does not serve price history for fully delisted securities. We therefore
*can* avoid look-ahead in universe membership, but we *cannot* fully avoid survivorship
bias, because the price panel is missing names that were in the index and later
disappeared. `audit_survivorship()` measures exactly how large that hole is at each
rebalance date so the bias is reported rather than hidden.
"""
from __future__ import annotations

import io
import time
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore", category=FutureWarning)

_UA = {"User-Agent": "Mozilla/5.0 (research; multifactor-equity)"}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


# ---------------------------------------------------------------------------
# 1. Point-in-time universe
# ---------------------------------------------------------------------------
def download_membership(force: bool = False) -> pd.DataFrame:
    """Fetch the historical S&P 500 constituent file. Columns: date, tickers."""
    path = C.RAW / "sp500_historical_components.csv"
    if force or not path.exists():
        txt = _get(C.SP500_MEMBERSHIP_URL).decode("utf-8")
        path.write_text(txt, encoding="utf-8")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def membership_matrix(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Boolean DataFrame (dates x tickers): True where the ticker was an index
    member on that date. Uses as-of (backward) matching so a stock is only
    ever a member from its addition date onward -- no look-ahead.
    """
    raw = download_membership()
    idx = raw["date"].values
    all_tickers = sorted({t.strip() for row in raw["tickers"] for t in row.split(",")})
    tick_pos = {t: i for i, t in enumerate(all_tickers)}

    mat = np.zeros((len(dates), len(all_tickers)), dtype=bool)
    # For each requested date, take the most recent membership snapshot <= date.
    pos = np.searchsorted(idx, dates.values, side="right") - 1
    cache: dict[int, np.ndarray] = {}
    for i, p in enumerate(pos):
        if p < 0:
            continue
        if p not in cache:
            row = np.zeros(len(all_tickers), dtype=bool)
            for t in raw["tickers"].iloc[p].split(","):
                j = tick_pos.get(t.strip())
                if j is not None:
                    row[j] = True
            cache[p] = row
        mat[i] = cache[p]
    return pd.DataFrame(mat, index=dates, columns=all_tickers)


def all_historical_tickers() -> list[str]:
    raw = download_membership()
    u = {t.strip() for row in raw["tickers"] for t in row.split(",")}
    return sorted(u)


# ---------------------------------------------------------------------------
# 2. Prices
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str],
    start: str = C.DOWNLOAD_START,
    end: str = C.DOWNLOAD_END,
    chunk: int = 60,
    pause: float = 1.0,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download adjusted prices and volume for `tickers`, caching to parquet.

    Returns a dict of three wide DataFrames (dates x tickers):
        adj_close : split- and dividend-adjusted close (total-return basis)
        close     : split-adjusted close (used for the $5 price screen)
        volume    : share volume (used with close for the ADV liquidity screen)
    """
    import yfinance as yf

    out_paths = {k: C.INTERIM / f"{k}.parquet" for k in ("adj_close", "close", "volume")}
    if not force and all(p.exists() for p in out_paths.values()):
        return {k: pd.read_parquet(p) for k, p in out_paths.items()}

    frames: dict[str, list[pd.DataFrame]] = {"adj_close": [], "close": [], "volume": []}
    failed: list[str] = []

    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        try:
            df = yf.download(
                batch, start=start, end=end, progress=False,
                auto_adjust=False, threads=True, group_by="column",
            )
        except Exception as exc:  # pragma: no cover - network flake
            print(f"  batch {i//chunk}: download error {exc}")
            failed.extend(batch)
            continue

        if df is None or len(df) == 0:
            failed.extend(batch)
            continue

        # yfinance returns a MultiIndex (field, ticker) for multi-ticker requests.
        for key, field in (("adj_close", "Adj Close"), ("close", "Close"), ("volume", "Volume")):
            if isinstance(df.columns, pd.MultiIndex):
                if field not in df.columns.get_level_values(0):
                    continue
                sub = df[field]
            else:
                sub = df[[field]].rename(columns={field: batch[0]})
            sub = sub.dropna(axis=1, how="all")
            if sub.shape[1]:
                frames[key].append(sub)

        got = set(frames["adj_close"][-1].columns) if frames["adj_close"] else set()
        failed.extend([t for t in batch if t not in got])
        print(f"  batch {i//chunk + 1}/{-(-len(tickers)//chunk)}: "
              f"{len(got)}/{len(batch)} tickers returned data", flush=True)
        time.sleep(pause)

    result = {}
    for key, parts in frames.items():
        if not parts:
            raise RuntimeError(f"no data downloaded for {key}")
        wide = pd.concat(parts, axis=1).sort_index()
        wide = wide.loc[:, ~wide.columns.duplicated()]
        wide.index = pd.to_datetime(wide.index).tz_localize(None)
        wide.to_parquet(out_paths[key])
        result[key] = wide

    pd.Series(sorted(set(failed))).to_csv(C.INTERIM / "failed_tickers.csv", index=False,
                                          header=["ticker"])
    print(f"\n  downloaded {result['adj_close'].shape[1]} tickers; "
          f"{len(set(failed))} unavailable")
    return result


def download_benchmark(force: bool = False) -> pd.DataFrame:
    """Benchmark total-return series (SPY adjusted close)."""
    import yfinance as yf

    path = C.INTERIM / "benchmark.parquet"
    if not force and path.exists():
        return pd.read_parquet(path)
    df = yf.download(C.BENCHMARK_TICKER, start=C.DOWNLOAD_START, end=C.DOWNLOAD_END,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    out = df[["Adj Close"]].rename(columns={"Adj Close": C.BENCHMARK_TICKER})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.to_parquet(path)
    return out


# ---------------------------------------------------------------------------
# 3. Fama-French factors
# ---------------------------------------------------------------------------
def _parse_ff_zip(url: str, path, skip_marker: str = ",") -> pd.DataFrame:
    """Ken French CSVs have a preamble and an annual block after the daily block."""
    if not path.exists():
        path.write_bytes(_get(url))
    z = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()

    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().startswith(",") and any(ch.isalpha() for ch in ln))
    rows = []
    header = [c.strip() for c in lines[start].split(",")]
    header[0] = "date"
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header) or not parts[0].isdigit() or len(parts[0]) != 8:
            continue  # blank line, annual block, or copyright footer
        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # percent -> decimal
    return df.set_index("date").sort_index()


def download_ff_factors(force: bool = False) -> pd.DataFrame:
    """Daily FF5 + momentum, as decimal returns. Columns: Mkt-RF SMB HML RMW CMA RF MOM."""
    path = C.INTERIM / "ff_factors.parquet"
    if not force and path.exists():
        return pd.read_parquet(path)

    ff5 = _parse_ff_zip(C.FF5_DAILY_URL, C.RAW / "ff5_daily.zip")
    mom = _parse_ff_zip(C.FF_MOM_DAILY_URL, C.RAW / "ff_mom_daily.zip")
    mom.columns = ["MOM"]
    out = ff5.join(mom, how="left")
    out.to_parquet(path)
    return out


# ---------------------------------------------------------------------------
# 4. Survivorship audit
# ---------------------------------------------------------------------------
def audit_survivorship(prices: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    At each rebalance date, compare the true index membership against the set of
    members we can actually price. The gap is unrecoverable survivorship bias.
    """
    mem = membership_matrix(rebal_dates)
    have = set(prices.columns)
    rows = []
    for d in rebal_dates:
        true_members = set(mem.columns[mem.loc[d].values])
        priced = {t for t in true_members if t in have
                  and prices.loc[:d, t].notna().sum() >= 252}
        rows.append({
            "date": d,
            "index_members": len(true_members),
            "priced_members": len(priced),
            "missing": len(true_members) - len(priced),
            "coverage": len(priced) / max(len(true_members), 1),
        })
    return pd.DataFrame(rows).set_index("date")
