r"""
Value, quality and size factors -- and an honest account of why they are NOT in the
headline backtest.

WHY THESE FACTORS ARE NOT BACKTESTED ON REAL DATA HERE
------------------------------------------------------
Computing a value or quality factor without look-ahead bias requires *point-in-time*
(PIT) fundamentals: the accounting numbers as they were known to the market on the
rebalance date. Two properties are essential:

  1. FILING DATE, not period end. A fiscal quarter ending 31 Dec is typically not
     filed until late February. Using it on 1 Jan grants ~8 weeks of hindsight,
     which is enough to manufacture large fake alpha.
  2. NO RESTATEMENTS. Vendors that serve "current" financials overwrite history
     when a company restates. Backtesting on restated numbers means trading on
     figures nobody could have seen.

The free source used for prices in this project (Yahoo Finance via yfinance) fails
both tests. Measured directly in this environment:

    * `Ticker.balance_sheet` returns 5 annual periods; `quarterly_balance_sheet`
      returns 7 quarters -- roughly 5 years of history against the 27-year price
      sample. A 27-year fundamental backtest is simply not constructible.
    * Columns are keyed by FISCAL PERIOD END with no filing-date field, so the
      publication lag is unknown and cannot be applied correctly.
    * Values reflect the latest restatement, not the original filing.

Rather than silently backtest a look-ahead-contaminated value factor and report an
inflated Sharpe ratio, this module does three things:

    (a) specifies exactly what data is required (`REQUIRED_SCHEMA`),
    (b) implements the factor mathematics against that schema so the code is ready
        the moment a real PIT feed is connected (`compute_fundamental_factors`),
    (c) provides a clearly-labelled SYNTHETIC generator so the pipeline is runnable
        end-to-end -- with a hard guard against mistaking its output for a result.

WHERE TO GET REAL PIT DATA
--------------------------
    Compustat Point-in-Time (WRDS)   gold standard; PIT snapshots from 1987
    CRSP/Compustat Merged (WRDS)     survivorship-free with delisting returns
    Sharadar Core US Equities (SF1)  ~$150/mo, dimension="ARQ"/"ART" is PIT,
                                     includes delisted tickers back to ~1998
    S&P Capital IQ / FactSet         institutional
    SEC EDGAR full-text + XBRL       free; filing dates are exact, but you must
                                     parse and normalise the filings yourself

Sharadar SF1 is the realistic choice for an individual: it carries `datekey`
(the filing date) and retains delisted securities, which fixes both the look-ahead
and the survivorship problem at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required schema
# ---------------------------------------------------------------------------
REQUIRED_SCHEMA: dict[str, str] = {
    "ticker":            "str    security identifier, consistent with the price panel",
    "period_end":        "date   fiscal period end",
    "filing_date":       "date   date the figures became PUBLIC -- the PIT key",
    "shares_diluted":    "float  diluted weighted-average shares outstanding",
    "book_equity":       "float  common shareholders' equity",
    "net_income":        "float  net income to common, trailing twelve months",
    "revenue":           "float  total revenue, TTM",
    "gross_profit":      "float  revenue - COGS, TTM",
    "total_assets":      "float  total assets",
    "total_debt":        "float  short-term + long-term debt",
    "cash":              "float  cash and equivalents",
    "operating_cf":      "float  cash flow from operations, TTM",
    "capex":             "float  capital expenditure, TTM",
    "ebit":              "float  earnings before interest and tax, TTM",
}

SYNTHETIC_FLAG = "__SYNTHETIC__"


class SyntheticDataError(RuntimeError):
    """Raised when synthetic fundamentals would be used to produce reported results."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_fundamentals(path, price_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    Load a vendor PIT fundamentals file and validate it against REQUIRED_SCHEMA.

    The file must be long-format, one row per (ticker, period_end). Any vendor works
    as long as it can supply `filing_date`.
    """
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = set(REQUIRED_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(
            f"fundamentals file is missing required columns: {sorted(missing)}\n"
            f"expected schema:\n" +
            "\n".join(f"  {k:18s} {v}" for k, v in REQUIRED_SCHEMA.items())
        )
    for c in ("period_end", "filing_date"):
        df[c] = pd.to_datetime(df[c])
    bad = (df["filing_date"] < df["period_end"]).sum()
    if bad:
        raise ValueError(f"{bad} rows have filing_date before period_end -- look-ahead risk")
    return df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


def as_of(fundamentals: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """
    The most recent filing available for each ticker STRICTLY ON OR BEFORE `date`.
    This single function is what makes the fundamental factors point-in-time.
    """
    vis = fundamentals[fundamentals["filing_date"] <= date]
    if vis.empty:
        return vis
    return vis.sort_values("filing_date").groupby("ticker").tail(1).set_index("ticker")


# ---------------------------------------------------------------------------
# Factor mathematics
# ---------------------------------------------------------------------------
FUNDAMENTAL_FACTOR_NAMES = ["value", "quality", "size", "investment"]


def compute_fundamental_factors(
    fundamentals: pd.DataFrame,
    price: pd.Series,
    date: pd.Timestamp,
) -> pd.DataFrame:
    r"""
    Value, quality, size and investment factors, all signed so higher = better.

    Let P = price per share, S = diluted shares, MC = P*S the market capitalisation,
    and EV = MC + total_debt - cash the enterprise value.

    Value (composite of four ratios, each z-scored then averaged):
        book-to-price       B/P   = book_equity / MC
        earnings yield      E/P   = net_income / MC
        free-cash-flow yield FCF/P = (operating_cf - capex) / MC
        EBIT yield          EBIT/EV = ebit / EV
      Using a composite rather than a single ratio matters: B/P is distorted by
      intangible-heavy balance sheets, E/P by one-off charges, FCF/P by working-capital
      swings. Averaging four noisy measures of the same construct raises the
      signal-to-noise ratio.

    Quality:
        return on equity        ROE = net_income / book_equity
        gross profitability     GP/A = gross_profit / total_assets   (Novy-Marx 2013)
        accruals                ACC = (net_income - operating_cf) / total_assets
        leverage                LEV = total_debt / total_assets
      Quality = z(ROE) + z(GP/A) - z(ACC) - z(LEV), i.e. profitable, cash-backed,
      unlevered firms score high. Accruals enter negatively because earnings not
      backed by cash flow tend to reverse (Sloan 1996).

    Size:
        SIZE = -log(MC)      small-cap tilt (Banz 1981)

    Investment:
        INV = -(total_assets_t / total_assets_{t-1} - 1)
      Conservative asset growth predicts higher returns (Fama-French CMA).
    """
    snap = as_of(fundamentals, date)
    if snap.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_FACTOR_NAMES, dtype=float)

    p = price.reindex(snap.index)
    mc = p * snap["shares_diluted"]
    ev = mc + snap["total_debt"] - snap["cash"]

    mc = mc.where(mc > 0)
    ev = ev.where(ev > 0)

    bp = snap["book_equity"] / mc
    ep = snap["net_income"] / mc
    fcfp = (snap["operating_cf"] - snap["capex"]) / mc
    ebit_ev = snap["ebit"] / ev

    def _z(s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        sd = s.std(ddof=1)
        return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * np.nan

    value = pd.concat([_z(bp), _z(ep), _z(fcfp), _z(ebit_ev)], axis=1).mean(axis=1)

    roe = snap["net_income"] / snap["book_equity"].where(snap["book_equity"] > 0)
    gpa = snap["gross_profit"] / snap["total_assets"]
    acc = (snap["net_income"] - snap["operating_cf"]) / snap["total_assets"]
    lev = snap["total_debt"] / snap["total_assets"]
    quality = (_z(roe) + _z(gpa) - _z(acc) - _z(lev)) / 4.0

    size = -np.log(mc)

    if "total_assets_prev" in snap.columns:
        inv = -(snap["total_assets"] / snap["total_assets_prev"] - 1.0)
    else:
        inv = pd.Series(np.nan, index=snap.index)

    return pd.DataFrame(
        {"value": value, "quality": quality, "size": size, "investment": inv}
    )


# ---------------------------------------------------------------------------
# Synthetic sample data -- for pipeline testing ONLY
# ---------------------------------------------------------------------------
def make_synthetic_fundamentals(
    tickers: list[str],
    start: str = "1999-01-01",
    end: str = "2026-06-30",
    seed: int = 42,
    filing_lag_days: int = 45,
) -> pd.DataFrame:
    """
    Generate schema-conformant SYNTHETIC fundamentals.

    ============================ WARNING ============================
    The output is random. It contains no information about real
    companies. Any backtest run on it measures nothing except that
    the code executes. Results derived from it MUST NOT be reported.
    =================================================================

    The generator produces persistent, cross-sectionally dispersed firm
    characteristics (so z-scores and rank sorts behave realistically) and applies a
    realistic filing lag, which is what the pipeline needs in order to be exercised.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.date_range(start, end, freq="QE")
    rows = []
    # persistent firm-level characteristics
    base = {t: {
        "size": rng.lognormal(22, 1.2),
        "margin": np.clip(rng.normal(0.12, 0.07), -0.05, 0.45),
        "roe": np.clip(rng.normal(0.13, 0.08), -0.2, 0.5),
        "lev": np.clip(rng.normal(0.28, 0.15), 0.0, 0.8),
    } for t in tickers}

    for t in tickers:
        b = base[t]
        assets = b["size"]
        for q in quarters:
            assets_prev = assets
            assets *= 1 + rng.normal(0.015, 0.04)
            rev = assets * np.clip(rng.normal(0.75, 0.15), 0.1, 2.0)
            gp = rev * np.clip(b["margin"] + rng.normal(0, 0.02), -0.1, 0.6)
            ni = assets * np.clip(b["roe"] * 0.35 + rng.normal(0, 0.01), -0.15, 0.3)
            ocf = ni * np.clip(rng.normal(1.25, 0.25), 0.2, 3.0)
            rows.append({
                "ticker": t,
                "period_end": q,
                "filing_date": q + pd.Timedelta(days=filing_lag_days),
                "shares_diluted": b["size"] / 1e3 * np.clip(rng.normal(1, 0.02), 0.5, 2),
                "book_equity": assets * np.clip(rng.normal(0.42, 0.1), 0.05, 0.9),
                "net_income": ni,
                "revenue": rev,
                "gross_profit": gp,
                "total_assets": assets,
                "total_assets_prev": assets_prev,
                "total_debt": assets * b["lev"],
                "cash": assets * np.clip(rng.normal(0.09, 0.05), 0.0, 0.5),
                "operating_cf": ocf,
                "capex": abs(ocf) * np.clip(rng.normal(0.35, 0.15), 0.0, 1.2),
                "ebit": ni * np.clip(rng.normal(1.35, 0.2), 0.3, 3.0),
                SYNTHETIC_FLAG: True,
            })
    return pd.DataFrame(rows)


def assert_not_synthetic(df: pd.DataFrame, context: str = "") -> None:
    """Guard: call before reporting any performance number built on fundamentals."""
    if SYNTHETIC_FLAG in df.columns:
        raise SyntheticDataError(
            f"Refusing to report results computed from SYNTHETIC fundamentals{
                ' in ' + context if context else ''}. "
            "Connect a real point-in-time feed (see module docstring) first."
        )
