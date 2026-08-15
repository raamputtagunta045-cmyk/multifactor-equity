"""
Central configuration for the multi-factor equity research project.

Every tunable lives here so that a backtest is fully specified by this file
plus a git commit hash. Robustness tests mutate copies of `BacktestConfig`,
never global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _p in (DATA, RAW, INTERIM, RESULTS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data sources (all free / public)
# ----------------------------------------------------------------------------
SP500_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF_MOM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

BENCHMARK_TICKER = "SPY"

# Download window: starts early enough to warm up a 252-day lookback before
# the first rebalance of the evaluation sample.
DOWNLOAD_START = "1996-01-01"
DOWNLOAD_END = "2026-06-30"

# ----------------------------------------------------------------------------
# Backtest configuration
# ----------------------------------------------------------------------------
TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestConfig:
    """A complete, reproducible specification of one backtest run."""

    # --- sample -------------------------------------------------------------
    start: str = "1999-01-01"
    end: str = "2026-06-30"

    # --- universe filters ---------------------------------------------------
    min_price: float = 5.0              # exclude sub-$5 stocks (microstructure noise)
    min_dollar_volume: float = 5e6      # 60d median ADV floor, USD
    min_history_days: int = 252         # need a full year of returns to score
    max_names: int | None = None        # optional cap on universe size

    # --- factor definitions -------------------------------------------------
    mom_lookback: int = 252             # 12 months
    mom_skip: int = 21                  # skip most recent month (reversal)
    vol_lookback: int = 252             # realised volatility window
    beta_lookback: int = 252            # market-model beta window
    rev_lookback: int = 21              # short-term reversal window
    resmom_lookback: int = 252          # residual accumulation window (12 months)
    resmom_skip: int = 21               # skip most recent month
    resmom_beta_window: int = 756       # market-model ESTIMATION window (3 years).
    # Must be materially longer than the accumulation window: OLS residuals sum to
    # zero over the estimation window, so if the two coincide the "residual
    # momentum" signal degenerates into residual short-term reversal.

    # Weights on each z-scored factor in the composite score.
    factor_weights: dict = field(
        default_factory=lambda: {
            "momentum": 0.30,
            "residual_momentum": 0.20,
            "low_volatility": 0.20,
            "low_beta": 0.15,
            "reversal": 0.15,
        }
    )

    # --- portfolio construction --------------------------------------------
    rebalance: str = "ME"               # pandas offset alias: ME, QE, W-FRI
    n_long: int = 50                    # names in the long book
    n_short: int = 0                    # 0 => long-only; >0 => long/short
    weighting: str = "score_tilt"       # equal | score_tilt | inverse_vol
    max_weight: float = 0.05            # position cap
    sector_neutral: bool = False        # requires sector map
    winsorize: float = 0.01             # two-sided winsorisation of factor values
    turnover_buffer: float = 0.0        # rank buffer to damp turnover (0 = off)

    # --- costs & frictions --------------------------------------------------
    commission_bps: float = 1.0         # per-side commission
    spread_bps: float = 5.0             # per-side half-spread cost
    slippage_bps: float = 2.0           # per-side market-impact allowance
    participation_cap: float = 0.05     # max fraction of ADV tradable per day
    aum: float = 100e6                  # assumed fund size (drives liquidity limits)

    # --- misc ---------------------------------------------------------------
    execution_lag: int = 1              # trade on the close AFTER the signal date
    seed: int = 42

    @property
    def total_cost_bps(self) -> float:
        """Round-trip-agnostic per-side cost in basis points."""
        return self.commission_bps + self.spread_bps + self.slippage_bps

    def variant(self, **kwargs) -> "BacktestConfig":
        """Return a copy with overrides — used by the robustness sweeps."""
        return replace(self, **kwargs)


DEFAULT = BacktestConfig()
