"""
Frozen strategy specifications.

The weights below were derived from IN-SAMPLE (1999-2012) evidence only, using a
pre-specified mechanical rule, and are frozen before the out-of-sample window is
examined. They are recorded here as literals -- not recomputed at run time -- so
that the out-of-sample test cannot silently absorb information from the test period.

Derivation of SPEC_IC (rule fixed in advance)
---------------------------------------------
    w_k  proportional to  max(IC_IR_k, 0)

where IC_IR_k = mean(IC_k) / stdev(IC_k) measured over the 167 in-sample monthly
cross-sections (see results/tables/is_factor_ic.csv):

    factor              IC_IR (IS)     floored     weight
    momentum               0.0300       0.0300      0.115
    residual_momentum      0.0673       0.0673      0.258
    low_volatility        -0.0056       0.0000      0.000
    low_beta               0.0170       0.0170      0.065
    reversal               0.1464       0.1464      0.562
                                        ------      -----
                                        0.2607      1.000

Weighting by information ratio (rather than by mean IC) rewards signals that are
*consistent*, not merely large on average; a factor whose IC swings wildly around a
positive mean contributes little to a diversified composite.

SPEC_EW is the a priori, unfitted hypothesis: equal weight on all five factors.
It uses no in-sample information whatsoever and is therefore the cleanest test of
the original economic thesis. It is the primary specification; SPEC_IC is reported
alongside it to show what in-sample fitting buys (and what it costs out of sample).
"""
from __future__ import annotations

from .config import DEFAULT, BacktestConfig

IS_START, IS_END = "1999-01-01", "2012-12-31"
OOS_START, OOS_END = "2013-01-01", "2026-06-30"
FULL_START, FULL_END = "1999-01-01", "2026-06-30"

# In-sample IC information ratios, copied from results/tables/is_factor_ic.csv.
IS_IC_IR = {
    "momentum": 0.0300,
    "residual_momentum": 0.0673,
    "low_volatility": -0.0056,
    "low_beta": 0.0170,
    "reversal": 0.1464,
}

EW_WEIGHTS = {
    "momentum": 0.20,
    "residual_momentum": 0.20,
    "low_volatility": 0.20,
    "low_beta": 0.20,
    "reversal": 0.20,
}

_floored = {k: max(v, 0.0) for k, v in IS_IC_IR.items()}
_total = sum(_floored.values())
IC_WEIGHTS = {k: round(v / _total, 4) for k, v in _floored.items()}

SPEC_EW: BacktestConfig = DEFAULT.variant(factor_weights=EW_WEIGHTS)
SPEC_IC: BacktestConfig = DEFAULT.variant(factor_weights=IC_WEIGHTS)

SPECS = {"EW (a priori)": SPEC_EW, "IC-weighted (IS-fitted)": SPEC_IC}
