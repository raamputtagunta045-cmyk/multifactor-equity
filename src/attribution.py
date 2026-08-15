r"""
Risk-factor attribution.

The central question for any strategy is whether its return is *compensation for
known risk* or something the standard factor models cannot explain. We regress
daily excess strategy returns on nested factor models:

    CAPM      r_p - r_f = a + b_M (r_M - r_f) + e
    FF3       ... + b_S SMB + b_H HML
    Carhart   ... + b_U MOM
    FF5       ... + b_R RMW + b_C CMA
    FF5+MOM   all six

`a` (alpha) is the average return unexplained by the model. It is annualised as
252*a for reporting. Standard errors use Newey-West HAC with automatic lag length,
because daily strategy returns are serially correlated.

Interpretation guide
--------------------
    * Alpha shrinking toward zero as factors are added => the strategy is a
      repackaging of known premia, not new information.
    * A large positive b_U with the strategy built on momentum is expected and is
      not evidence of skill.
    * A significantly negative b_M or b_S tells you the strategy's benchmark-relative
      performance is driven by a structural tilt, which will reverse when that
      tilt is out of favour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import TRADING_DAYS

MODELS: dict[str, list[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart4": ["Mkt-RF", "SMB", "HML", "MOM"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+MOM": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def _nw_lags(T: int) -> int:
    return max(int(np.floor(4 * (T / 100) ** (2 / 9))), 1)


def regress(strategy: pd.Series, ff: pd.DataFrame, factors: list[str]) -> dict:
    """Single HAC regression of excess strategy returns on `factors`."""
    df = pd.concat([strategy.rename("r"), ff[factors + ["RF"]]], axis=1).dropna()
    if len(df) < 60:
        return {}
    y = df["r"] - df["RF"]
    X = sm.add_constant(df[factors])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(df))})

    out = {
        "alpha_ann": float(fit.params["const"] * TRADING_DAYS),
        "alpha_t": float(fit.tvalues["const"]),
        "alpha_p": float(fit.pvalues["const"]),
        "R2": float(fit.rsquared),
        "R2_adj": float(fit.rsquared_adj),
        "n_obs": int(len(df)),
    }
    for f in factors:
        out[f"beta_{f}"] = float(fit.params[f])
        out[f"t_{f}"] = float(fit.tvalues[f])
    return out


def attribution_table(strategy: pd.Series, ff: pd.DataFrame) -> pd.DataFrame:
    """Run every nested model and stack the results."""
    rows = {name: regress(strategy, ff, cols) for name, cols in MODELS.items()}
    rows = {k: v for k, v in rows.items() if v}
    df = pd.DataFrame(rows).T
    order = (["alpha_ann", "alpha_t", "alpha_p", "R2", "R2_adj", "n_obs"]
             + [c for c in df.columns if c.startswith(("beta_", "t_"))])
    return df[[c for c in order if c in df.columns]]


def rolling_alpha_beta(strategy: pd.Series, ff: pd.DataFrame,
                       factors: list[str] | None = None,
                       window: int = TRADING_DAYS) -> pd.DataFrame:
    """Rolling OLS alpha (annualised) and factor betas -- shows exposure drift."""
    factors = factors or MODELS["FF5+MOM"]
    df = pd.concat([strategy.rename("r"), ff[factors + ["RF"]]], axis=1).dropna()
    y = (df["r"] - df["RF"]).to_numpy()
    X = sm.add_constant(df[factors]).to_numpy()
    names = ["alpha"] + factors

    out = np.full((len(df), len(names)), np.nan)
    for i in range(window, len(df) + 1):
        xs, ys = X[i - window:i], y[i - window:i]
        try:
            beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
            out[i - 1] = beta
        except np.linalg.LinAlgError:
            continue
    res = pd.DataFrame(out, index=df.index, columns=names).dropna(how="all")
    res["alpha"] *= TRADING_DAYS
    return res


def factor_contribution(strategy: pd.Series, ff: pd.DataFrame,
                        model: str = "FF5+MOM") -> pd.Series:
    """
    Decompose average annual excess return into the part explained by each factor
    (beta_k * mean(f_k)) plus alpha. Components sum to the mean excess return.
    """
    cols = MODELS[model]
    fit = regress(strategy, ff, cols)
    if not fit:
        return pd.Series(dtype=float)
    df = pd.concat([strategy.rename("r"), ff[cols + ["RF"]]], axis=1).dropna()
    parts = {f: fit[f"beta_{f}"] * df[f].mean() * TRADING_DAYS for f in cols}
    parts["alpha"] = fit["alpha_ann"]
    parts["total excess"] = float((df["r"] - df["RF"]).mean() * TRADING_DAYS)
    return pd.Series(parts)
