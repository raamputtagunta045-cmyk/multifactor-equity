"""
Statistical inference tests.

`test_engine.py` covers the backtest engine -- look-ahead, costs, factor
definitions. This file covers the layer that turns a return series into the
report's *claims*: the attribution regressions (the headline alpha and
t-statistic), the multiple-testing correction, the HAC estimator, the
falsification controls, and the frozen specifications.

These are the calculations a reader has to trust to accept the conclusion, and
they fail silently rather than loudly: a broken HAC estimator still returns a
plausible-looking number, and a broken Deflated Sharpe still returns a
probability. Every test below therefore checks a value that is known
analytically, or an invariant the module's own docstring promises.

Run:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import attribution as A
from src import metrics as M
from src import robustness as R
from src import specs as S
from src.config import TRADING_DAYS


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
TRUE_ALPHA_DAILY = 0.0001
TRUE_BETAS = {
    "Mkt-RF": 0.95, "SMB": 0.30, "HML": -0.20,
    "RMW": 0.10, "CMA": 0.05, "MOM": 0.40,
}


@pytest.fixture
def ff_panel():
    """
    A strategy series built from KNOWN alpha and betas, so the regression has a
    right answer to be measured against rather than merely a self-consistent one.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2010-01-01", periods=2000)
    ff = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0003, 0.010, len(dates)),
        "SMB": rng.normal(0.0000, 0.005, len(dates)),
        "HML": rng.normal(0.0000, 0.005, len(dates)),
        "RMW": rng.normal(0.0000, 0.004, len(dates)),
        "CMA": rng.normal(0.0000, 0.004, len(dates)),
        "MOM": rng.normal(0.0000, 0.006, len(dates)),
        "RF": 0.00002,
    }, index=dates)

    signal = sum(b * ff[f] for f, b in TRUE_BETAS.items())
    noise = pd.Series(rng.normal(0.0, 0.0006, len(dates)), index=dates)
    strategy = ff["RF"] + TRUE_ALPHA_DAILY + signal + noise
    return strategy, ff


def _panel(fwd_by_period, n_factors=5):
    """Build the (date, raw_factors, forward_returns) triples the controls consume."""
    out = []
    for i, fwd in enumerate(fwd_by_period):
        raw = pd.DataFrame(0.0, index=fwd.index,
                           columns=[f"f{j}" for j in range(n_factors)])
        out.append((pd.Timestamp("2015-01-01") + pd.DateOffset(months=i), raw, fwd))
    return out


# ---------------------------------------------------------------------------
# 1. Attribution -- the headline alpha must be recoverable
# ---------------------------------------------------------------------------
def test_regress_recovers_known_alpha_and_betas(ff_panel):
    """
    The report's central claim is an alpha and a t-statistic. If the regression
    cannot recover coefficients it was handed by construction, every number in
    section 10.2 is unfounded.
    """
    strategy, ff = ff_panel
    fit = A.regress(strategy, ff, A.MODELS["FF5+MOM"])

    assert fit["alpha_ann"] == pytest.approx(TRUE_ALPHA_DAILY * TRADING_DAYS, abs=0.012)
    for factor, beta in TRUE_BETAS.items():
        assert fit[f"beta_{factor}"] == pytest.approx(beta, abs=0.02)
    assert fit["n_obs"] == len(strategy)
    assert 0.0 <= fit["R2"] <= 1.0


def test_alpha_is_annualised_by_trading_days(ff_panel):
    """alpha_ann must be the daily intercept scaled by 252, not the raw intercept."""
    strategy, ff = ff_panel
    fit = A.regress(strategy, ff, ["Mkt-RF"])
    implied_daily = fit["alpha_ann"] / TRADING_DAYS
    assert abs(implied_daily) < 0.01, "alpha_ann looks like a daily number"
    assert fit["alpha_ann"] == pytest.approx(implied_daily * TRADING_DAYS, rel=1e-12)


def test_regress_refuses_short_samples(ff_panel):
    """Fewer than 60 observations returns {} rather than an unstable fit."""
    strategy, ff = ff_panel
    assert A.regress(strategy.iloc[:30], ff.iloc[:30], ["Mkt-RF"]) == {}
    assert A.regress(strategy.iloc[:200], ff.iloc[:200], ["Mkt-RF"]) != {}


def test_factor_contribution_components_sum_to_total(ff_panel):
    """
    The docstring promises the components sum to the mean excess return. This is
    exact for OLS with an intercept, so any drift means the decomposition in the
    report does not add up.
    """
    strategy, ff = ff_panel
    parts = A.factor_contribution(strategy, ff, "FF5+MOM")
    rebuilt = parts.drop("total excess").sum()
    assert rebuilt == pytest.approx(parts["total excess"], rel=1e-9)


def test_attribution_table_covers_every_nested_model(ff_panel):
    strategy, ff = ff_panel
    tbl = A.attribution_table(strategy, ff)
    assert list(tbl.index) == list(A.MODELS)
    assert {"alpha_ann", "alpha_t", "n_obs"} <= set(tbl.columns)
    # CAPM has no SMB loading; FF3 does. A shared column must stay NaN for CAPM.
    assert pd.isna(tbl.loc["CAPM", "beta_SMB"])


def test_rolling_beta_tracks_the_true_loading(ff_panel):
    strategy, ff = ff_panel
    roll = A.rolling_alpha_beta(strategy, ff, ["Mkt-RF"], window=252)
    assert roll["Mkt-RF"].median() == pytest.approx(TRUE_BETAS["Mkt-RF"], abs=0.05)
    assert len(roll) == len(strategy) - 252 + 1


# ---------------------------------------------------------------------------
# 2. HAC inference -- the correction must actually correct
# ---------------------------------------------------------------------------
def test_newey_west_with_zero_lags_is_the_naive_tstat():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0.001, 0.01, 500))
    naive = x.mean() / np.sqrt(((x - x.mean()) ** 2).mean() / len(x))
    assert M.newey_west_tstat(x, lags=0) == pytest.approx(naive, rel=1e-12)


def test_newey_west_shrinks_tstat_under_positive_autocorrelation():
    """
    The entire reason for using HAC: positively autocorrelated returns carry less
    independent information than their count suggests. If the lagged terms were
    dropped or mis-signed, the t-statistic would not shrink -- and the report
    would overstate significance.
    """
    rng = np.random.default_rng(5)
    n, rho = 2000, 0.6
    e = rng.normal(0, 0.01, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    s = pd.Series(x + 0.002)

    naive = M.newey_west_tstat(s, lags=0)
    hac = M.newey_west_tstat(s, lags=20)
    assert abs(hac) < abs(naive)


def test_newey_west_returns_nan_on_tiny_samples():
    assert np.isnan(M.newey_west_tstat(pd.Series([0.01, -0.01, 0.02])))


# ---------------------------------------------------------------------------
# 3. Multiple-testing correction
# ---------------------------------------------------------------------------
def test_probabilistic_sharpe_falls_as_the_hurdle_rises():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0006, 0.01, 1500))
    assert M.probabilistic_sharpe(r, 0.0) > M.probabilistic_sharpe(r, 1.0)
    assert 0.0 <= M.probabilistic_sharpe(r, 0.0) <= 1.0


def test_deflated_sharpe_penalises_more_trials():
    """
    More configurations examined => a higher expected best-by-luck Sharpe => a
    lower deflated probability. A correction that ignored the trial count would
    let the parameter sweep in section 12.2 look free.
    """
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0.0006, 0.01, 1500))

    dsr_few, star_few = M.deflated_sharpe(r, np.linspace(-1, 1, 20))
    dsr_many, star_many = M.deflated_sharpe(r, np.linspace(-1, 1, 400))

    assert star_many > star_few > 0
    assert dsr_many < dsr_few


def test_deflated_sharpe_needs_a_real_trial_spread():
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 500))
    assert np.isnan(M.deflated_sharpe(r, [0.5])[0])          # too few trials
    assert np.isnan(M.deflated_sharpe(r, [0.5, 0.5, 0.5])[0])  # zero variance


# ---------------------------------------------------------------------------
# 4. Benchmark-relative statistics
# ---------------------------------------------------------------------------
def test_information_ratio_is_nan_against_itself():
    r = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 400))
    assert np.isnan(M.information_ratio(r, r))


def test_capture_ratios_on_a_hand_built_series():
    b = pd.Series([0.02, -0.01, 0.04, -0.03])
    r = pd.Series([0.01, -0.02, 0.02, -0.06])
    up, dn = M.capture_ratios(r, b)
    assert up == pytest.approx(0.015 / 0.03)
    assert dn == pytest.approx(-0.04 / -0.02)


# ---------------------------------------------------------------------------
# 5. Cost break-even
# ---------------------------------------------------------------------------
def test_breakeven_interpolates_the_zero_crossing():
    df = pd.DataFrame({"total_bps_1way": [0, 10, 20],
                       "excess_CAGR": [0.02, 0.01, -0.01]})
    assert R.implied_breakeven_bps(df) == pytest.approx(15.0)


def test_breakeven_is_infinite_when_cost_never_bites():
    df = pd.DataFrame({"total_bps_1way": [0, 10, 20],
                       "excess_CAGR": [0.03, 0.02, 0.01]})
    assert np.isinf(R.implied_breakeven_bps(df))


def test_breakeven_is_nan_when_never_profitable():
    df = pd.DataFrame({"total_bps_1way": [0, 10, 20],
                       "excess_CAGR": [-0.01, -0.02, -0.03]})
    assert np.isnan(R.implied_breakeven_bps(df))


# ---------------------------------------------------------------------------
# 6. The falsification controls
# ---------------------------------------------------------------------------
def test_equal_weight_universe_charges_turnover_exactly():
    """First period rebalances from cash (turnover 1.0); an unchanged roster is free."""
    fwd1 = pd.Series({"A": 0.01, "B": 0.03})
    fwd2 = pd.Series({"A": 0.02, "B": 0.04})
    out = R.equal_weight_universe(_panel([fwd1, fwd2]), cost_bps=8.0)
    assert out.iloc[0] == pytest.approx(0.02 - 8.0 / 1e4)
    assert out.iloc[1] == pytest.approx(0.03)


def test_random_null_is_reproducible_under_its_seed():
    """
    The README claims the null is seeded and reproducible. If it drifts, the
    "31st percentile" headline is not a fixed quantity.
    """
    rng = np.random.default_rng(21)
    panel = _panel([pd.Series(rng.normal(0.01, 0.05, 30),
                              index=[f"T{i}" for i in range(30)]) for _ in range(24)])

    a = R.random_portfolio_null(panel, n_names=10, n_sims=40, seed=42)
    b = R.random_portfolio_null(panel, n_names=10, n_sims=40, seed=42)
    c = R.random_portfolio_null(panel, n_names=10, n_sims=40, seed=7)

    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["CAGR"], c["CAGR"])
    assert set(a.columns) == {"CAGR", "vol", "Sharpe"}


def test_random_null_skips_periods_thinner_than_the_portfolio():
    thin = pd.Series([0.01, 0.02], index=["A", "B"])
    out = R.random_portfolio_null(_panel([thin] * 24), n_names=10, n_sims=5)
    assert out.empty


def test_null_percentile_counts_strictly_below():
    null = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert R.null_percentile(3.0, null) == pytest.approx(0.5)
    assert R.null_percentile(0.0, null) == 0.0
    assert R.null_percentile(9.0, null) == 1.0


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    rng = np.random.default_rng(17)
    r = pd.Series(rng.normal(0.0005, 0.01, 600))

    a = R.stationary_bootstrap_sharpe(r, n_boot=60, seed=42)
    b = R.stationary_bootstrap_sharpe(r, n_boot=60, seed=42)

    assert a == b
    assert a["ci_2.5%"] <= a["sharpe"] <= a["ci_97.5%"]
    assert a["sharpe"] == pytest.approx(M.sharpe(r))
    assert 0.0 <= a["p_sharpe_le_0"] <= 1.0


def test_bootstrap_refuses_short_series():
    assert R.stationary_bootstrap_sharpe(pd.Series(np.zeros(50))) == {}


def test_subperiod_analysis_drops_regimes_without_enough_data():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    r = pd.Series(0.0004, index=dates)
    bmk = pd.Series(0.0003, index=dates)
    rf = pd.Series(0.00001, index=dates)

    tbl = R.subperiod_analysis(r, bmk, rf, regimes={
        "full year": ("2020-01-01", "2020-12-31"),
        "one week": ("2020-03-01", "2020-03-07"),
    })
    assert list(tbl.index) == ["full year"]
    assert tbl.loc["full year", "excess"] > 0


def test_collect_trial_sharpes_gathers_and_drops_missing():
    a = pd.DataFrame({"Sharpe_net": [0.4, np.nan, 0.6]})
    b = pd.DataFrame({"Sharpe_net": [0.5]})
    unrelated = pd.DataFrame({"other": [1.0]})
    got = R.collect_trial_sharpes(a, b, unrelated)
    assert sorted(got) == [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# 7. Frozen specifications
# ---------------------------------------------------------------------------
def test_ic_weights_match_the_documented_derivation():
    """
    These weights are frozen literals precisely so the out-of-sample test cannot
    absorb test-period information. Pinning them here makes any silent edit a
    test failure rather than an unnoticed change to the protocol.
    """
    assert S.IC_WEIGHTS["momentum"] == pytest.approx(0.115, abs=5e-4)
    assert S.IC_WEIGHTS["residual_momentum"] == pytest.approx(0.258, abs=5e-4)
    assert S.IC_WEIGHTS["low_volatility"] == 0.0
    assert S.IC_WEIGHTS["low_beta"] == pytest.approx(0.065, abs=5e-4)
    assert S.IC_WEIGHTS["reversal"] == pytest.approx(0.562, abs=5e-4)
    # rounded to 4dp, so the sum is 1 only to within rounding error
    assert sum(S.IC_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-3)


def test_negative_information_ratios_are_floored_not_shorted():
    """A negative IC_IR must contribute zero weight, never a negative one."""
    assert S.IS_IC_IR["low_volatility"] < 0
    assert all(w >= 0 for w in S.IC_WEIGHTS.values())


def test_equal_weight_spec_uses_no_fitted_information():
    assert set(S.EW_WEIGHTS) == set(S.IS_IC_IR)
    assert len(set(S.EW_WEIGHTS.values())) == 1
    assert sum(S.EW_WEIGHTS.values()) == pytest.approx(1.0)


def test_in_sample_and_out_of_sample_windows_do_not_overlap():
    assert pd.Timestamp(S.IS_END) < pd.Timestamp(S.OOS_START)
    assert pd.Timestamp(S.FULL_START) == pd.Timestamp(S.IS_START)
    assert pd.Timestamp(S.FULL_END) == pd.Timestamp(S.OOS_END)


def test_specs_carry_their_weights_into_the_config():
    assert S.SPEC_EW.factor_weights == S.EW_WEIGHTS
    assert S.SPEC_IC.factor_weights == S.IC_WEIGHTS
    assert set(S.SPECS) == {"EW (a priori)", "IC-weighted (IS-fitted)"}
