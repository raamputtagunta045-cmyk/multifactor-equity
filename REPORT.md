# Does a Price-Based Multi-Factor Composite Add Value in US Large Caps?

**A systematic research study on the S&P 500, 1999–2026**

---

## 1. Executive Summary

I built a systematic long-only multi-factor equity strategy on the S&P 500 and tested
it against the strictest controls I could construct. **The evidence does not support
the hypothesis.** The strategy's apparent edge dissolves under every control that
matters:

| Test | Result |
|---|---|
| Out-of-sample vs SPY (2013–2026) | **−5.3%/yr** excess return, IR −0.46 |
| Alpha vs FF5 + momentum (full sample) | **−2.5%/yr**, t = −1.56 (indistinguishable from zero) |
| vs 500 random 50-stock portfolios | CAGR at the **31st percentile** — below the random median |
| vs equal-weighting the whole universe | Equal-weight wins on **both** return (9.2% vs 7.0%) and Sharpe (0.61 vs 0.57) |
| Parameter sweep gradient | Sharpe improves monotonically as the strategy becomes **less** active |
| In-sample data coverage | Only **46–70%** of true index members are priceable in the window where the strategy "works" |

The one genuinely positive in-sample signal — short-term reversal (IC t = 1.89) — is
also the most expensive to trade, and the cost break-even analysis shows why that
matters.

The most useful output of this project is not a strategy. It is a **falsification
pipeline**: point-in-time universe reconstruction, a measured (not assumed-away)
survivorship audit, a random-portfolio null, walk-forward re-fitting, and a Deflated
Sharpe correction. Applied honestly, it says *no* — which is the correct answer for
the overwhelming majority of backtests that get published saying *yes*.

---

## 2. Research Question

> Within a liquid US large-cap universe, does a transparent composite of price-derived
> cross-sectional factors — momentum, residual momentum, low volatility, low beta and
> short-term reversal — generate risk-adjusted returns that survive (a) realistic
> transaction costs, (b) correction for known risk-factor exposures, and (c) an
> out-of-sample test?

Three deliberate scoping choices:

1. **Large caps only.** The S&P 500 is where factor anomalies are most heavily arbitraged
   and where an individual researcher's data is most reliable. If a factor works here,
   it is economically interesting; if it only works in micro-caps, it is likely
   uninvestable.
2. **Price-derived factors only in the headline test.** These are the only factors that
   can be made strictly point-in-time with freely available data (§5).
3. **The benchmark is the hurdle, not zero.** A strategy that returns 8% while SPY
   returns 9% has failed, regardless of its Sharpe ratio.

---

## 3. Hypothesis

**H₁ (primary).** An equally-weighted composite of the five factors above, used to
select the top 50 names from the S&P 500 and rebalanced monthly, earns positive
risk-adjusted excess return relative to SPY, net of costs, that is not fully explained
by the Fama-French five factors plus momentum.

Formally, in the regression

$$r_{p,t} - r_{f,t} = \alpha + \beta_M \text{MKT}_t + \beta_S \text{SMB}_t + \beta_H \text{HML}_t + \beta_R \text{RMW}_t + \beta_C \text{CMA}_t + \beta_U \text{MOM}_t + \varepsilon_t$$

H₁ predicts $\alpha > 0$ with $t(\alpha) > 2$ using Newey-West standard errors.

**H₀ (null).** $\alpha \le 0$: the composite is a repackaging of known risk premia and
a mechanical equal-weighting tilt, and any gross edge is consumed by trading costs.

**Falsification criteria — fixed before running the out-of-sample test.** H₁ is rejected
if *any* of the following holds:

- out-of-sample excess return over SPY is negative;
- FF5+MOM alpha is not significantly positive;
- the strategy does not beat the median of a random-portfolio null drawn from the same
  universe;
- performance is not robust to reasonable parameter perturbation.

Stating these in advance is what stops a negative result from being quietly reframed as
a positive one.

---

## 4. Literature and Economic Intuition

Each factor is included because there is a *mechanism*, not merely a historical
correlation. A factor with no economic story is a data-mining candidate.

**Momentum (Jegadeesh & Titman 1993).** Twelve-month winners continue to outperform.
The mechanisms proposed are behavioural — underreaction to news, the disposition
effect, and delayed diffusion of information. The most recent month is skipped because
one-month returns reverse (Jegadeesh 1990); including it contaminates the signal.
Momentum is known to suffer rare, violent crashes when the market rebounds off a
bottom (Daniel & Moskowitz 2016) — most famously in 2009.

**Residual momentum (Blitz, Huij & Martens 2011).** Strip the market-beta component
out of momentum and what remains is stock-specific. Because the crash risk in momentum
comes largely from its time-varying beta, residual momentum has historically delivered
a higher Sharpe ratio with far smaller drawdowns. *(Note: implementing this correctly
turned out to be the single most important technical detail in this project — §11.)*

**Low volatility (Ang, Hodrick, Xing & Zhang 2006) and low beta (Frazzini & Pedersen
2014).** Low-risk stocks have historically earned higher risk-adjusted returns than
CAPM predicts. The mechanism is leverage aversion: investors who want high returns but
cannot borrow bid up high-beta stocks instead, depressing their expected returns.
This is an *inversion* of the textbook risk-return relationship, which is precisely why
it is interesting — and why it should be treated sceptically in a universe where every
member is already a large, relatively low-volatility company.

**Short-term reversal (Jegadeesh 1990; Lehmann 1990).** One-month losers bounce. The
mechanism is liquidity provision: a stock that falls sharply on non-fundamental order
flow pays a premium to whoever absorbs the imbalance. This is a genuine premium, but it
is compensation for *providing a service* — and it decays quickly, which means high
turnover and high cost. This tension is central to the results.

**Why combine them?** Factors that are individually noisy but weakly correlated
diversify. If each factor's information coefficient is $IC_k$ and the average pairwise
correlation is $\rho$, the composite's IC scales roughly as

$$IC_{\text{composite}} \approx \frac{\sum_k w_k IC_k}{\sqrt{\sum_k w_k^2 + \rho \sum_{j \ne k} w_j w_k}}$$

so combining genuinely distinct signals raises the information ratio. The measured
correlation matrix (§10) shows how much diversification this composite actually gets —
considerably less than the five-factor count suggests.

---

## 5. Data

| Dataset | Source | Coverage | Status |
|---|---|---|---|
| S&P 500 point-in-time membership | `fja05680/sp500` | 1996-01 – 2026-06, 2,718 change dates | real |
| Daily adjusted prices, close, volume | Yahoo Finance (`yfinance`) | 1996-01 – 2026-06, 7,672 days | real, 767 tickers |
| Fama-French 5 factors + momentum, daily | Ken French Data Library | through 2026-06 | real |
| Benchmark | SPY adjusted close | 1996–2026 | real |
| Point-in-time fundamentals | — | — | **unavailable** |

### 5.1 The survivorship problem, measured rather than assumed away

Point-in-time membership fixes *universe* look-ahead: a stock enters the eligible set
only from the date it actually joined the index. It does **not** fix survivorship bias,
because Yahoo Finance serves no history for delisted securities. Verified directly —
`SIVB`, `FRC`, `ATVI`, `TWTR`, `CERN`, `XLNX`, `LEH`, `ENRNQ` all return **zero rows**.

Of **1,206** tickers that were ever S&P 500 members between 1996 and 2026, only **767
(63.6%)** can be priced. `data.audit_survivorship()` measures the hole at every
rebalance:

| Year | Coverage | Year | Coverage |
|---|---|---|---|
| 1999 | 46.2% | 2015 | 74.7% |
| 2003 | 53.5% | 2020 | 87.3% |
| 2008 | 62.2% | 2024 | 95.8% |
| 2012 | 69.5% | 2026 | 98.8% |

**Mean coverage 71.2%; worst year 46.2%.** The bias is not uniform — it is *worst
exactly where the strategy looks best* (the 1999–2012 in-sample window) and nearly
absent in the out-of-sample window. Any in-sample outperformance must be discounted
accordingly, and §10 shows this is not a hypothetical concern.

*Figure 9 — `09_survivorship.png`. The shaded band is the unrecoverable survivorship
hole: index members that existed on the date but cannot be priced.*

The direction of the bias is upward: companies that were removed from the index
generally underperformed first. Excluding them inflates measured returns for the
universe *and* for any strategy selecting from it.

### 5.2 Why value and quality are specified but not backtested

A non-look-ahead value factor requires fundamentals keyed by **filing date**, not fiscal
period end, and **never restated**. A quarter ending 31 December is typically filed in
late February; using it on 1 January grants eight weeks of hindsight — more than enough
to manufacture large fake alpha.

`yfinance` fails on both counts, verified directly:
- `balance_sheet` returns **5 annual periods**; `quarterly_balance_sheet` returns
  **7 quarters** — against a 27-year price sample;
- columns are keyed by **fiscal period end with no filing-date field**, so the
  publication lag cannot be applied;
- values reflect the **latest restatement**.

I chose not to backtest a contaminated value factor and report the resulting inflated
Sharpe. Instead `src/fundamentals.py` (a) specifies the required schema, (b) implements
the full value/quality/size/investment mathematics against it, and (c) ships a
clearly-labelled synthetic generator behind a hard guard (`assert_not_synthetic`) that
raises if synthetic output ever reaches a reported result. Connecting Sharadar SF1
(which carries `datekey` and retains delisted tickers) or Compustat PIT activates these
factors with no other code change.

**This is a real limitation, not a stylistic choice: the study tests a price-based
composite, and its conclusions do not extend to value or quality.**

---

## 6. Methodology

### 6.1 Universe screens

Applied at each rebalance date *t*, using only data available at *t*:

1. member of the S&P 500 on *t* (point-in-time);
2. ≥ 252 trading days of price history;
3. close ≥ \$5 (avoids microstructure noise and sub-penny tick effects);
4. 60-day median dollar volume ≥ \$5m (tradability).

Mean eligible universe: **348 names** (273 in-sample, 427 out-of-sample — the
difference is the survivorship coverage gradient, not a change in the index).

### 6.2 Timing convention

This is where most backtests leak information, so it is stated explicitly:

- factor values on date *t* use prices through the **close of *t***;
- target weights are therefore known only **after** that close;
- trades execute at the **close of *t* + 1** (`execution_lag = 1`);
- new weights earn returns from *t* + 2 onward.

Between rebalances, positions **drift with prices** rather than being silently held at
constant weight:

$$w_{i,d} = \frac{w_{i,d-1}(1 + r_{i,d})}{1 + r_{p,d}}, \qquad r_{p,d} = \sum_i w_{i,d-1} r_{i,d}$$

Holding weights constant without trading is a common and subtle error: it implicitly
assumes free continuous rebalancing and overstates returns.

### 6.3 Frictions

| Component | Setting | Rationale |
|---|---|---|
| Commission | 1 bp/side | institutional electronic execution |
| Spread | 5 bps/side | half-spread on S&P 500 names |
| Slippage | 2 bps/side | market-impact allowance |
| **Total** | **8 bps/side** | charged on realised one-way turnover |
| Participation cap | 5% of 60-day ADV | per name, per day |
| Assumed AUM | \$100m | scales the liquidity constraint |

Liquidity is enforced *before* costing. The maximum weight change for name *i* in one
day is

$$\Delta w_i^{\max} = \frac{\text{participation cap} \times \text{ADV}_i}{\text{AUM}}$$

and any unfilled remainder stays in the previous position — so a large fund genuinely
cannot reach its target in thin names, exactly as in live trading.

---

## 7. Mathematical Formulation

Let $P_{i,t}$ be the split- and dividend-adjusted close, $r_{i,t} = P_{i,t}/P_{i,t-1} - 1$
the daily total return, $r_{m,t}$ the market return and $r_{f,t}$ the risk-free rate.
Set $L = 252$ (one year) and $s = 21$ (one month).

**1 — Momentum (12-1)**

$$\text{MOM}_{i,t} = \frac{P_{i,t-s}}{P_{i,t-L}} - 1$$

**2 — Residual momentum.** Estimate the market model over a **three-year** window
$(t - L_e, t]$ with $L_e = 756$:

$$r_{i,u} - r_{f,u} = \alpha_i + \beta_i (r_{m,u} - r_{f,u}) + \varepsilon_{i,u}$$

then accumulate residuals over the 12-1 sub-window only:

$$\text{RESMOM}_{i,t} = \frac{\sum_{u = t-L}^{t-s} \varepsilon_{i,u}}{\sigma(\varepsilon_i)}$$

> **The estimation window must be strictly longer than the accumulation window.** OLS
> forces $\sum_u \varepsilon_{i,u} = 0$ over the estimation window. If the two windows
> coincide, then $\sum_{u=t-L}^{t-s}\varepsilon_{i,u} = -\sum_{u=t-s}^{t}\varepsilon_{i,u}$
> — the signal becomes *minus the skipped month's residual*, i.e. reversal wearing a
> momentum label. This is not hypothetical: it is the bug I shipped first and caught
> via the factor correlation matrix (§11). Tests
> `test_residual_momentum_is_not_reversal` and
> `test_residual_momentum_degenerates_when_windows_coincide` pin both directions.

**3 — Low volatility**

$$\text{LOWVOL}_{i,t} = -\sqrt{252}\thickspace\thickspace \sigma\big(r_{i,u}\big)_{u \in (t-L,\thinspace t]}$$

**4 — Low beta**

$$\text{LOWBETA}_{i,t} = -\frac{\operatorname{Cov}(r_i, r_m)}{\operatorname{Var}(r_m)} \bigg|_{(t-L,\thinspace t]}$$

**5 — Short-term reversal**

$$\text{REV}_{i,t} = -\left(\frac{P_{i,t}}{P_{i,t-s}} - 1\right)$$

All five are signed so that **higher is better**, which makes compositing a plain
weighted sum.

**Cross-sectional standardisation.** At each date, within the eligible universe:
winsorise at the 1st/99th percentiles, then

$$z^{(k)}_{i,t} = \frac{f^{(k)}_{i,t} - \mu_t\big(f^{(k)}\big)}{\sigma_t\big(f^{(k)}\big)}, \qquad S_{i,t} = \sum_k w_k\thinspace z^{(k)}_{i,t}$$

re-standardised to unit cross-sectional variance. Winsorising *before* z-scoring matters:
a single extreme value inflates $\sigma_t$ and compresses every other stock's score.

A stock is scored only if it has at least half its factor weight present; the weights
are then renormalised over the factors it does have, so one missing input does not
drop a name entirely.

---

## 8. Portfolio Construction

**Selection.** Rank by $S_{i,t}$; take the top $n = 50$.

**Weighting** (three schemes, all tested):

| Scheme | Definition |
|---|---|
| Equal | $w_i = 1/N$ |
| Score tilt (default) | $w_i \propto S_i - \min_j S_j + 0.25$ |
| Inverse volatility | $w_i \propto 1/\sigma_i$ |

Each is capped at 5% per position and renormalised. Capping is applied **iteratively**,
because renormalising after a cap can push another name back above it — a single pass
silently violates the constraint.

**Rebalancing.** Month-end, with an optional rank buffer: a held name is retained while
it stays inside the top $n(1 + b)$ ranks, which damps turnover from names oscillating
around the selection boundary.

---

## 9. Backtesting Framework

Implemented in `src/backtest.py`. Per trading day:

1. accrue the day's return on yesterday's drifted weights;
2. execute any trade scheduled for today (liquidity-throttled, then costed on realised
   turnover);
3. if today is a rebalance date, compute the signal and schedule execution for
   *t* + `execution_lag`.

Three analyses share one cached cross-sectional panel (`collect_factor_panel`) so that
the decile test, the IC test and the null test are computed on **identical** universes
and forward returns — otherwise differences between them could reflect sample
construction rather than signal quality.

**Statistical treatment.** Daily strategy returns are autocorrelated and fat-tailed, so:
Newey-West HAC standard errors throughout (Bartlett kernel, $L = \lfloor 4(T/100)^{2/9}\rfloor$);
a stationary block bootstrap (Politis-Romano) for Sharpe confidence intervals; and the
Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado 2014) to correct for
multiple testing:

$$\widehat{SR^*} = \sqrt{\operatorname{Var}(SR_{\text{trials}})}\left[(1-\gamma)\thinspace\Phi^{-1}\negthinspace\left(1 - \tfrac{1}{N}\right) + \gamma\thinspace\Phi^{-1}\negthinspace\left(1 - \tfrac{1}{Ne}\right)\right]$$

where $N$ is the number of configurations examined and $\gamma$ is the Euler-Mascheroni
constant. Reporting a Sharpe ratio selected as the best of many trials without this
correction is one of the most common failures in published backtests.

---

## 10. Results

### 10.1 Headline performance (net of 8 bps/side)

| Metric | EW · IS<br>1999–2012 | EW · **OOS**<br>2013–2026 | EW · Full | IC-wtd · IS | IC-wtd · **OOS** | SPY · IS | SPY · **OOS** | SPY · Full |
|---|---|---|---|---|---|---|---|---|
| CAGR | 6.04% | 9.65% | 8.04% | 6.35% | 10.35% | 2.84% | 14.99% | 8.63% |
| Volatility | 17.6% | 16.0% | 16.9% | 28.4% | 21.9% | 21.3% | 16.9% | 19.3% |
| Sharpe | 0.290 | 0.545 | 0.422 | 0.277 | 0.479 | 0.130 | 0.810 | 0.421 |
| Sortino | 0.407 | 0.756 | 0.589 | 0.401 | 0.681 | 0.184 | 1.139 | 0.594 |
| Max drawdown | −55.2% | −36.5% | −55.2% | −75.6% | −45.6% | −55.2% | −33.7% | −55.2% |
| Calmar | 0.109 | 0.264 | 0.146 | 0.084 | 0.227 | 0.052 | 0.445 | 0.156 |
| Win rate (monthly) | 63.5% | 62.7% | 63.2% | 58.1% | 61.5% | 57.1% | 69.8% | 63.3% |
| Turnover (ann., 1-way) | 11.7× | 14.2× | 12.9× | 17.6× | 20.5× | — | — | — |
| **Excess CAGR vs SPY** | **+3.20%** | **−5.34%** | **−0.60%** | **+3.51%** | **−4.63%** | — | — | — |
| **Information ratio** | **+0.204** | **−0.459** | **−0.089** | **+0.354** | **−0.268** | — | — | — |
| Up capture | 0.720 | 0.713 | 0.719 | 1.134 | 1.054 | — | — | — |
| Down capture | 0.686 | 0.720 | 0.701 | 1.093 | 1.105 | — | — | — |

*Figure 1 — `results/figures/01_cumulative_returns.png`. Over the full 27 years SPY
compounds to 9.7×, the IC-weighted strategy to 9.5×, the equal-weighted composite to 8.3×.*

**The sign flip between in-sample and out-of-sample is the headline.** Both
specifications beat SPY by ~3.2–3.5%/yr in 1999–2012 and lost to it by ~4.6–5.3%/yr in
2013–2026. Note that the a priori equal-weighted spec — which used **no** fitted
information at all — flips just as hard as the IS-fitted one. This is important: the
reversal is **not** caused by parameter overfitting. It is caused by a structural
defensive tilt (up capture 0.72, down capture 0.70, market beta 0.71) that was rewarded
by a decade containing two −50% drawdowns and punished by a decade that had none.

Note also that the OOS Sharpe (0.545) is *higher* than the IS Sharpe (0.290) while the
OOS **excess** return is far worse. A strategy can look better on standalone
risk-adjusted metrics while losing decisively to its benchmark — which is why the
benchmark, not zero, must be the hurdle.

### 10.2 Risk-factor attribution (full sample, net returns)

| Model | Alpha (ann.) | t(alpha) | R² | β_MKT | β_SMB | β_HML | β_MOM | β_RMW | β_CMA |
|---|---|---|---|---|---|---|---|---|---|
| CAPM | +1.03% | 0.56 | 0.673 | 0.710 | | | | | |
| FF3 | +0.98% | 0.55 | 0.681 | 0.718 | −0.106 | +0.108 | | | |
| Carhart 4 | −0.77% | −0.46 | 0.731 | 0.764 | −0.102 | +0.216 | +0.243 | | |
| FF5 | −1.26% | −0.75 | 0.723 | 0.799 | −0.049 | −0.062 | | +0.249 | +0.442 |
| **FF5 + MOM** | **−2.49%** | **−1.56** | **0.760** | 0.828 | −0.045 | +0.065 | +0.212 | +0.243 | +0.334 |

**Alpha is zero and drifts negative as the model is enriched** — the signature of a
strategy that repackages known premia. R² climbs from 0.673 to 0.760: three-quarters
of the strategy's daily variance is explained by six public factors. The loadings say
what the strategy actually is: a **0.83-beta portfolio with momentum, profitability and
conservative-investment tilts** — all purchasable through liquid factor ETFs at ~15 bps
a year, versus this strategy's ~112 bps of annual trading cost.

*Figure 13 — `13_rolling_beta.png`. These exposures are not stable: market beta drifts
between roughly 0.6 and 1.0 across regimes, so even the factor description of the
strategy is a moving target.*

### 10.3 Factor efficacy, in-sample (1999–2012, 167 monthly cross-sections)

| Factor | Q1 | Q3 | Q5 | Spread | t(spread) | Mean IC | IC IR | t(IC) |
|---|---|---|---|---|---|---|---|---|
| momentum | 7.50% | 6.74% | 6.79% | −0.71% | −0.11 | 0.0066 | 0.030 | 0.39 |
| residual momentum | 5.25% | 9.24% | 7.72% | +2.48% | 0.55 | 0.0110 | 0.067 | 0.87 |
| low volatility | 10.05% | 6.71% | 6.10% | −3.95% | −0.57 | −0.0015 | −0.006 | −0.07 |
| low beta | 7.64% | 6.82% | 5.77% | −1.88% | −0.27 | 0.0048 | 0.017 | 0.22 |
| **reversal** | 4.83% | 6.87% | 9.54% | **+4.71%** | **1.38** | **0.0239** | **0.146** | **1.89** |
| composite | | | | | | 0.0114 | 0.052 | 0.67 |

*Figure 8 — `08_factor_ic.png`.*

**Only short-term reversal shows meaningful predictive power, and even it clears t=1.89
— short of the t>3 that Harvey, Liu & Zhu (2016) argue is the appropriate bar once
multiple testing across the published factor zoo is accounted for.**

Momentum and low volatility are the headline results here, and both are *negative*:

- **Momentum does not work in this universe** (t = 0.39). This is consistent with the
  literature — momentum is concentrated in small- and mid-caps, and 1999–2012 contains
  the 2009 momentum crash. It is not evidence the anomaly does not exist; it is evidence
  it is not harvestable in S&P 500 names.
- **Low volatility has the wrong sign** (Q1 high-vol 10.05% vs Q5 low-vol 6.10%). The
  low-risk anomaly is largely a leverage-constraint story that operates across the
  *whole* market; inside a universe already filtered to 500 mega-caps, most of that
  dispersion has been removed. Survivorship bias also cuts directly here: high-volatility
  names that blew up are the ones missing from the panel, which mechanically flatters
  the high-vol bucket.

### 10.4 Factor correlations — how much diversification is really there

|  | mom | resmom | lowvol | lowbeta | rev |
|---|---|---|---|---|---|
| momentum | 1.000 | 0.755 | 0.103 | 0.040 | 0.003 |
| residual momentum | 0.755 | 1.000 | 0.065 | 0.051 | 0.101 |
| low volatility | 0.065 | 0.065 | 1.000 | **0.772** | 0.003 |
| low beta | 0.040 | 0.051 | **0.772** | 1.000 | −0.001 |
| reversal | 0.003 | 0.101 | 0.003 | −0.001 | 1.000 |

Five factors, but effectively **three** independent bets: a momentum block (ρ = 0.76),
a risk block (ρ = 0.77), and reversal standing alone. Equal-weighting five names
therefore places ~40% of the risk budget on the momentum block and ~40% on the risk
block — both of which have zero measured efficacy — and only 20% on the one signal that
works. This mis-allocation is a direct consequence of counting factors instead of
counting independent sources of return.

---

## 11. A bug worth documenting

The correlation matrix caught a serious error before it reached any result. My first
implementation of residual momentum estimated the market model over the *same* 252-day
window it accumulated residuals over. It produced a beautiful in-sample signal:
spread +5.61%/yr, t = **2.38**, monotonic across quintiles, IC t = 2.53 — comfortably
the best factor in the study.

It also correlated **0.905** with short-term reversal, which is impossible for a signal
that explicitly skips the most recent month.

The cause is algebraic, not numerical. OLS with an intercept forces
$\sum_u \varepsilon_{i,u} = 0$ over the estimation window. Splitting that window into the
accumulation period and the skipped month gives

$$\sum_{u=t-L}^{t-s} \varepsilon_{i,u} \thickspace=\thickspace -\sum_{u=t-s}^{t} \varepsilon_{i,u}$$

so the "12-month residual momentum" signal was exactly **minus the skipped month's
residual return** — a reversal signal with a momentum label. Since reversal is the one
factor that genuinely works here, the bug was *stealing that factor's performance* and
reporting it under a different name.

The fix is to estimate betas over a strictly longer window (756 days) than the
accumulation window (231 days), so the zero-sum constraint does not bind on the
sub-window. After the fix, residual momentum correlates 0.755 with momentum (as theory
requires — it *is* momentum with beta removed) and 0.101 with reversal, and its
t-statistic falls from 2.38 to 0.55.

Two regression tests pin this permanently:
`test_residual_momentum_is_not_reversal` asserts |ρ| < 0.6 under the fixed
configuration, and `test_residual_momentum_degenerates_when_windows_coincide` asserts
ρ > 0.8 under the broken one, so the failure mode cannot silently return.

**The general lesson: a factor that is suspiciously good deserves a correlation check
against every other factor before it deserves a backtest.**

---

## 12. Robustness Tests

### 12.1 Control 1 — is the strategy better than random?

500 random 50-stock portfolios drawn from the *same* eligible universe, with the same
rebalance schedule and turnover costs:

| Portfolio | CAGR | Sharpe | CAGR percentile | Sharpe percentile |
|---|---|---|---|---|
| Random portfolios (mean of 500) | 7.41% | 0.502 | — | — |
| **Strategy EW (top 50 by score)** | **7.02%** | 0.572 | **31st** | 95th |
| **Strategy IC-weighted (top 50)** | 8.08% | **0.479** | 81st | **29th** |
| **Equal-weight the ENTIRE universe** | **9.24%** | **0.611** | **99.6th** | **100th** |

*Figure 10 — `10_null_distribution.png`.*

This is the most damaging result in the study:

- The EW strategy's CAGR sits **below the median random portfolio**. Its higher Sharpe
  comes entirely from selecting lower-volatility names (16.9% vs 17.3% for random), not
  from picking better ones — the same defensive tilt the attribution already identified.
- The IC-weighted strategy — the one fitted on in-sample data — lands at the **29th
  percentile on Sharpe**, i.e. worse risk-adjusted performance than a coin flip.
- **Simply equal-weighting all ~348 eligible names beats both strategies and 99.6% of
  random 50-stock subsets**, at a fraction of the turnover. The gain is pure
  diversification: same expected return, lower variance, no concentration.

If the factor score contained real information, the selected portfolio would sit in the
right tail of this distribution. It does not.

### 12.2 Control 2 — parameter sensitivity

| Parameter | Values → net Sharpe |
|---|---|
| `n_long` | 25 → 0.408 · 50 → 0.422 · 75 → 0.445 · 100 → 0.466 · **150 → 0.493** |
| `rebalance` | monthly → 0.422 · **quarterly → 0.489** |
| `weighting` | equal → 0.434 · score-tilt → 0.422 · inverse-vol → 0.435 |
| `mom_lookback` | 126 → 0.426 · 189 → 0.436 · 252 → 0.422 · 378 → 0.415 |
| `rev_lookback` | **10 → 0.500** · 21 → 0.422 · 42 → 0.485 |
| `max_weight` | 3% → 0.428 · 5% → 0.422 · 10% → 0.420 |
| `turnover_buffer` | 0 → 0.422 · 0.25 → 0.441 · **0.5 → 0.460** · 1.0 → 0.450 |
| `min_dollar_volume` | \$1m → 0.421 · \$5m → 0.422 · \$20m → 0.424 |

There is no parameter cliff — results are stable, which rules out gross curve-fitting.
But the **direction** of the gradient is the finding: Sharpe improves monotonically as
the portfolio holds **more** names (25→150), trades **less often** (monthly→quarterly),
and applies a **larger turnover buffer**. Every axis points toward *doing less*.

A signal with real information should degrade when you dilute it across 150 names
instead of concentrating it in 25. This one improves. That is the parameter sweep
independently confirming the null result.

### 12.3 Control 3 — performance by market regime

| Regime | Strategy CAGR | SPY CAGR | Excess | Strategy MaxDD | IR |
|---|---|---|---|---|---|
| Dot-com bust 1999–2002 | +7.45% | −6.91% | **+14.36%** | −22.0% | **+0.81** |
| Recovery 2003–2007 | +12.16% | +12.64% | −0.49% | −13.0% | −0.08 |
| GFC 2008–2009 | −16.02% | −10.62% | −5.40% | −50.4% | −0.56 |
| QE bull 2010–2019 | +12.67% | +13.46% | −0.79% | −17.0% | −0.11 |
| COVID 2020 | −3.62% | +18.25% | **−21.88%** | −36.5% | **−1.32** |
| Inflation bear 2022 | −2.78% | −18.24% | **+15.46%** | −15.1% | **+1.05** |
| AI bull 2023–2026 | +10.21% | +22.59% | **−12.38%** | −12.9% | −0.90 |

A coherent and unflattering picture: the strategy wins in **slow, grinding bear markets**
(2000–02, 2022) where its defensive tilt pays, and loses badly in **sharp V-shaped
recoveries** (COVID 2020, −21.9%) and **momentum-driven bulls** (2023–26, −12.4%).

Note it also *lost* during the GFC (−5.4% excess) despite being defensive — the 2009
momentum crash hit the book precisely when its defensiveness should have helped. A
strategy whose protective quality fails in the largest crisis in the sample is not
offering reliable protection.

### 12.4 Control 4 — transaction-cost break-even

| One-way cost | 0 bps | 2 | 5 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| Net CAGR | 9.16% | 8.88% | 8.46% | 7.76% | 7.06% | 6.37% | 5.00% | 2.30% |
| Excess vs SPY | +0.53% | +0.25% | −0.18% | −0.88% | −1.57% | −2.26% | −3.63% | −6.33% |

**Break-even one-way cost: 3.74 bps.** *Figure 11 — `11_cost_breakeven.png`.*

At 12.9× annual turnover, every basis point of cost removes ~13 bps of annual return.
The break-even sits below any realistic execution cost — and, decisively, **even at
literally zero trading cost the strategy beats SPY by only 0.53%/yr.** There is no
gross edge for better execution to rescue.

### 12.5 The real out-of-sample test — walk-forward

The strongest test available: re-run the *entire research procedure* at every point in
time. Train on 6 years, set factor weights by the same IC-IR rule, trade the next 2
years, roll forward. No parameter ever sees its own evaluation period.

| Test window | Strategy CAGR | SPY CAGR | Excess | Dominant fitted weight |
|---|---|---|---|---|
| 2005–2006 | +13.74% | +10.22% | +3.52% | reversal 0.57 |
| 2007–2008 | −33.45% | −18.48% | **−14.97%** | reversal 0.50 |
| 2009–2010 | +27.01% | +20.57% | +6.44% | reversal 0.49 |
| 2011–2012 | +5.22% | +8.75% | −3.54% | reversal 1.00 |
| 2013–2014 | +20.58% | +22.52% | −1.95% | resmom 0.41 |
| 2015–2016 | +10.56% | +6.48% | +4.08% | reversal 0.84 |
| 2017–2018 | +1.63% | +7.80% | −6.18% | reversal 0.55 |
| 2019–2020 | +8.95% | +24.56% | **−15.60%** | reversal 0.45 |
| 2021–2022 | −0.23% | +2.64% | −2.87% | reversal 0.87 |
| 2023–2024 | +0.33% | +25.64% | **−25.32%** | reversal 0.42 |
| 2025–2026 H1 | +7.81% | +18.58% | −10.77% | reversal 0.94 |

**Stitched walk-forward, 21.4 years:**

| | |
|---|---|
| CAGR | **4.35%** |
| Volatility | 23.2% |
| Sharpe | 0.224 |
| **Max drawdown** | **−70.3%** |
| Recovery time | 2,071 days (5.7 years) |
| **Excess CAGR vs SPY** | **−6.48%** |
| **Information ratio** | **−0.42** |
| Up / down capture | 0.958 / 1.009 |

*Figure 12 — `12_walk_forward.png`.*

**Only 3 of 11 folds beat the benchmark.** The honest, fully out-of-sample
implementation of this research produces **4.35% a year with a 70% drawdown**, against
SPY's ~11% — and up/down capture of 0.96/1.01 shows that once weights are fitted
adaptively, even the defensive quality disappears.

The fold table also exposes the mechanism: the IC-IR rule repeatedly loads 40–100% onto
**reversal**, because reversal has the best in-sample IC in nearly every training
window. It then pays that signal's very high turnover in live trading. The procedure
reliably selects the most expensive signal to trade.

### 12.6 Statistical significance — and why the favourable numbers here mislead

| Test | Value |
|---|---|
| Full-sample Sharpe | 0.422 |
| Stationary block bootstrap 95% CI | **[0.207, 0.905]** |
| P(Sharpe ≤ 0) | 0.0005 |
| Trials examined | 27 |
| Expected max Sharpe from noise, SR\* | 0.053 |
| **Deflated Sharpe Ratio** | **0.9946** |

Taken at face value these look like strong evidence. **They are not, and it is important
to say why.**

The bootstrap CI and the Deflated Sharpe both test the hypothesis **SR > 0**. For a
long-only equity portfolio with market beta 0.83, SR > 0 is very nearly a tautology:
equities carry a positive risk premium, so *any* long-only large-cap portfolio — random
ones included — clears that bar. Indeed all 500 random portfolios in §12.1 had Sharpe
ratios between 0.38 and 0.61, every one of them "significant" by this standard.

**The DSR is answering the wrong question.** The right questions, and their answers:

| Question | Correct test | Answer |
|---|---|---|
| Does it beat the benchmark? | excess CAGR, OOS | **−5.34%/yr** |
| Is there skill beyond known factors? | FF5+MOM alpha | **−2.49%/yr, t = −1.56** |
| Does selection beat chance? | random-portfolio null | **31st percentile** |
| Does it survive honest refitting? | walk-forward | **−6.48%/yr, −70% DD** |

All four say no. A high Deflated Sharpe against a zero benchmark is not a defence
against any of them — and quoting it as though it were is a common way that null
results get published as positive ones.

---

## 13. Risk Analysis

**Factor risk.** 76% of daily variance is explained by FF5+MOM. The portfolio is a
levered-down market position (β = 0.83) with momentum (+0.21), profitability (+0.24) and
conservative-investment (+0.33) tilts. Its behaviour is therefore hostage to those
factor cycles, not to stock selection.

**Tail risk.** Excess kurtosis 12.6 and skew −0.23 on daily net returns; 95% daily VaR
−1.57% and CVaR −2.49%. Worst day −11.7%. Gaussian risk models would badly understate
this — position limits and drawdown controls must be set from the empirical distribution.

**Drawdown risk.** −55% peak-to-trough on the frozen specification; **−75.6%** for the
IC-weighted variant and **−70.3%** for the walk-forward implementation, the last taking
5.7 years to recover. Any of these would end most real mandates before recovery arrived.

**Concentration and capacity.** 50 names with a 5% cap. At \$100m AUM with a 5%-of-ADV
participation limit, ~5% of desired trade value went unexecuted per rebalance in-sample
(where the liquid universe is smaller) and 0.7% out-of-sample. Capacity degrades
roughly linearly with AUM; at \$1bn the same limits would leave a large fraction of the
signal untraded, and realised performance would diverge further from the paper result.

**Turnover risk.** 12.9× annual one-way turnover means the strategy's viability is a bet
on execution quality. With break-even at 3.74 bps, a single bad quarter of spreads
erases a year of edge.

**Regime risk.** The strategy is short the "sharp recovery" scenario. Its two worst
periods (COVID 2020, −21.9% excess; AI bull 2023–26, −12.4%) share a signature: violent
upside led by high-beta, high-momentum names that its defensive tilt systematically
underweights.

**Model risk.** As §11 shows, a single algebraic subtlety produced a t = 2.38 factor
that was an artifact. Discovered late, that would have anchored the entire study.

---

## 14. Limitations

1. **Survivorship bias is present and concentrated where it hurts most.** Mean coverage
   71.2%, ranging 46.2% (1999) → 98.8% (2026). The in-sample window — the only window
   where the strategy outperforms — is the one where up to 54% of true index members are
   unpriceable. The bias is upward (removed companies underperformed first), so in-sample
   outperformance is inflated by an unknown but certainly non-trivial amount.
   *Figure 9 — `09_survivorship.png`.* Fixing this requires CRSP delisting returns or
   Sharadar SF1.
2. **No value, quality or size factors.** The study tests a price-based composite only.
   Its conclusions do **not** extend to value or quality, which require point-in-time
   fundamentals (§5.2). It is entirely possible a value/quality composite would behave
   differently; this study cannot say.
3. **Single market, single macro history.** One universe, one country, 27 years
   containing exactly three major drawdowns. Regime-conditional conclusions rest on a
   handful of episodes — the "wins in bear markets" claim rests on two observations.
4. **Costs are modelled as a constant.** Real spreads widen in exactly the volatile
   conditions when this strategy trades most, and market impact scales with the square
   root of participation rate. A constant 8 bps is therefore *optimistic*, which makes
   the break-even result conservative in the right direction.
5. **No short book, no leverage, no shorting costs.** The long-only constraint discards
   half the cross-sectional information; a long/short mode is implemented (`n_short`) but
   is not reported because borrow costs and availability cannot be modelled with free data.
6. **No sector or industry neutralisation in the headline spec.** Sector-neutral
   standardisation is implemented (`sector_neutral`) but unused, because a point-in-time
   GICS history is not freely available. Uncontrolled sector tilts are therefore part of
   the measured risk.
7. **Index-membership data is third-party.** The membership file is derived from index
   change announcements and is not the official S&P index history; small errors near
   reconstitution dates are possible.
8. **Multiple testing is only partly corrected.** The DSR uses the 27 sweep
   configurations, but the true trial count includes every exploratory choice made
   during development — universe filters, factor definitions, the bug fix. The effective
   $N$ is larger than 27, so even the DSR is optimistic.

### Where overfitting could still hide

The IS/OOS split, the frozen specs and the walk-forward defend against parameter
overfitting. They do **not** defend against **selection of the research question itself**
— I chose these five factors because the literature says they work, and that literature
is the product of decades of collective mining over this same US equity sample. This is
Harvey, Liu & Zhu's central point, and no amount of within-study discipline can remove
it. Genuine out-of-sample evidence would require a different market or a different era.

---

## 15. Conclusion

**The hypothesis is rejected.** Every pre-registered falsification criterion fired:

| Criterion (fixed in advance) | Result | Verdict |
|---|---|---|
| OOS excess return over SPY positive | −5.34%/yr | ✗ |
| FF5+MOM alpha significantly positive | −2.49%/yr, t = −1.56 | ✗ |
| Beats the random-portfolio null | 31st percentile CAGR | ✗ |
| Robust to parameter perturbation | Stable, but gradient points to "trade less" | ✗ |

The composite is not a stock-selection strategy. It is a **0.83-beta portfolio with
momentum, profitability and investment tilts**, and those exposures explain 76% of its
variance and all of its return. Its in-sample advantage came from a defensive tilt that
happened to suit 1999–2012, amplified by survivorship bias that is worst in exactly that
window. Out of sample the tilt became a liability.

Three findings are worth carrying forward beyond this strategy:

1. **Within the S&P 500, classic momentum and low-volatility have no measurable
   cross-sectional efficacy** (IC t = 0.39 and −0.07). These anomalies live in smaller,
   less liquid names. Applying large-cap implementations of small-cap anomalies is a
   widespread and expensive error.
2. **The only factor with real signal (reversal, IC t = 1.89) is the one whose economic
   rationale — liquidity provision — guarantees it is expensive to harvest.** The
   walk-forward procedure detected that signal correctly and then destroyed the return
   paying for it. Signal quality and signal *cost* are not independent, and an IC-based
   weighting rule that ignores turnover is systematically biased toward the signals it
   can least afford.
3. **Diversification beat selection, decisively.** Equal-weighting all 348 eligible names
   returned 9.24% at Sharpe 0.611, versus 7.02%/0.572 for the 50-stock factor portfolio
   and 8.63%/0.421 for SPY, with far lower turnover. The best portfolio in this entire
   study is the one that makes no forecast at all.

**A defensible negative result is a real result.** The value of this project is the
apparatus that produced it — the point-in-time universe, the measured survivorship
audit, the random-portfolio null, the walk-forward, and the correlation check that
caught a t = 2.38 factor that did not exist. Remove any one of those, and this study
would have reported a "successful" multi-factor strategy with a 3.2% in-sample edge.

---

## 16. Future Improvements

**Fix the data first — it dominates everything else.**

1. **Sharadar SF1 (~\$150/mo)** solves the two biggest limitations at once: `datekey`
   gives true filing dates for point-in-time value/quality, and delisted tickers are
   retained, closing the survivorship hole. This is the single highest-value next step.
2. **CRSP delisting returns via WRDS** for a fully survivorship-free panel with correct
   terminal returns for bankruptcies and mergers.

**Then improve the research design.**

3. **Weight factors by IC-IR *net of turnover*.** The walk-forward's failure mode is
   diagnosable and fixable: penalise each factor's weight by its expected trading cost,
   $w_k \propto \max(IC\text{-}IR_k - \lambda \cdot \text{turnover}_k, 0)$, so a signal must
   pay for the turnover it creates.
4. **Allocate risk across *independent* factors, not factor names.** Cluster the
   correlation matrix (§10.4) and equalise risk across the three real blocks rather than
   the five labels.
5. **Beta- and sector-neutralise.** Given that 76% of variance is factor exposure,
   neutralising market beta and GICS sector would isolate whatever selection skill exists
   — and would have revealed the absence of it far earlier.
6. **Test the universe hypothesis directly.** Re-run on the Russell 2000. If momentum and
   low-vol show efficacy there and not here, that confirms the universe explanation
   rather than a broken implementation.
7. **Model costs properly.** Replace the constant with a square-root impact model,
   $\text{cost} \propto \sigma \sqrt{Q/ADV}$, and make spreads volatility-dependent — the
   current constant flatters a strategy that trades hardest when markets are worst.
8. **Extend to international markets** for genuinely independent out-of-sample evidence,
   which is the only real defence against the collective-data-mining problem in §14.

---

## Appendix A — Complete Python Implementation

All code is in this repository and runs end-to-end from `scripts/01`–`06`.

| Module | Responsibility |
|---|---|
| `src/config.py` | `BacktestConfig` — a run is fully specified by this frozen dataclass |
| `src/data.py` | PIT membership, price download + cache, Fama-French, survivorship audit |
| `src/factors.py` | factor mathematics, vectorised market model, winsorise/z-score/composite |
| `src/fundamentals.py` | PIT schema, value/quality/size/investment maths, synthetic guard |
| `src/portfolio.py` | universe screens, three weighting schemes, iterative position cap |
| `src/backtest.py` | event-timed engine, IC / decile / correlation diagnostics |
| `src/metrics.py` | performance stats, Newey-West, PSR, Deflated Sharpe |
| `src/attribution.py` | CAPM→FF5+MOM regressions with HAC errors, rolling betas |
| `src/robustness.py` | sweeps, regimes, break-even, walk-forward, bootstrap, null test |
| `src/plots.py` | all 14 figures |
| `src/specs.py` | **frozen** specifications with their derivation recorded |
| `tests/test_engine.py` | 19 tests — look-ahead, cost accounting, cap logic, PIT guard |
| `tests/test_inference.py` | 30 tests — attribution, HAC, Deflated Sharpe, controls, frozen specs |

**Test suite: 49 passed; 50% statement coverage of `src/`.** The tests that matter most
are the look-ahead tests (`test_factors_ignore_future_data` asserts factor values at
date *t* are unchanged when all data after *t* is deleted), the cost-accounting identity
(`test_cost_equals_turnover_times_bps`), and the two residual-momentum regression tests
from §11.

The inference tests exist because this layer fails *quietly*: a broken HAC estimator
still returns a plausible t-statistic, and a broken Deflated Sharpe still returns a
number between 0 and 1. Each one therefore checks a value that is known analytically
or an invariant the code itself promises — the regressions are run against a series
built from known alpha and betas and must recover them;
`test_factor_contribution_components_sum_to_total` asserts the decomposition in §10.2
actually adds up; `test_newey_west_shrinks_tstat_under_positive_autocorrelation`
asserts the HAC correction reduces significance on autocorrelated returns rather than
merely being applied; and `test_deflated_sharpe_penalises_more_trials` asserts the
multiple-testing correction gets stricter as the §12.2 sweep grows. Both were verified
by mutation: reverting the annualisation in `attribution.regress` and zeroing the
Newey-West lag terms each make the corresponding test fail.

Uncovered by design: `plots.py` (rendering) and the network paths in `data.py`.

```bash
pip install -r requirements.txt
python scripts/01_download_data.py      # data acquisition (~10 min, cached)
python scripts/02_factor_analysis.py    # IS factor efficacy + survivorship audit
python scripts/03_backtest.py           # headline backtests + attribution
python scripts/04_robustness.py         # controls, sweeps, walk-forward, bootstrap
python scripts/05_figures.py            # all figures
python scripts/06_fundamentals_demo.py  # PIT schema demo + synthetic guard
python -m pytest tests/ -v              # 49 tests
```

## Appendix B — References

- Ang, Hodrick, Xing & Zhang (2006), *The Cross-Section of Volatility and Expected Returns*, JF
- Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, J. Portfolio Management
- Banz (1981), *The Relationship Between Return and Market Value of Common Stocks*, JFE
- Blitz, Huij & Martens (2011), *Residual Momentum*, J. Empirical Finance
- Daniel & Moskowitz (2016), *Momentum Crashes*, JFE
- Fama & French (2015), *A Five-Factor Asset Pricing Model*, JFE
- Frazzini & Pedersen (2014), *Betting Against Beta*, JFE
- Harvey, Liu & Zhu (2016), *...and the Cross-Section of Expected Returns*, RFS
- Jegadeesh (1990), *Evidence of Predictable Behavior of Security Returns*, JF
- Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, JF
- Lehmann (1990), *Fads, Martingales, and Market Efficiency*, QJE
- Newey & West (1987, 1994), HAC covariance estimation
- Novy-Marx (2013), *The Other Side of Value: The Gross Profitability Premium*, JFE
- Politis & Romano (1994), *The Stationary Bootstrap*, JASA
- Sloan (1996), *Do Stock Prices Fully Reflect Information in Accruals...*, Accounting Review
